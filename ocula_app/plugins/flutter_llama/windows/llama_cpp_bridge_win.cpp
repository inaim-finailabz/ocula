/*
 * Flutter Llama - llama.cpp Bridge for Windows
 *
 * Windows port of ios/Classes/llama_cpp_bridge.mm.
 * No Objective-C, no NSLog, no autoreleasepool.
 * Uses VirtualQuery for heap-pointer validation.
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <string>
#include <vector>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "../../llama.cpp/include/llama.h"
#include "llama_cpp_bridge_win.h"

// ── logging helper ────────────────────────────────────────────────────────────

static void bridge_log(const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    OutputDebugStringA(buf);
    fprintf(stderr, "%s\n", buf);
}

// ── Heap-pointer sanity check ─────────────────────────────────────────────────

static bool is_plausible_heap_ptr(const void* ptr) {
    if (!ptr) return false;
    MEMORY_BASIC_INFORMATION mbi = {};
    if (!VirtualQuery(ptr, &mbi, sizeof(mbi))) return false;
    return mbi.State == MEM_COMMIT &&
           (mbi.Protect & (PAGE_READWRITE | PAGE_WRITECOPY |
                           PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY));
}

// ── Global state: generative model ───────────────────────────────────────────

static llama_model*   g_model      = nullptr;
static llama_context* g_context    = nullptr;
static const llama_vocab* g_vocab  = nullptr;
static llama_sampler* g_sampler    = nullptr;
static std::mutex     g_mutex;
static bool           g_should_stop = false;
static std::vector<std::string> g_stream_tokens;
static size_t         g_stream_pos = 0;
static bool           g_backends_loaded = false;
static llama_context* g_embed_ctx  = nullptr;

// ── Global state: dedicated embedding model ───────────────────────────────────

static llama_model*   g_embed_model     = nullptr;
static llama_context* g_embed_model_ctx = nullptr;
static const llama_vocab* g_embed_vocab = nullptr;
static std::mutex     g_embed_mutex;

extern "C" {

// ─────────────────────────────────────────────────────────────────────────────
// llama_init_model
// ─────────────────────────────────────────────────────────────────────────────

bool llama_init_model(
    const char* model_path,
    int32_t n_threads,
    int32_t n_gpu_layers,
    int32_t context_size,
    int32_t batch_size,
    bool use_gpu,
    bool verbose
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    bridge_log("[llama_cpp_bridge] Initializing model: %s", model_path);
    bridge_log("[llama_cpp_bridge] Threads: %d, GPU layers: %d, Context: %d",
               n_threads, n_gpu_layers, context_size);

    if (g_embed_ctx) {
        llama_context* ectx = g_embed_ctx;
        g_embed_ctx = nullptr;
        llama_free(ectx);
    }
    if (g_sampler) {
        llama_sampler* s = g_sampler;
        g_sampler = nullptr;
        llama_sampler_free(s);
    }
    if (g_context) {
        llama_context* ctx = g_context;
        g_context = nullptr;
        llama_free(ctx);
    }
    if (g_model) {
        llama_model* mdl = g_model;
        g_model = nullptr;
        g_vocab = nullptr;
        if (is_plausible_heap_ptr(mdl)) {
            llama_model_free(mdl);
        } else {
            bridge_log("[llama_cpp_bridge] WARNING: corrupt model pointer %p in init — skipping free",
                       (void*)mdl);
        }
    } else {
        g_vocab = nullptr;
    }

    if (!g_backends_loaded) {
        ggml_backend_load_all();
        g_backends_loaded = true;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = use_gpu ? n_gpu_layers : 0;

    g_model = llama_model_load_from_file(model_path, model_params);
    if (!g_model) {
        bridge_log("[llama_cpp_bridge] Failed to load model from: %s", model_path);
        return false;
    }

    g_vocab = llama_model_get_vocab(g_model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = context_size;
    ctx_params.n_batch  = batch_size;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    g_context = llama_init_from_model(g_model, ctx_params);
    if (!g_context) {
        bridge_log("[llama_cpp_bridge] Context creation failed with n_ctx=%d, retrying with 1024",
                   context_size);
        ctx_params.n_ctx = 1024;
        ctx_params.n_batch = std::min(batch_size, (int32_t)512);
        g_context = llama_init_from_model(g_model, ctx_params);
    }
    if (!g_context) {
        bridge_log("[llama_cpp_bridge] Failed to create context even at 1024");
        llama_model* mdl = g_model;
        g_model = nullptr;
        g_vocab = nullptr;
        llama_model_free(mdl);
        return false;
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    g_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(1234));

    bridge_log("[llama_cpp_bridge] Model loaded successfully");
    bridge_log("[llama_cpp_bridge] Context size: %d", llama_n_ctx(g_context));
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// llama_generate (blocking)
// ─────────────────────────────────────────────────────────────────────────────

bool llama_generate(
    const char* prompt,
    float temperature,
    float top_p,
    int32_t top_k,
    int32_t max_tokens,
    float repeat_penalty,
    char* output,
    int32_t output_size,
    int32_t* tokens_generated
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_model || !g_context || !g_vocab) {
        bridge_log("[llama_cpp_bridge] Model not loaded");
        return false;
    }

    bridge_log("[llama_cpp_bridge] Generating with prompt: %.50s...", prompt);

    llama_memory_clear(llama_get_memory(g_context), true);

    std::string prompt_text(prompt);
    const int n_prompt = -llama_tokenize(g_vocab, prompt_text.c_str(),
                                         (int32_t)prompt_text.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(g_vocab, prompt_text.c_str(), (int32_t)prompt_text.size(),
                       prompt_tokens.data(), (int32_t)prompt_tokens.size(), true, true) < 0) {
        bridge_log("[llama_cpp_bridge] Failed to tokenize prompt");
        return false;
    }

    const int32_t n_batch_size = llama_n_batch(g_context);
    llama_batch batch;
    for (size_t i = 0; i < prompt_tokens.size(); i += n_batch_size) {
        int n_eval = std::min((int)(prompt_tokens.size() - i), (int)n_batch_size);
        batch = llama_batch_get_one(prompt_tokens.data() + i, n_eval);
        if (llama_decode(g_context, batch) != 0) {
            bridge_log("[llama_cpp_bridge] Failed to decode prompt at pos %zu", i);
            return false;
        }
    }

    llama_sampler_free(g_sampler);
    auto sparams = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_penalties(64, repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(1234));

    std::string result;
    int n_gen = 0;
    g_should_stop = false;

    for (int i = 0; i < max_tokens; i++) {
        if (g_should_stop) {
            bridge_log("[llama_cpp_bridge] Generation stopped by user");
            break;
        }

        llama_token new_token = llama_sampler_sample(g_sampler, g_context, -1);

        if (llama_vocab_is_eog(g_vocab, new_token)) {
            bridge_log("[llama_cpp_bridge] EOS token reached");
            break;
        }

        char token_str[256] = {0};
        int n = llama_token_to_piece(g_vocab, new_token, token_str, sizeof(token_str) - 1, 0, true);
        if (n > 0) {
            token_str[n] = '\0';
            result.append(token_str);
        }

        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(g_context, batch) != 0) {
            bridge_log("[llama_cpp_bridge] Failed to decode token");
            break;
        }
        n_gen++;
    }

    size_t copy_len = std::min(result.length(), (size_t)(output_size - 1));
    memcpy(output, result.c_str(), copy_len);
    output[copy_len] = '\0';
    *tokens_generated = n_gen;

    bridge_log("[llama_cpp_bridge] Generated %d tokens", n_gen);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming
// ─────────────────────────────────────────────────────────────────────────────

void llama_generate_stream_init(
    const char* prompt,
    float temperature,
    float top_p,
    int32_t top_k,
    int32_t max_tokens,
    float repeat_penalty
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    bridge_log("[llama_cpp_bridge] Initializing stream generation");

    if (!g_model || !g_context || !g_vocab) {
        bridge_log("[llama_cpp_bridge] Model not loaded");
        return;
    }

    g_should_stop = false;
    g_stream_tokens.clear();
    g_stream_pos = 0;

    llama_memory_clear(llama_get_memory(g_context), true);

    std::string prompt_text(prompt);
    const int n_prompt = -llama_tokenize(g_vocab, prompt_text.c_str(),
                                         (int32_t)prompt_text.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(g_vocab, prompt_text.c_str(), (int32_t)prompt_text.size(),
                       prompt_tokens.data(), (int32_t)prompt_tokens.size(), true, true) < 0) {
        bridge_log("[llama_cpp_bridge] Failed to tokenize prompt");
        return;
    }

    const int32_t n_batch_size = llama_n_batch(g_context);
    llama_batch batch;
    for (size_t i = 0; i < prompt_tokens.size(); i += n_batch_size) {
        int n_eval = std::min((int)(prompt_tokens.size() - i), (int)n_batch_size);
        batch = llama_batch_get_one(prompt_tokens.data() + i, n_eval);
        if (llama_decode(g_context, batch) != 0) {
            bridge_log("[llama_cpp_bridge] Failed to decode prompt at pos %zu", i);
            return;
        }
    }

    if (g_sampler) llama_sampler_free(g_sampler);
    auto sparams = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_penalties(64, repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(1234));

    for (int i = 0; i < max_tokens; i++) {
        if (g_should_stop) break;

        llama_token new_token = llama_sampler_sample(g_sampler, g_context, -1);
        if (llama_vocab_is_eog(g_vocab, new_token)) break;

        char token_str[256] = {0};
        int n = llama_token_to_piece(g_vocab, new_token, token_str, sizeof(token_str) - 1, 0, true);
        if (n > 0) {
            token_str[n] = '\0';
            g_stream_tokens.push_back(std::string(token_str));
        }

        batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(g_context, batch) != 0) break;
    }

    bridge_log("[llama_cpp_bridge] Pre-generated %zu tokens for streaming",
               g_stream_tokens.size());
}

bool llama_generate_stream_next(char* output, int32_t output_size) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_should_stop || g_stream_pos >= g_stream_tokens.size()) return false;

    const std::string& token = g_stream_tokens[g_stream_pos++];
    size_t copy_len = std::min(token.length(), (size_t)(output_size - 1));
    memcpy(output, token.c_str(), copy_len);
    output[copy_len] = '\0';
    return true;
}

void llama_generate_stream_end() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_stream_tokens.clear();
    g_stream_pos = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Model info / free / stop
// ─────────────────────────────────────────────────────────────────────────────

void llama_get_model_info(int64_t* n_params, int32_t* n_layers, int32_t* context_size) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || !g_context) {
        *n_params = 0; *n_layers = 0; *context_size = 0;
        return;
    }
    *n_params     = llama_model_n_params(g_model);
    *n_layers     = llama_model_n_layer(g_model);
    *context_size = llama_n_ctx(g_context);
}

void llama_bridge_free_model() {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_model && !g_context && !g_sampler && !g_embed_ctx) {
        bridge_log("[llama_cpp_bridge] free_model called but nothing loaded — skipping");
        return;
    }

    bridge_log("[llama_cpp_bridge] Freeing model — g_model=%p g_context=%p g_sampler=%p",
               (void*)g_model, (void*)g_context, (void*)g_sampler);

    if (g_embed_ctx) {
        llama_context* ectx = g_embed_ctx;
        g_embed_ctx = nullptr;
        llama_free(ectx);
    }
    if (g_sampler) {
        llama_sampler* s = g_sampler;
        g_sampler = nullptr;
        llama_sampler_free(s);
    }
    if (g_context) {
        llama_context* ctx = g_context;
        g_context = nullptr;
        llama_free(ctx);
    }
    g_vocab = nullptr;

    if (g_model) {
        llama_model* mdl = g_model;
        g_model = nullptr;
        if (!is_plausible_heap_ptr(mdl)) {
            bridge_log("[llama_cpp_bridge] WARNING: corrupt model pointer %p — skipping free",
                       (void*)mdl);
        } else {
            bridge_log("[llama_cpp_bridge] Freeing model pointer %p ...", (void*)mdl);
            llama_model_free(mdl);
            bridge_log("[llama_cpp_bridge] Model freed OK");
        }
    }

    bridge_log("[llama_cpp_bridge] Model freed successfully");
}

void llama_stop_generation() {
    std::lock_guard<std::mutex> lock(g_mutex);
    bridge_log("[llama_cpp_bridge] Stopping generation");
    g_should_stop = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding (generative model context)
// ─────────────────────────────────────────────────────────────────────────────

int32_t llama_get_embedding(const char* text, float* output, int32_t output_size) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_model || !g_vocab) {
        bridge_log("[llama_cpp_bridge] getEmbedding: model not loaded");
        return 0;
    }

    if (!g_embed_ctx) {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx       = 512;
        ctx_params.n_batch     = 512;
        ctx_params.n_threads   = 4;
        ctx_params.n_threads_batch = 4;
        ctx_params.embeddings  = true;
        ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        g_embed_ctx = llama_init_from_model(g_model, ctx_params);
        if (!g_embed_ctx) {
            bridge_log("[llama_cpp_bridge] getEmbedding: failed to create embedding context");
            return 0;
        }
        bridge_log("[llama_cpp_bridge] Embedding context created (n_embd=%d)",
                   llama_model_n_embd(g_model));
    }

    std::string text_str(text);
    const int n_tokens = -llama_tokenize(g_vocab, text_str.c_str(),
                                         (int32_t)text_str.size(), nullptr, 0, true, true);
    if (n_tokens <= 0) return 0;

    int n_use = std::min(n_tokens, (int)llama_n_ctx(g_embed_ctx));
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(g_vocab, text_str.c_str(), (int32_t)text_str.size(),
                   tokens.data(), (int32_t)tokens.size(), true, true);
    if (n_use < n_tokens) tokens.resize(n_use);

    llama_memory_t mem = llama_get_memory(g_embed_ctx);
    if (mem) llama_memory_clear(mem, true);

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    if (llama_decode(g_embed_ctx, batch) != 0) return 0;

    int n_embd = llama_model_n_embd(g_model);
    const float* embd = llama_get_embeddings_ith(g_embed_ctx, -1);
    if (!embd) return 0;

    int copy_n = std::min(n_embd, (int)output_size);
    float norm = 0.0f;
    for (int i = 0; i < copy_n; i++) norm += embd[i] * embd[i];
    norm = sqrtf(norm);
    if (norm > 0.0f) {
        for (int i = 0; i < copy_n; i++) output[i] = embd[i] / norm;
    } else {
        memcpy(output, embd, copy_n * sizeof(float));
    }
    return copy_n;
}

int32_t llama_get_embedding_dim() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model) return 0;
    return llama_model_n_embd(g_model);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dedicated embedding model
// ─────────────────────────────────────────────────────────────────────────────

bool llama_load_embedding_model(const char* model_path) {
    std::lock_guard<std::mutex> lock(g_embed_mutex);

    bridge_log("[llama_cpp_bridge] Loading embedding model: %s", model_path);

    if (g_embed_model_ctx) {
        llama_context* ctx = g_embed_model_ctx;
        g_embed_model_ctx = nullptr;
        llama_free(ctx);
    }
    if (g_embed_model) {
        llama_model* mdl = g_embed_model;
        g_embed_model = nullptr;
        g_embed_vocab = nullptr;
        if (is_plausible_heap_ptr(mdl)) llama_model_free(mdl);
    } else {
        g_embed_vocab = nullptr;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    g_embed_model = llama_model_load_from_file(model_path, model_params);
    if (!g_embed_model) {
        bridge_log("[llama_cpp_bridge] Failed to load embedding model from: %s", model_path);
        return false;
    }

    g_embed_vocab = llama_model_get_vocab(g_embed_model);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx       = 256;
    ctx_params.n_batch     = 256;
    ctx_params.n_threads   = 2;
    ctx_params.n_threads_batch = 2;
    ctx_params.embeddings  = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    g_embed_model_ctx = llama_init_from_model(g_embed_model, ctx_params);
    if (!g_embed_model_ctx) {
        bridge_log("[llama_cpp_bridge] Failed to create embedding model context");
        llama_model* mdl = g_embed_model;
        g_embed_model = nullptr;
        g_embed_vocab = nullptr;
        llama_model_free(mdl);
        return false;
    }

    bridge_log("[llama_cpp_bridge] Embedding model loaded (n_embd=%d)",
               llama_model_n_embd(g_embed_model));
    return true;
}

int32_t llama_get_embedding_v2(const char* text, float* output, int32_t output_size) {
    std::lock_guard<std::mutex> lock(g_embed_mutex);

    if (!g_embed_model || !g_embed_model_ctx || !g_embed_vocab) return 0;

    std::string text_str(text);
    const int n_tokens = -llama_tokenize(g_embed_vocab, text_str.c_str(),
                                         (int32_t)text_str.size(), nullptr, 0, true, true);
    if (n_tokens <= 0) {
        bridge_log("[llama_cpp_bridge] getEmbeddingV2: tokenization failed");
        return 0;
    }

    int n_use = std::min(n_tokens, (int)llama_n_ctx(g_embed_model_ctx));
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(g_embed_vocab, text_str.c_str(), (int32_t)text_str.size(),
                   tokens.data(), (int32_t)tokens.size(), true, true);
    if (n_use < n_tokens) tokens.resize(n_use);

    llama_memory_t mem = llama_get_memory(g_embed_model_ctx);
    if (mem) llama_memory_clear(mem, true);

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    int rc = llama_encode(g_embed_model_ctx, batch);
    if (rc != 0) {
        rc = llama_decode(g_embed_model_ctx, batch);
        if (rc != 0) {
            bridge_log("[llama_cpp_bridge] getEmbeddingV2: encode/decode failed");
            return 0;
        }
    }

    int n_embd = llama_model_n_embd(g_embed_model);
    const float* embd = llama_get_embeddings(g_embed_model_ctx);
    if (!embd) embd = llama_get_embeddings_ith(g_embed_model_ctx, -1);
    if (!embd) {
        bridge_log("[llama_cpp_bridge] getEmbeddingV2: no embeddings available");
        return 0;
    }

    int copy_n = std::min(n_embd, (int)output_size);
    float norm = 0.0f;
    for (int i = 0; i < copy_n; i++) norm += embd[i] * embd[i];
    norm = sqrtf(norm);
    if (norm > 0.0f) {
        for (int i = 0; i < copy_n; i++) output[i] = embd[i] / norm;
    } else {
        memcpy(output, embd, copy_n * sizeof(float));
    }
    return copy_n;
}

void llama_unload_embedding_model() {
    std::lock_guard<std::mutex> lock(g_embed_mutex);

    bridge_log("[llama_cpp_bridge] Unloading embedding model");

    if (g_embed_model_ctx) {
        llama_context* ctx = g_embed_model_ctx;
        g_embed_model_ctx = nullptr;
        llama_free(ctx);
    }
    if (g_embed_model) {
        llama_model* mdl = g_embed_model;
        g_embed_model = nullptr;
        g_embed_vocab = nullptr;
        if (is_plausible_heap_ptr(mdl)) llama_model_free(mdl);
    } else {
        g_embed_vocab = nullptr;
    }

    bridge_log("[llama_cpp_bridge] Embedding model unloaded");
}

bool llama_is_embedding_model_loaded() {
    std::lock_guard<std::mutex> lock(g_embed_mutex);
    return g_embed_model != nullptr && g_embed_model_ctx != nullptr;
}

int32_t llama_get_embedding_model_dim() {
    std::lock_guard<std::mutex> lock(g_embed_mutex);
    if (!g_embed_model) return 0;
    return llama_model_n_embd(g_embed_model);
}

// ─────────────────────────────────────────────────────────────────────────────
// Multimodal stub — returns false/no-op until mtmd is wired up for Windows
// ─────────────────────────────────────────────────────────────────────────────

bool mtmd_bridge_load(const char*, const char*, int32_t, int32_t, int32_t, int32_t, int32_t, bool) {
    bridge_log("[llama_cpp_bridge] mtmd_bridge_load: multimodal not yet supported on Windows");
    return false;
}
bool mtmd_bridge_generate(const char*, const char*, const char*,
                           float, float, int32_t, int32_t, float,
                           char*, int32_t, int32_t*) { return false; }
void mtmd_bridge_stream_init(const char*, const char*, const char*,
                              float, float, int32_t, int32_t, float) {}
bool mtmd_bridge_stream_next(char*, int32_t) { return false; }
void mtmd_bridge_stream_end() {}
void mtmd_bridge_get_info(int64_t* p, int32_t* l, int32_t* c, bool* v, bool* a) {
    *p = 0; *l = 0; *c = 0; *v = false; *a = false;
}
void mtmd_bridge_free() {}
void mtmd_bridge_stop() {}

} // extern "C"
