/*
 * Flutter Llama - llama.cpp Bridge for iOS
 * 
 * This file provides a C++ bridge between Swift and llama.cpp
 * Updated for latest llama.cpp API
 */

#import <Foundation/Foundation.h>
#include <string>
#include <vector>
#include <mutex>
#include <unistd.h>
#include <mach/mach.h>

// Include llama.cpp headers
#include "../../llama.cpp/include/llama.h"

// Global state — generative model
static llama_model* g_model = nullptr;
static llama_context* g_context = nullptr;
static const llama_vocab* g_vocab = nullptr;
static llama_sampler* g_sampler = nullptr;
static std::mutex g_mutex;
static bool g_should_stop = false;
static std::vector<std::string> g_stream_tokens;
static size_t g_stream_pos = 0;
static bool g_backends_loaded = false;
static llama_context* g_embed_ctx = nullptr;  // Embedding context (shares g_model)

// Global state — dedicated embedding model (e.g. all-MiniLM-L6-v2)
static llama_model*   g_embed_model = nullptr;
static llama_context* g_embed_model_ctx = nullptr;
static const llama_vocab* g_embed_vocab = nullptr;
static std::mutex g_embed_mutex;

/// Check whether a pointer looks like it lives in a valid heap region.
/// Catches obviously-bogus values like 0x1 that pass a simple != nullptr check.
static bool is_plausible_heap_ptr(const void* ptr) {
    if (!ptr) return false;
    uintptr_t addr = (uintptr_t)ptr;
    // Heap pointers on arm64 Darwin are page-aligned or at least > 4 KB.
    // Anything below the first page is a corrupt/sentinel value.
    if (addr < 0x1000) return false;
    // Probe with vm_region to confirm the address is mapped.
    vm_address_t region_addr = (vm_address_t)addr;
    vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object_name = MACH_PORT_NULL;
    kern_return_t kr = vm_region_64(mach_task_self(), &region_addr, &region_size,
                                     VM_REGION_BASIC_INFO_64,
                                     (vm_region_info_t)&info, &count, &object_name);
    if (kr != KERN_SUCCESS) return false;
    // The pointer must fall within the returned region.
    return addr >= region_addr && addr < (region_addr + region_size);
}

extern "C" {

// Initialize and load model
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
    
    NSLog(@"[llama_cpp_bridge] Initializing model: %s", model_path);
    NSLog(@"[llama_cpp_bridge] Threads: %d, GPU layers: %d, Context: %d", 
          n_threads, n_gpu_layers, context_size);
    
    // Free existing model if any — null pointers BEFORE freeing to prevent
    // double-free / use-after-free if another thread checks the pointer.
    // Also free embed_ctx first since it shares g_model.
    // Split into two pools so Metal objects drain between context and model frees.
    @autoreleasepool {
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
    } // Pool drains — Metal context objects deallocated

    if (g_model) {
        usleep(500 * 1000); // 500 ms — let Metal GPU fully drain before model free
        @autoreleasepool {
            llama_model* mdl = g_model;
            g_model = nullptr;
            g_vocab = nullptr;
            if (is_plausible_heap_ptr(mdl)) {
                llama_model_free(mdl);
            } else {
                NSLog(@"[llama_cpp_bridge] ⚠ CORRUPT model pointer %p in init — skipping free", (void*)mdl);
            }
        }
    } else {
        g_vocab = nullptr;
    }
    
    // Load dynamic backends (only once per process)
    if (!g_backends_loaded) {
        ggml_backend_load_all();
        g_backends_loaded = true;
    }
    
    // Set up model parameters
    llama_model_params model_params = llama_model_default_params();
#if TARGET_OS_SIMULATOR
    // iOS Simulator has no real Metal GPU — force CPU to avoid
    // backend cleanup crashes in llama_model destructor.
    model_params.n_gpu_layers = 0;
    NSLog(@"[llama_cpp_bridge] Simulator detected — forcing n_gpu_layers=0");
#else
    model_params.n_gpu_layers = use_gpu ? n_gpu_layers : 0;
#endif
    
    // Load model
    g_model = llama_model_load_from_file(model_path, model_params);
    if (!g_model) {
        NSLog(@"[llama_cpp_bridge] Failed to load model from: %s", model_path);
        return false;
    }
    
    // Get vocab
    g_vocab = llama_model_get_vocab(g_model);
    
    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_size;
    ctx_params.n_batch = batch_size;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    
    g_context = llama_init_from_model(g_model, ctx_params);
    if (!g_context) {
        // Context creation failed (likely Metal OOM). Try smaller context.
        NSLog(@"[llama_cpp_bridge] Context creation failed with n_ctx=%d, retrying with 1024", context_size);
        ctx_params.n_ctx = 1024;
        ctx_params.n_batch = std::min(batch_size, (int32_t)512);
        g_context = llama_init_from_model(g_model, ctx_params);
    }
    if (!g_context) {
        NSLog(@"[llama_cpp_bridge] Failed to create context even at 1024");
        llama_model* mdl = g_model;
        g_model = nullptr;
        g_vocab = nullptr;
        llama_model_free(mdl);
        return false;
    }
    
    // Initialize sampler chain
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    g_sampler = llama_sampler_chain_init(sparams);
    
    // Add samplers
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(1234));
    
    NSLog(@"[llama_cpp_bridge] Model loaded successfully");
    NSLog(@"[llama_cpp_bridge] Context size: %d", llama_n_ctx(g_context));
    
    return true;
}

// Generate text
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
        NSLog(@"[llama_cpp_bridge] Model not loaded");
        return false;
    }
    
    NSLog(@"[llama_cpp_bridge] Generating with prompt: %.50s...", prompt);
    
    // Clear KV cache so each call starts fresh
    llama_memory_clear(llama_get_memory(g_context), true);
    
    std::string prompt_text(prompt);
    
    // Tokenize prompt
    const int n_prompt = -llama_tokenize(g_vocab, prompt_text.c_str(), prompt_text.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    
    if (llama_tokenize(g_vocab, prompt_text.c_str(), prompt_text.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        NSLog(@"[llama_cpp_bridge] Failed to tokenize prompt");
        return false;
        }
    
    // Decode prompt in batches (prompt may exceed n_batch)
    const int32_t n_batch_size = llama_n_batch(g_context);
    llama_batch batch;
    for (size_t i = 0; i < prompt_tokens.size(); i += n_batch_size) {
        int n_eval = std::min((int)(prompt_tokens.size() - i), (int)n_batch_size);
        batch = llama_batch_get_one(prompt_tokens.data() + i, n_eval);
        if (llama_decode(g_context, batch) != 0) {
            NSLog(@"[llama_cpp_bridge] Failed to decode prompt at pos %zu", i);
            return false;
        }
    }
    
    // Update sampler with new parameters
    llama_sampler_free(g_sampler);
    
    auto sparams = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_penalties(64, repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(1234));
    
    // Generate tokens
    std::string result;
    int n_gen = 0;
    int n_pos = prompt_tokens.size();
    
    g_should_stop = false;
    
    for (int i = 0; i < max_tokens; i++) {
        if (g_should_stop) {
            NSLog(@"[llama_cpp_bridge] Generation stopped by user");
            break;
        }
        
        // Sample next token
        llama_token new_token = llama_sampler_sample(g_sampler, g_context, -1);
        
        // Check for EOS
        if (llama_vocab_is_eog(g_vocab, new_token)) {
            NSLog(@"[llama_cpp_bridge] EOS token reached");
            break;
        }
        
        // Convert token to text
        char token_str[256] = {0};
        int n = llama_token_to_piece(g_vocab, new_token, token_str, sizeof(token_str) - 1, 0, true);
        if (n > 0) {
            token_str[n] = '\0';
            result.append(token_str);
        }
        
        // Prepare for next iteration
        batch = llama_batch_get_one(&new_token, 1);
        n_pos++;
        
        if (llama_decode(g_context, batch) != 0) {
            NSLog(@"[llama_cpp_bridge] Failed to decode token");
            break;
        }
        
        n_gen++;
    }
    
    // Copy result
    size_t copy_len = std::min(result.length(), (size_t)(output_size - 1));
    memcpy(output, result.c_str(), copy_len);
    output[copy_len] = '\0';
    *tokens_generated = n_gen;
    
    NSLog(@"[llama_cpp_bridge] Generated %d tokens", n_gen);
    return true;
}

// Initialize streaming generation
void llama_generate_stream_init(
    const char* prompt,
    float temperature,
    float top_p,
    int32_t top_k,
    int32_t max_tokens,
    float repeat_penalty
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    NSLog(@"[llama_cpp_bridge] Initializing stream generation");
    
    if (!g_model || !g_context || !g_vocab) {
        NSLog(@"[llama_cpp_bridge] Model not loaded");
        return;
    }
    
    g_should_stop = false;
    g_stream_tokens.clear();
    g_stream_pos = 0;
    
    // Clear KV cache so each call starts fresh
    llama_memory_clear(llama_get_memory(g_context), true);
    
    std::string prompt_text(prompt);
    
    // Tokenize prompt
    const int n_prompt = -llama_tokenize(g_vocab, prompt_text.c_str(), prompt_text.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    
    if (llama_tokenize(g_vocab, prompt_text.c_str(), prompt_text.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        NSLog(@"[llama_cpp_bridge] Failed to tokenize prompt");
        return;
    }
    
    // Decode prompt in batches (prompt may exceed n_batch)
    const int32_t n_batch_size = llama_n_batch(g_context);
    llama_batch batch;
    for (size_t i = 0; i < prompt_tokens.size(); i += n_batch_size) {
        int n_eval = std::min((int)(prompt_tokens.size() - i), (int)n_batch_size);
        batch = llama_batch_get_one(prompt_tokens.data() + i, n_eval);
        if (llama_decode(g_context, batch) != 0) {
            NSLog(@"[llama_cpp_bridge] Failed to decode prompt at pos %zu", i);
            return;
        }
    }
    
    // Update sampler
    if (g_sampler) {
        llama_sampler_free(g_sampler);
    }
    
    auto sparams = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_penalties(64, repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(1234));
    
    // Pre-generate tokens and convert to strings
    int n_pos = prompt_tokens.size();
    for (int i = 0; i < max_tokens; i++) {
        if (g_should_stop) break;
        
        llama_token new_token = llama_sampler_sample(g_sampler, g_context, -1);
        
        if (llama_vocab_is_eog(g_vocab, new_token)) {
            break;
        }
        
        // Convert token to text and store
        char token_str[256] = {0};
        int n = llama_token_to_piece(g_vocab, new_token, token_str, sizeof(token_str) - 1, 0, true);
        if (n > 0) {
            token_str[n] = '\0';
            g_stream_tokens.push_back(std::string(token_str));
        }
        
        batch = llama_batch_get_one(&new_token, 1);
        n_pos++;
        
        if (llama_decode(g_context, batch) != 0) {
            break;
        }
    }
    
    NSLog(@"[llama_cpp_bridge] Pre-generated %zu tokens for streaming", g_stream_tokens.size());
}

// Get next token in stream
bool llama_generate_stream_next(
    char* output,
    int32_t output_size
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (g_should_stop || g_stream_pos >= g_stream_tokens.size()) {
        return false;
    }
    
    const std::string& token = g_stream_tokens[g_stream_pos++];
    
    size_t copy_len = std::min(token.length(), (size_t)(output_size - 1));
    memcpy(output, token.c_str(), copy_len);
    output[copy_len] = '\0';
    
    return true;
}

// End streaming generation
void llama_generate_stream_end() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    NSLog(@"[llama_cpp_bridge] Ending stream generation");
    g_stream_tokens.clear();
    g_stream_pos = 0;
}

// Get model information
void llama_get_model_info(
    int64_t* n_params,
    int32_t* n_layers,
    int32_t* context_size
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (!g_model || !g_context) {
        *n_params = 0;
        *n_layers = 0;
        *context_size = 0;
        return;
    }
    
    *n_params = llama_model_n_params(g_model);
    *n_layers = llama_model_n_layer(g_model);
    *context_size = llama_n_ctx(g_context);
}

// Free model — no-op if nothing is loaded
void llama_cpp_bridge_free_model() {
    std::lock_guard<std::mutex> lock(g_mutex);

    // Guard: don't run any free/drain logic when nothing is loaded
    if (!g_model && !g_context && !g_sampler && !g_embed_ctx) {
        NSLog(@"[llama_cpp_bridge] free_model called but nothing loaded — skipping");
        return;
    }

    NSLog(@"[llama_cpp_bridge] Freeing model — g_model=%p g_context=%p g_sampler=%p",
          (void*)g_model, (void*)g_context, (void*)g_sampler);

    // Null pointers BEFORE freeing to prevent double-free / use-after-free.
    // Free embedding context first since it shares g_model.

    // ── Pool 1: free contexts + sampler ──
    // The @autoreleasepool drain at the closing brace forces Metal/ObjC
    // objects (command buffers, encoders) to be deallocated immediately.
    @autoreleasepool {
        if (g_embed_ctx) {
            NSLog(@"[llama_cpp_bridge] Freeing embedding context...");
            llama_context* ectx = g_embed_ctx;
            g_embed_ctx = nullptr;
            llama_free(ectx);
            NSLog(@"[llama_cpp_bridge] Embedding context freed OK");
        }

        if (g_sampler) {
            NSLog(@"[llama_cpp_bridge] Freeing sampler...");
            llama_sampler* s = g_sampler;
            g_sampler = nullptr;
            llama_sampler_free(s);
            NSLog(@"[llama_cpp_bridge] Sampler freed OK");
        }

        if (g_context) {
            NSLog(@"[llama_cpp_bridge] Freeing context...");
            llama_context* ctx = g_context;
            g_context = nullptr;
            llama_free(ctx);
            NSLog(@"[llama_cpp_bridge] Context freed OK");
        }

        g_vocab = nullptr;
    } // Pool 1 drains here — Metal context objects are deallocated

    // ── Metal sync barrier ──
    // Metal command buffers freed above may still have in-flight GPU work
    // referencing model weight buffers. Give the GPU time to finish before
    // we tear down the model. 50ms covers typical A-series GPU drain.
    usleep(50 * 1000); // 50 ms

    // ── Pool 2: free model ──
    @autoreleasepool {
        if (g_model) {
            llama_model* mdl = g_model;
            g_model = nullptr;

            if (!is_plausible_heap_ptr(mdl)) {
                NSLog(@"[llama_cpp_bridge] ⚠ CORRUPT model pointer %p — skipping free to avoid crash", (void*)mdl);
            } else {
                NSLog(@"[llama_cpp_bridge] Freeing model pointer %p ...", (void*)mdl);
                llama_model_free(mdl);
                NSLog(@"[llama_cpp_bridge] Model freed OK");
            }
        }
    } // Pool 2 drains here — Metal model buffers released

    NSLog(@"[llama_cpp_bridge] Model freed successfully");
}

// Stop generation
void llama_stop_generation() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    NSLog(@"[llama_cpp_bridge] Stopping generation");
    g_should_stop = true;
}

// ──────────────────────────────────────────
// EMBEDDING — for RAG pipeline
// ──────────────────────────────────────────

/// Compute an embedding vector for the given text.
/// Creates a dedicated embedding context lazily (small, 512 tokens).
/// Returns the number of floats written to `output`, or 0 on failure.
int32_t llama_get_embedding(
    const char* text,
    float* output,
    int32_t output_size
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (!g_model || !g_vocab) {
        NSLog(@"[llama_cpp_bridge] getEmbedding: model not loaded");
        return 0;
    }
    
    // Lazily create embedding context.
    // All our models (SmolVLM2, Moondream2, Qwen3VL) are decoder-based for text,
    // even if has_encoder=true (vision encoder). We MUST use:
    //   POOLING_TYPE_NONE  — so KV cache is allocated (MEAN skips it → crash)
    //   llama_decode       — llama_encode is for BERT-style text encoders
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
            NSLog(@"[llama_cpp_bridge] getEmbedding: failed to create embedding context");
            return 0;
        }
        NSLog(@"[llama_cpp_bridge] Embedding context created (n_embd=%d)",
              llama_model_n_embd(g_model));
    }
    
    // Tokenize
    std::string text_str(text);
    const int n_tokens = -llama_tokenize(g_vocab, text_str.c_str(), text_str.size(), NULL, 0, true, true);
    if (n_tokens <= 0) {
        NSLog(@"[llama_cpp_bridge] getEmbedding: tokenization failed");
        return 0;
    }
    
    // Truncate to context size
    int n_use = std::min(n_tokens, (int)llama_n_ctx(g_embed_ctx));
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(g_vocab, text_str.c_str(), text_str.size(), tokens.data(), tokens.size(), true, true);
    if (n_use < n_tokens) {
        tokens.resize(n_use);
    }
    
    // Clear KV cache (only exists for decoder-only contexts with POOLING_TYPE_NONE)
    llama_memory_t mem = llama_get_memory(g_embed_ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }
    
    // Always use llama_decode for text — our models are all decoder-based
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    if (llama_decode(g_embed_ctx, batch) != 0) {
        NSLog(@"[llama_cpp_bridge] getEmbedding: decode failed");
        return 0;
    }
    
    // Get embedding dimension
    int n_embd = llama_model_n_embd(g_model);
    
    // With POOLING_TYPE_NONE, use last-token hidden state as embedding
    const float* embd = llama_get_embeddings_ith(g_embed_ctx, -1);
    if (!embd) {
        NSLog(@"[llama_cpp_bridge] getEmbedding: no embeddings available");
        return 0;
    }
    
    // Copy and L2-normalize
    int copy_n = std::min(n_embd, (int)output_size);
    float norm = 0.0f;
    for (int i = 0; i < copy_n; i++) {
        norm += embd[i] * embd[i];
    }
    norm = sqrtf(norm);
    if (norm > 0.0f) {
        for (int i = 0; i < copy_n; i++) {
            output[i] = embd[i] / norm;
        }
    } else {
        memcpy(output, embd, copy_n * sizeof(float));
    }
    
    return copy_n;
}

/// Get the embedding dimension of the currently loaded model.
int32_t llama_get_embedding_dim() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model) return 0;
    return llama_model_n_embd(g_model);
}

// ──────────────────────────────────────────
// DEDICATED EMBEDDING MODEL — for high-quality RAG
// ──────────────────────────────────────────
// Loads a separate GGUF model (e.g. all-MiniLM-L6-v2) specifically
// trained for sentence similarity. Independent from the generative model.

/// Load a dedicated embedding model from a GGUF file.
/// Uses BERT-style mean pooling and a small context (256 tokens).
/// Returns true on success.
bool llama_load_embedding_model(const char* model_path) {
    std::lock_guard<std::mutex> lock(g_embed_mutex);

    NSLog(@"[llama_cpp_bridge] Loading embedding model: %s", model_path);

    // Free existing embedding model if any
    @autoreleasepool {
        if (g_embed_model_ctx) {
            llama_context* ctx = g_embed_model_ctx;
            g_embed_model_ctx = nullptr;
            llama_free(ctx);
        }
    }
    if (g_embed_model) {
        @autoreleasepool {
            llama_model* mdl = g_embed_model;
            g_embed_model = nullptr;
            g_embed_vocab = nullptr;
            if (is_plausible_heap_ptr(mdl)) {
                llama_model_free(mdl);
            }
        }
    } else {
        g_embed_vocab = nullptr;
    }

    // Load model — CPU only to avoid competing with generative model for Metal
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU-only — embedding model is tiny

    g_embed_model = llama_model_load_from_file(model_path, model_params);
    if (!g_embed_model) {
        NSLog(@"[llama_cpp_bridge] Failed to load embedding model from: %s", model_path);
        return false;
    }

    g_embed_vocab = llama_model_get_vocab(g_embed_model);

    // Create embedding context with BERT-style mean pooling
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx       = 256;  // MiniLM max sequence length
    ctx_params.n_batch     = 256;
    ctx_params.n_threads   = 2;    // Lightweight — don't compete with generation
    ctx_params.n_threads_batch = 2;
    ctx_params.embeddings  = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN; // BERT-style mean pooling

    g_embed_model_ctx = llama_init_from_model(g_embed_model, ctx_params);
    if (!g_embed_model_ctx) {
        NSLog(@"[llama_cpp_bridge] Failed to create embedding model context");
        llama_model* mdl = g_embed_model;
        g_embed_model = nullptr;
        g_embed_vocab = nullptr;
        llama_model_free(mdl);
        return false;
    }

    int n_embd = llama_model_n_embd(g_embed_model);
    NSLog(@"[llama_cpp_bridge] Embedding model loaded (n_embd=%d)", n_embd);
    return true;
}

/// Compute an embedding using the dedicated embedding model.
/// Uses llama_encode (BERT-style encoder) with mean pooling.
/// Returns the number of floats written, or 0 on failure.
int32_t llama_get_embedding_v2(
    const char* text,
    float* output,
    int32_t output_size
) {
    std::lock_guard<std::mutex> lock(g_embed_mutex);

    if (!g_embed_model || !g_embed_model_ctx || !g_embed_vocab) {
        return 0;
    }

    // Tokenize
    std::string text_str(text);
    const int n_tokens = -llama_tokenize(g_embed_vocab, text_str.c_str(), text_str.size(), NULL, 0, true, true);
    if (n_tokens <= 0) {
        NSLog(@"[llama_cpp_bridge] getEmbeddingV2: tokenization failed");
        return 0;
    }

    int n_use = std::min(n_tokens, (int)llama_n_ctx(g_embed_model_ctx));
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(g_embed_vocab, text_str.c_str(), text_str.size(), tokens.data(), tokens.size(), true, true);
    if (n_use < n_tokens) {
        tokens.resize(n_use);
    }

    // Clear memory
    llama_memory_t mem = llama_get_memory(g_embed_model_ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    // Use llama_encode for BERT-style encoder models (mean pooling)
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    int rc = llama_encode(g_embed_model_ctx, batch);
    if (rc != 0) {
        // Fallback to llama_decode if encode isn't supported
        rc = llama_decode(g_embed_model_ctx, batch);
        if (rc != 0) {
            NSLog(@"[llama_cpp_bridge] getEmbeddingV2: encode/decode failed");
            return 0;
        }
    }

    // Get pooled embedding (mean pooling → single vector for the whole sequence)
    int n_embd = llama_model_n_embd(g_embed_model);
    const float* embd = llama_get_embeddings(g_embed_model_ctx);
    if (!embd) {
        // Fallback: try getting last token embedding
        embd = llama_get_embeddings_ith(g_embed_model_ctx, -1);
    }
    if (!embd) {
        NSLog(@"[llama_cpp_bridge] getEmbeddingV2: no embeddings available");
        return 0;
    }

    // Copy and L2-normalize
    int copy_n = std::min(n_embd, (int)output_size);
    float norm = 0.0f;
    for (int i = 0; i < copy_n; i++) {
        norm += embd[i] * embd[i];
    }
    norm = sqrtf(norm);
    if (norm > 0.0f) {
        for (int i = 0; i < copy_n; i++) {
            output[i] = embd[i] / norm;
        }
    } else {
        memcpy(output, embd, copy_n * sizeof(float));
    }

    return copy_n;
}

/// Unload the dedicated embedding model and free all associated memory.
void llama_unload_embedding_model() {
    std::lock_guard<std::mutex> lock(g_embed_mutex);

    NSLog(@"[llama_cpp_bridge] Unloading embedding model");

    @autoreleasepool {
        if (g_embed_model_ctx) {
            llama_context* ctx = g_embed_model_ctx;
            g_embed_model_ctx = nullptr;
            llama_free(ctx);
        }
    }

    if (g_embed_model) {
        @autoreleasepool {
            llama_model* mdl = g_embed_model;
            g_embed_model = nullptr;
            g_embed_vocab = nullptr;
            if (is_plausible_heap_ptr(mdl)) {
                llama_model_free(mdl);
            }
        }
    } else {
        g_embed_vocab = nullptr;
    }

    NSLog(@"[llama_cpp_bridge] Embedding model unloaded");
}

/// Check if a dedicated embedding model is loaded.
bool llama_is_embedding_model_loaded() {
    std::lock_guard<std::mutex> lock(g_embed_mutex);
    return g_embed_model != nullptr && g_embed_model_ctx != nullptr;
}

/// Get the embedding dimension of the dedicated embedding model.
int32_t llama_get_embedding_model_dim() {
    std::lock_guard<std::mutex> lock(g_embed_mutex);
    if (!g_embed_model) return 0;
    return llama_model_n_embd(g_embed_model);
}

} // extern "C"
