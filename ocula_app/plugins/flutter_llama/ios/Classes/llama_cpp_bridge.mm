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

// Include llama.cpp headers
#include "../../llama.cpp/include/llama.h"

// Global state
static llama_model* g_model = nullptr;
static llama_context* g_context = nullptr;
static const llama_vocab* g_vocab = nullptr;
static llama_sampler* g_sampler = nullptr;
static std::mutex g_mutex;
static bool g_should_stop = false;
static std::vector<std::string> g_stream_tokens;
static size_t g_stream_pos = 0;
static bool g_backends_loaded = false;
static llama_context* g_embed_ctx = nullptr;  // Embedding context (for RAG)

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
    // Wrap in @autoreleasepool to force immediate Metal/ObjC object deallocation.
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
        if (g_model) {
            llama_model* mdl = g_model;
            g_model = nullptr;
            g_vocab = nullptr;
            llama_model_free(mdl);
        }
    } // @autoreleasepool — Metal buffers released here
    
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
    
    // Create batch
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    
    // Decode prompt
    if (llama_decode(g_context, batch) != 0) {
        NSLog(@"[llama_cpp_bridge] Failed to decode prompt");
        return false;
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
    
    // Create batch
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    
    // Decode prompt
    if (llama_decode(g_context, batch) != 0) {
        NSLog(@"[llama_cpp_bridge] Failed to decode prompt");
        return;
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

// Free model
void llama_cpp_bridge_free_model() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    NSLog(@"[llama_cpp_bridge] Freeing model — g_model=%p g_context=%p g_sampler=%p",
          (void*)g_model, (void*)g_context, (void*)g_sampler);
    
    // Null pointers BEFORE freeing to prevent double-free / use-after-free.
    // Free embedding context first since it shares g_model.
    // Wrap in @autoreleasepool to force immediate Metal/ObjC object deallocation.
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
        
        if (g_model) {
            NSLog(@"[llama_cpp_bridge] Freeing model pointer %p ...", (void*)g_model);
            llama_model* mdl = g_model;
            g_model = nullptr;
            llama_model_free(mdl);
            NSLog(@"[llama_cpp_bridge] Model freed OK");
        }
    } // @autoreleasepool — Metal buffers released here
    
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
    
    // Detect model type once
    bool has_encoder = llama_model_has_encoder(g_model);
    
    // Lazily create embedding context
    if (!g_embed_ctx) {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx       = 512;
        ctx_params.n_batch     = 512;
        ctx_params.n_threads   = 4;
        ctx_params.n_threads_batch = 4;
        ctx_params.embeddings  = true;
        
        // Decoder-only models (SmolVLM, Moondream, Qwen): MUST use POOLING_TYPE_NONE
        // so that KV cache is allocated (MEAN pooling skips KV cache → crash).
        // Encoder models (BERT, etc.): can use MEAN pooling.
        ctx_params.pooling_type = has_encoder
            ? LLAMA_POOLING_TYPE_MEAN
            : LLAMA_POOLING_TYPE_NONE;
        
        g_embed_ctx = llama_init_from_model(g_model, ctx_params);
        if (!g_embed_ctx) {
            NSLog(@"[llama_cpp_bridge] getEmbedding: failed to create embedding context");
            return 0;
        }
        NSLog(@"[llama_cpp_bridge] Embedding context created (n_embd=%d, encoder=%d)",
              llama_model_n_embd(g_model), has_encoder);
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
    
    // Process batch:
    // - Encoder models: llama_encode (no KV cache needed)
    // - Decoder models: llama_decode (uses KV cache)
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    int rc;
    if (has_encoder) {
        rc = llama_encode(g_embed_ctx, batch);
    } else {
        rc = llama_decode(g_embed_ctx, batch);
    }
    if (rc != 0) {
        NSLog(@"[llama_cpp_bridge] getEmbedding: %s failed (rc=%d)",
              has_encoder ? "encode" : "decode", rc);
        return 0;
    }
    
    // Get embedding dimension
    int n_embd = llama_model_n_embd(g_model);
    
    // Encoder models with MEAN pooling → sequence-level embedding.
    // Decoder models with NONE pooling → last-token hidden state.
    const float* embd = nullptr;
    if (has_encoder) {
        embd = llama_get_embeddings_seq(g_embed_ctx, 0);
    }
    if (!embd) {
        embd = llama_get_embeddings_ith(g_embed_ctx, -1);
    }
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
    if (!g_model) return 0;
    return llama_model_n_embd(g_model);
}

} // extern "C"
