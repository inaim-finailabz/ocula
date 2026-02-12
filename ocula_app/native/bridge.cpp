#include <cstdint>
#include <cstring>
#include <string>

// Forward declarations for llama.cpp types
struct llama_model;
struct llama_context;
struct clip_ctx;

// Global state for the currently loaded model
static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static clip_ctx* g_clip = nullptr;

extern "C" {

/// Load a GGUF model from the given file path.
/// Returns 0 on success, -1 on failure.
int32_t load_model(const char* model_path, const char* projector_path, int32_t n_threads) {
    // Unload any existing model first
    if (g_model != nullptr) {
        unload_model();
    }

    // TODO: Initialize llama.cpp backend
    // llama_backend_init();
    //
    // llama_model_params model_params = llama_model_default_params();
    // g_model = llama_load_model_from_file(model_path, model_params);
    // if (!g_model) return -1;
    //
    // llama_context_params ctx_params = llama_context_default_params();
    // ctx_params.n_threads = n_threads;
    // g_ctx = llama_new_context_with_model(g_model, ctx_params);
    // if (!g_ctx) return -1;
    //
    // if (projector_path) {
    //     g_clip = clip_model_load(projector_path, 1);
    //     if (!g_clip) return -1;
    // }

    return 0;
}

/// Unload the current model and free all memory.
void unload_model() {
    // if (g_clip) { clip_free(g_clip); g_clip = nullptr; }
    // if (g_ctx)  { llama_free(g_ctx);  g_ctx = nullptr;  }
    // if (g_model){ llama_free_model(g_model); g_model = nullptr; }
    // llama_backend_free();

    g_clip = nullptr;
    g_ctx = nullptr;
    g_model = nullptr;
}

/// Run inference on an image with a text prompt.
/// Writes the result into `output_buf` (up to `output_buf_len` bytes).
/// Returns the number of bytes written, or -1 on error.
int32_t infer_image(
    const uint8_t* image_data,
    int32_t image_len,
    const char* prompt,
    char* output_buf,
    int32_t output_buf_len
) {
    if (!g_model || !g_ctx) return -1;

    // TODO: Implement vision inference pipeline
    // 1. Decode image bytes via clip
    // 2. Embed image tokens
    // 3. Tokenize text prompt
    // 4. Run llama_decode in a loop
    // 5. Sample tokens and write to output_buf

    const char* placeholder = "Model inference not yet implemented";
    int32_t len = static_cast<int32_t>(strlen(placeholder));
    if (len >= output_buf_len) len = output_buf_len - 1;
    memcpy(output_buf, placeholder, len);
    output_buf[len] = '\0';
    return len;
}

/// Check if a model is currently loaded.
int32_t is_model_loaded() {
    return (g_model != nullptr) ? 1 : 0;
}

} // extern "C"
