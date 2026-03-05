// Flutter Llama - Windows C bridge declarations
// Mirror of ios/Classes/llama_cpp_bridge.h

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

// ── Generative model ─────────────────────────────────────────────────────────

bool llama_init_model(
    const char* model_path,
    int32_t n_threads,
    int32_t n_gpu_layers,
    int32_t context_size,
    int32_t batch_size,
    bool use_gpu,
    bool verbose
);

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
);

void llama_generate_stream_init(
    const char* prompt,
    float temperature,
    float top_p,
    int32_t top_k,
    int32_t max_tokens,
    float repeat_penalty
);

bool llama_generate_stream_next(char* output, int32_t output_size);
void llama_generate_stream_end();

void llama_get_model_info(
    int64_t* n_params,
    int32_t* n_layers,
    int32_t* context_size
);

void llama_bridge_free_model();
void llama_stop_generation();

// ── Embedding (generative model context) ─────────────────────────────────────

int32_t llama_get_embedding(
    const char* text,
    float* output,
    int32_t output_size
);

int32_t llama_get_embedding_dim();

// ── Dedicated embedding model ─────────────────────────────────────────────────

bool llama_load_embedding_model(const char* model_path);

int32_t llama_get_embedding_v2(
    const char* text,
    float* output,
    int32_t output_size
);

void llama_unload_embedding_model();
bool llama_is_embedding_model_loaded();
int32_t llama_get_embedding_model_dim();

// ── Multimodal (mtmd) ─────────────────────────────────────────────────────────

bool mtmd_bridge_load(
    const char* model_path,
    const char* mmproj_path,
    int32_t n_threads,
    int32_t n_gpu_layers,
    int32_t context_size,
    int32_t batch_size,
    int32_t image_min_tokens,
    bool use_gpu
);

bool mtmd_bridge_generate(
    const char* prompt,
    const char* image_path,
    const char* audio_path,
    float temperature,
    float top_p,
    int32_t top_k,
    int32_t max_tokens,
    float repeat_penalty,
    char* output,
    int32_t output_size,
    int32_t* tokens_generated
);

void mtmd_bridge_stream_init(
    const char* prompt,
    const char* image_path,
    const char* audio_path,
    float temperature,
    float top_p,
    int32_t top_k,
    int32_t max_tokens,
    float repeat_penalty
);

bool mtmd_bridge_stream_next(char* output, int32_t output_size);
void mtmd_bridge_stream_end();

void mtmd_bridge_get_info(
    int64_t* n_params,
    int32_t* n_layers,
    int32_t* context_size,
    bool* supports_vision,
    bool* supports_audio
);

void mtmd_bridge_free();
void mtmd_bridge_stop();

#ifdef __cplusplus
}
#endif
