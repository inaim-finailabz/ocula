/*
 * Flutter Llama - Multimodal Bridge for iOS/macOS
 *
 * Wraps the llama.cpp mtmd (multimodal) API to provide image+text
 * inference through a Flutter method channel.
 *
 * This bridge maintains its OWN separate model/context state,
 * independent of the text-only bridge in llama_cpp_bridge.mm.
 */

#import <Foundation/Foundation.h>
#include <string>
#include <vector>
#include <mutex>
#include <cstring>

// llama.cpp core headers
#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// mtmd multimodal headers
#include "mtmd.h"
#include "mtmd-helper.h"

// Compatibility shim: llama_model_n_embd_inp was added after the xcframework
// was built.  For standard models (no deepstack) it equals llama_model_n_embd.
extern "C" int32_t llama_model_n_embd(const struct llama_model *);
extern "C" int32_t llama_model_n_embd_inp(const struct llama_model *model) __attribute__((weak));
extern "C" int32_t llama_model_n_embd_inp(const struct llama_model *model) {
    return llama_model_n_embd(model);
}

// ---------------------------------------------------------------------------
// Global state — separate from the text-only bridge
// ---------------------------------------------------------------------------
static llama_model   *mm_model   = nullptr;
static llama_context *mm_context = nullptr;
static const llama_vocab *mm_vocab = nullptr;
static llama_sampler *mm_sampler = nullptr;
static mtmd_context  *mm_mtmd    = nullptr;

static std::mutex     mm_mutex;
static bool           mm_should_stop  = false;
static bool           mm_backends_loaded = false;

// Stream state
static std::vector<std::string> mm_stream_tokens;
static size_t                   mm_stream_pos = 0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void mm_reset_sampler(float temperature, float top_p, int32_t top_k,
                             float repeat_penalty) {
    if (mm_sampler) {
        llama_sampler_free(mm_sampler);
        mm_sampler = nullptr;
    }
    auto sparams = llama_sampler_chain_default_params();
    mm_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(mm_sampler,
        llama_sampler_init_penalties(64, repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(mm_sampler,
        llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(mm_sampler,
        llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(mm_sampler,
        llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(mm_sampler,
        llama_sampler_init_dist(1234));
}

// ---------------------------------------------------------------------------
// Public C API — called from Swift via @_silgen_name
// ---------------------------------------------------------------------------
extern "C" {

// ---- Load multimodal model ------------------------------------------------
bool mtmd_bridge_load(const char *model_path,
                      const char *mmproj_path,
                      int32_t     n_threads,
                      int32_t     n_gpu_layers,
                      int32_t     context_size,
                      int32_t     batch_size,
                      int32_t     image_min_tokens,
                      bool        use_gpu) {
    std::lock_guard<std::mutex> lock(mm_mutex);
    NSLog(@"[mtmd_bridge] Loading model: %s", model_path);
    NSLog(@"[mtmd_bridge] MMProj: %s", mmproj_path);

    // Tear down previous session
    if (mm_sampler)  { llama_sampler_free(mm_sampler); mm_sampler = nullptr; }
    if (mm_mtmd)     { mtmd_free(mm_mtmd);             mm_mtmd    = nullptr; }
    if (mm_context)  { llama_free(mm_context);          mm_context = nullptr; }
    if (mm_model)    { llama_model_free(mm_model);      mm_model   = nullptr; }
    mm_vocab = nullptr;

    // Load backends once
    if (!mm_backends_loaded) {
        ggml_backend_load_all();
        mm_backends_loaded = true;
    }

    // --- Load text model ---
    llama_model_params model_params = llama_model_default_params();
#if TARGET_OS_SIMULATOR
    model_params.n_gpu_layers = 0;
    NSLog(@"[mtmd_bridge] Simulator — forcing n_gpu_layers=0");
#else
    model_params.n_gpu_layers = use_gpu ? n_gpu_layers : 0;
#endif

    mm_model = llama_model_load_from_file(model_path, model_params);
    if (!mm_model) {
        NSLog(@"[mtmd_bridge] Failed to load text model");
        return false;
    }
    mm_vocab = llama_model_get_vocab(mm_model);

    // --- Create context ---
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx         = context_size;
    ctx_params.n_batch       = batch_size;
    ctx_params.n_threads     = n_threads;
    ctx_params.n_threads_batch = n_threads;

    mm_context = llama_init_from_model(mm_model, ctx_params);
    if (!mm_context) {
        NSLog(@"[mtmd_bridge] Failed to create context");
        llama_model_free(mm_model); mm_model = nullptr;
        return false;
    }

    // --- Initialise mtmd (vision / audio projector) ---
    struct mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu   = use_gpu;
    mtmd_params.n_threads = n_threads;
    mtmd_params.image_min_tokens = image_min_tokens;
#if TARGET_OS_SIMULATOR
    mtmd_params.use_gpu = false;
#endif

    mm_mtmd = mtmd_init_from_file(mmproj_path, mm_model, mtmd_params);
    if (!mm_mtmd) {
        NSLog(@"[mtmd_bridge] Failed to initialise mtmd projector");
        llama_free(mm_context);    mm_context = nullptr;
        llama_model_free(mm_model); mm_model   = nullptr;
        mm_vocab = nullptr;
        return false;
    }

    // Default sampler
    mm_reset_sampler(0.8f, 0.95f, 40, 1.1f);

    NSLog(@"[mtmd_bridge] Multimodal model loaded OK. vision=%d audio=%d",
          mtmd_support_vision(mm_mtmd), mtmd_support_audio(mm_mtmd));
    if (image_min_tokens > 0) {
        NSLog(@"[mtmd_bridge] image_min_tokens=%d", image_min_tokens);
    }
    return true;
}

// ---- Generate (blocking) with image/audio ---------------------------------
bool mtmd_bridge_generate(const char *prompt,
                          const char *image_path,   // nullable
                          const char *audio_path,   // nullable
                          float       temperature,
                          float       top_p,
                          int32_t     top_k,
                          int32_t     max_tokens,
                          float       repeat_penalty,
                          char       *output,
                          int32_t     output_size,
                          int32_t    *tokens_generated) {
    std::lock_guard<std::mutex> lock(mm_mutex);

    if (!mm_model || !mm_context || !mm_mtmd || !mm_vocab) {
        NSLog(@"[mtmd_bridge] Model not loaded");
        return false;
    }

    mm_should_stop = false;

    // Clear KV cache
    llama_memory_clear(llama_get_memory(mm_context), true);

    // --- Build bitmap list -------------------------------------------------
    std::vector<mtmd_bitmap *> bitmaps;   // raw ptrs, freed at end

    if (image_path && strlen(image_path) > 0) {
        mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_file(mm_mtmd, image_path);
        if (!bmp) {
            NSLog(@"[mtmd_bridge] Failed to load image: %s", image_path);
            return false;
        }
        bitmaps.push_back(bmp);
        NSLog(@"[mtmd_bridge] Image loaded: %ux%u",
              mtmd_bitmap_get_nx(bmp), mtmd_bitmap_get_ny(bmp));
    }

    if (audio_path && strlen(audio_path) > 0) {
        mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_file(mm_mtmd, audio_path);
        if (!bmp) {
            NSLog(@"[mtmd_bridge] Failed to load audio: %s", audio_path);
            for (auto *b : bitmaps) mtmd_bitmap_free(b);
            return false;
        }
        bitmaps.push_back(bmp);
        NSLog(@"[mtmd_bridge] Audio loaded");
    }

    // --- Tokenize (text + media placeholders) ------------------------------
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    mtmd_input_text input_text;
    input_text.text         = prompt;
    input_text.add_special  = true;
    input_text.parse_special = true;

    std::vector<const mtmd_bitmap *> bitmaps_cptr(bitmaps.size());
    for (size_t i = 0; i < bitmaps.size(); i++) bitmaps_cptr[i] = bitmaps[i];

    int32_t tok_ret = mtmd_tokenize(mm_mtmd, chunks, &input_text,
                                    bitmaps_cptr.data(), bitmaps_cptr.size());
    if (tok_ret != 0) {
        NSLog(@"[mtmd_bridge] mtmd_tokenize failed (%d)", tok_ret);
        mtmd_input_chunks_free(chunks);
        for (auto *b : bitmaps) mtmd_bitmap_free(b);
        return false;
    }

    // --- Evaluate all chunks (text + encoded images) -----------------------
    llama_pos n_past = 0;
    int32_t eval_ret = mtmd_helper_eval_chunks(
        mm_mtmd, mm_context, chunks, n_past, 0,
        llama_n_batch(mm_context), true, &n_past);

    mtmd_input_chunks_free(chunks);
    for (auto *b : bitmaps) mtmd_bitmap_free(b);

    if (eval_ret != 0) {
        NSLog(@"[mtmd_bridge] mtmd_helper_eval_chunks failed (%d)", eval_ret);
        return false;
    }

    // --- Autoregressive generation -----------------------------------------
    mm_reset_sampler(temperature, top_p, top_k, repeat_penalty);

    std::string result;
    int n_gen = 0;

    for (int i = 0; i < max_tokens && !mm_should_stop; i++) {
        llama_token new_token = llama_sampler_sample(mm_sampler, mm_context, -1);

        if (llama_vocab_is_eog(mm_vocab, new_token)) {
            NSLog(@"[mtmd_bridge] EOS at token %d", i);
            break;
        }

        char buf[256] = {0};
        int n = llama_token_to_piece(mm_vocab, new_token, buf, sizeof(buf) - 1, 0, true);
        if (n > 0) { buf[n] = '\0'; result.append(buf); }

        llama_batch batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(mm_context, batch) != 0) {
            NSLog(@"[mtmd_bridge] decode failed at token %d", i);
            break;
        }
        n_gen++;
    }

    size_t copy_len = std::min(result.length(), (size_t)(output_size - 1));
    memcpy(output, result.c_str(), copy_len);
    output[copy_len] = '\0';
    *tokens_generated = n_gen;

    NSLog(@"[mtmd_bridge] Generated %d tokens", n_gen);
    return true;
}

// ---- Stream: init ---------------------------------------------------------
void mtmd_bridge_stream_init(const char *prompt,
                             const char *image_path,
                             const char *audio_path,
                             float       temperature,
                             float       top_p,
                             int32_t     top_k,
                             int32_t     max_tokens,
                             float       repeat_penalty) {
    std::lock_guard<std::mutex> lock(mm_mutex);

    if (!mm_model || !mm_context || !mm_mtmd || !mm_vocab) {
        NSLog(@"[mtmd_bridge] Model not loaded (stream)");
        return;
    }

    mm_should_stop = false;
    mm_stream_tokens.clear();
    mm_stream_pos = 0;

    // Clear KV cache
    llama_memory_clear(llama_get_memory(mm_context), true);

    // Build bitmaps
    std::vector<mtmd_bitmap *> bitmaps;
    if (image_path && strlen(image_path) > 0) {
        mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_file(mm_mtmd, image_path);
        if (bmp) bitmaps.push_back(bmp);
    }
    if (audio_path && strlen(audio_path) > 0) {
        mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_file(mm_mtmd, audio_path);
        if (bmp) bitmaps.push_back(bmp);
    }

    // Tokenize
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    mtmd_input_text input_text;
    input_text.text         = prompt;
    input_text.add_special  = true;
    input_text.parse_special = true;

    std::vector<const mtmd_bitmap *> cptrs(bitmaps.size());
    for (size_t i = 0; i < bitmaps.size(); i++) cptrs[i] = bitmaps[i];

    int32_t tok = mtmd_tokenize(mm_mtmd, chunks, &input_text,
                                cptrs.data(), cptrs.size());
    if (tok != 0) {
        NSLog(@"[mtmd_bridge] stream tokenize failed (%d)", tok);
        mtmd_input_chunks_free(chunks);
        for (auto *b : bitmaps) mtmd_bitmap_free(b);
        return;
    }

    // Eval chunks
    llama_pos n_past = 0;
    int32_t eval = mtmd_helper_eval_chunks(
        mm_mtmd, mm_context, chunks, n_past, 0,
        llama_n_batch(mm_context), true, &n_past);

    mtmd_input_chunks_free(chunks);
    for (auto *b : bitmaps) mtmd_bitmap_free(b);

    if (eval != 0) {
        NSLog(@"[mtmd_bridge] stream eval failed (%d)", eval);
        return;
    }

    // Generate all tokens into buffer (same pattern as text bridge)
    mm_reset_sampler(temperature, top_p, top_k, repeat_penalty);

    for (int i = 0; i < max_tokens && !mm_should_stop; i++) {
        llama_token new_token = llama_sampler_sample(mm_sampler, mm_context, -1);
        if (llama_vocab_is_eog(mm_vocab, new_token)) break;

        char buf[256] = {0};
        int n = llama_token_to_piece(mm_vocab, new_token, buf, sizeof(buf) - 1, 0, true);
        if (n > 0) { buf[n] = '\0'; mm_stream_tokens.emplace_back(buf); }

        llama_batch batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(mm_context, batch) != 0) break;
    }

    NSLog(@"[mtmd_bridge] Pre-generated %zu tokens for streaming",
          mm_stream_tokens.size());
}

// ---- Stream: next token ---------------------------------------------------
bool mtmd_bridge_stream_next(char *output, int32_t output_size) {
    std::lock_guard<std::mutex> lock(mm_mutex);
    if (mm_should_stop || mm_stream_pos >= mm_stream_tokens.size()) return false;
    const std::string &tok = mm_stream_tokens[mm_stream_pos++];
    size_t len = std::min(tok.length(), (size_t)(output_size - 1));
    memcpy(output, tok.c_str(), len);
    output[len] = '\0';
    return true;
}

// ---- Stream: end ----------------------------------------------------------
void mtmd_bridge_stream_end() {
    std::lock_guard<std::mutex> lock(mm_mutex);
    mm_stream_tokens.clear();
    mm_stream_pos = 0;
}

// ---- Model info -----------------------------------------------------------
void mtmd_bridge_get_info(int64_t *n_params, int32_t *n_layers,
                          int32_t *context_size,
                          bool *supports_vision, bool *supports_audio) {
    std::lock_guard<std::mutex> lock(mm_mutex);
    if (!mm_model || !mm_context) {
        *n_params = 0; *n_layers = 0; *context_size = 0;
        *supports_vision = false; *supports_audio = false;
        return;
    }
    *n_params     = llama_model_n_params(mm_model);
    *n_layers     = llama_model_n_layer(mm_model);
    *context_size = llama_n_ctx(mm_context);
    *supports_vision = mm_mtmd ? mtmd_support_vision(mm_mtmd) : false;
    *supports_audio  = mm_mtmd ? mtmd_support_audio(mm_mtmd) : false;
}

// ---- Free everything ------------------------------------------------------
void mtmd_bridge_free() {
    std::lock_guard<std::mutex> lock(mm_mutex);
    NSLog(@"[mtmd_bridge] Freeing multimodal model");

    if (mm_sampler)  { llama_sampler_free(mm_sampler); mm_sampler = nullptr; }
    if (mm_mtmd)     { mtmd_free(mm_mtmd);             mm_mtmd    = nullptr; }
    if (mm_context)  { llama_free(mm_context);          mm_context = nullptr; }
    if (mm_model)    { llama_model_free(mm_model);      mm_model   = nullptr; }
    mm_vocab = nullptr;

    mm_stream_tokens.clear();
    mm_stream_pos = 0;

    NSLog(@"[mtmd_bridge] Multimodal model freed OK");
}

// ---- Stop generation ------------------------------------------------------
void mtmd_bridge_stop() {
    std::lock_guard<std::mutex> lock(mm_mutex);
    mm_should_stop = true;
}

} // extern "C"
