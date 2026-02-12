/*
 * Flutter Llama - Multimodal JNI Bridge for Android
 *
 * Port of llama_multimodal_bridge.mm (iOS/macOS) to Android JNI.
 * Wraps the llama.cpp mtmd (multimodal) API to provide image+text
 * inference through a Flutter method channel.
 *
 * This bridge maintains its OWN separate model/context state,
 * independent of the text-only bridge in flutter_llama_bridge.cpp.
 */

#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include <cstring>
#include <android/log.h>

#define LOG_TAG "FlutterLlamaMultimodal"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// llama.cpp core headers
#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// mtmd multimodal headers
#include "mtmd.h"
#include "mtmd-helper.h"

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

// Helper to get a C string from jstring (caller must release)
static const char* jstring_to_cstr(JNIEnv* env, jstring str) {
    if (!str) return nullptr;
    return env->GetStringUTFChars(str, nullptr);
}

static void release_cstr(JNIEnv* env, jstring str, const char* cstr) {
    if (str && cstr) env->ReleaseStringUTFChars(str, cstr);
}

// ---------------------------------------------------------------------------
// JNI Methods — called from Kotlin FlutterLlamaMultimodalPlugin
// ---------------------------------------------------------------------------
extern "C" {

// ---- Load multimodal model ------------------------------------------------
JNIEXPORT jboolean JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaMultimodalPlugin_nativeLoadMultimodalModel(
    JNIEnv* env,
    jobject thiz,
    jstring model_path,
    jstring mmproj_path,
    jint    n_threads,
    jint    n_gpu_layers,
    jint    context_size,
    jint    batch_size,
    jboolean use_gpu
) {
    std::lock_guard<std::mutex> lock(mm_mutex);

    const char* path = env->GetStringUTFChars(model_path, nullptr);
    const char* mmproj = env->GetStringUTFChars(mmproj_path, nullptr);

    LOGI("Loading model: %s", path);
    LOGI("MMProj: %s", mmproj);

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
    model_params.n_gpu_layers = use_gpu ? n_gpu_layers : 0;

    mm_model = llama_model_load_from_file(path, model_params);
    if (!mm_model) {
        LOGE("Failed to load text model");
        env->ReleaseStringUTFChars(model_path, path);
        env->ReleaseStringUTFChars(mmproj_path, mmproj);
        return JNI_FALSE;
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
        LOGE("Failed to create context");
        llama_model_free(mm_model); mm_model = nullptr;
        env->ReleaseStringUTFChars(model_path, path);
        env->ReleaseStringUTFChars(mmproj_path, mmproj);
        return JNI_FALSE;
    }

    // --- Initialise mtmd (vision / audio projector) ---
    struct mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu   = use_gpu;
    mtmd_params.n_threads = n_threads;

    mm_mtmd = mtmd_init_from_file(mmproj, mm_model, mtmd_params);
    if (!mm_mtmd) {
        LOGE("Failed to initialise mtmd projector");
        llama_free(mm_context);    mm_context = nullptr;
        llama_model_free(mm_model); mm_model   = nullptr;
        mm_vocab = nullptr;
        env->ReleaseStringUTFChars(model_path, path);
        env->ReleaseStringUTFChars(mmproj_path, mmproj);
        return JNI_FALSE;
    }

    // Default sampler
    mm_reset_sampler(0.8f, 0.95f, 40, 1.1f);

    LOGI("Multimodal model loaded OK. vision=%d audio=%d",
         mtmd_support_vision(mm_mtmd), mtmd_support_audio(mm_mtmd));

    env->ReleaseStringUTFChars(model_path, path);
    env->ReleaseStringUTFChars(mmproj_path, mmproj);
    return JNI_TRUE;
}

// ---- Generate (blocking) with image/audio ---------------------------------
JNIEXPORT jobject JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaMultimodalPlugin_nativeGenerateMultimodal(
    JNIEnv* env,
    jobject thiz,
    jstring j_prompt,
    jstring j_image_path,
    jstring j_audio_path,
    jfloat  temperature,
    jfloat  top_p,
    jint    top_k,
    jint    max_tokens,
    jfloat  repeat_penalty
) {
    std::lock_guard<std::mutex> lock(mm_mutex);

    if (!mm_model || !mm_context || !mm_mtmd || !mm_vocab) {
        LOGE("Model not loaded");
        return nullptr;
    }

    const char* prompt     = jstring_to_cstr(env, j_prompt);
    const char* image_path = jstring_to_cstr(env, j_image_path);
    const char* audio_path = jstring_to_cstr(env, j_audio_path);

    mm_should_stop = false;

    // Clear KV cache
    llama_memory_clear(llama_get_memory(mm_context), true);

    // --- Build bitmap list -------------------------------------------------
    std::vector<mtmd_bitmap *> bitmaps;

    if (image_path && strlen(image_path) > 0) {
        mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_file(mm_mtmd, image_path);
        if (!bmp) {
            LOGE("Failed to load image: %s", image_path);
            release_cstr(env, j_prompt, prompt);
            release_cstr(env, j_image_path, image_path);
            release_cstr(env, j_audio_path, audio_path);
            return nullptr;
        }
        bitmaps.push_back(bmp);
        LOGI("Image loaded: %ux%u",
             mtmd_bitmap_get_nx(bmp), mtmd_bitmap_get_ny(bmp));
    }

    if (audio_path && strlen(audio_path) > 0) {
        mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_file(mm_mtmd, audio_path);
        if (!bmp) {
            LOGE("Failed to load audio: %s", audio_path);
            for (auto *b : bitmaps) mtmd_bitmap_free(b);
            release_cstr(env, j_prompt, prompt);
            release_cstr(env, j_image_path, image_path);
            release_cstr(env, j_audio_path, audio_path);
            return nullptr;
        }
        bitmaps.push_back(bmp);
        LOGI("Audio loaded");
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
        LOGE("mtmd_tokenize failed (%d)", tok_ret);
        mtmd_input_chunks_free(chunks);
        for (auto *b : bitmaps) mtmd_bitmap_free(b);
        release_cstr(env, j_prompt, prompt);
        release_cstr(env, j_image_path, image_path);
        release_cstr(env, j_audio_path, audio_path);
        return nullptr;
    }

    // --- Evaluate all chunks (text + encoded images) -----------------------
    llama_pos n_past = 0;
    int32_t eval_ret = mtmd_helper_eval_chunks(
        mm_mtmd, mm_context, chunks, n_past, 0,
        llama_n_batch(mm_context), true, &n_past);

    mtmd_input_chunks_free(chunks);
    for (auto *b : bitmaps) mtmd_bitmap_free(b);

    if (eval_ret != 0) {
        LOGE("mtmd_helper_eval_chunks failed (%d)", eval_ret);
        release_cstr(env, j_prompt, prompt);
        release_cstr(env, j_image_path, image_path);
        release_cstr(env, j_audio_path, audio_path);
        return nullptr;
    }

    // --- Autoregressive generation -----------------------------------------
    mm_reset_sampler(temperature, top_p, top_k, repeat_penalty);

    std::string result;
    int n_gen = 0;

    for (int i = 0; i < max_tokens && !mm_should_stop; i++) {
        llama_token new_token = llama_sampler_sample(mm_sampler, mm_context, -1);

        if (llama_vocab_is_eog(mm_vocab, new_token)) {
            LOGI("EOS at token %d", i);
            break;
        }

        char buf[256] = {0};
        int n = llama_token_to_piece(mm_vocab, new_token, buf, sizeof(buf) - 1, 0, true);
        if (n > 0) { buf[n] = '\0'; result.append(buf); }

        llama_batch batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(mm_context, batch) != 0) {
            LOGE("decode failed at token %d", i);
            break;
        }
        n_gen++;
    }

    release_cstr(env, j_prompt, prompt);
    release_cstr(env, j_image_path, image_path);
    release_cstr(env, j_audio_path, audio_path);

    LOGI("Generated %d tokens", n_gen);

    // Create MultimodalGenerationResult object
    jclass result_class = env->FindClass(
        "net/nativemind/flutter_llama/FlutterLlamaMultimodalPlugin$MultimodalGenerationResult");
    if (!result_class) {
        LOGE("Failed to find MultimodalGenerationResult class");
        return nullptr;
    }

    jmethodID constructor = env->GetMethodID(result_class, "<init>", "(Ljava/lang/String;I)V");
    if (!constructor) {
        LOGE("Failed to find MultimodalGenerationResult constructor");
        return nullptr;
    }

    jstring j_result = env->NewStringUTF(result.c_str());
    return env->NewObject(result_class, constructor, j_result, n_gen);
}

// ---- Stream: init ---------------------------------------------------------
JNIEXPORT void JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaMultimodalPlugin_nativeStreamInit(
    JNIEnv* env,
    jobject thiz,
    jstring j_prompt,
    jstring j_image_path,
    jstring j_audio_path,
    jfloat  temperature,
    jfloat  top_p,
    jint    top_k,
    jint    max_tokens,
    jfloat  repeat_penalty
) {
    std::lock_guard<std::mutex> lock(mm_mutex);

    if (!mm_model || !mm_context || !mm_mtmd || !mm_vocab) {
        LOGE("Model not loaded (stream)");
        return;
    }

    const char* prompt     = jstring_to_cstr(env, j_prompt);
    const char* image_path = jstring_to_cstr(env, j_image_path);
    const char* audio_path = jstring_to_cstr(env, j_audio_path);

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
        LOGE("stream tokenize failed (%d)", tok);
        mtmd_input_chunks_free(chunks);
        for (auto *b : bitmaps) mtmd_bitmap_free(b);
        release_cstr(env, j_prompt, prompt);
        release_cstr(env, j_image_path, image_path);
        release_cstr(env, j_audio_path, audio_path);
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
        LOGE("stream eval failed (%d)", eval);
        release_cstr(env, j_prompt, prompt);
        release_cstr(env, j_image_path, image_path);
        release_cstr(env, j_audio_path, audio_path);
        return;
    }

    // Generate all tokens into buffer
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

    release_cstr(env, j_prompt, prompt);
    release_cstr(env, j_image_path, image_path);
    release_cstr(env, j_audio_path, audio_path);

    LOGI("Pre-generated %zu tokens for streaming", mm_stream_tokens.size());
}

// ---- Stream: next token ---------------------------------------------------
JNIEXPORT jstring JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaMultimodalPlugin_nativeStreamNext(
    JNIEnv* env,
    jobject thiz
) {
    std::lock_guard<std::mutex> lock(mm_mutex);
    if (mm_should_stop || mm_stream_pos >= mm_stream_tokens.size()) return nullptr;
    const std::string &tok = mm_stream_tokens[mm_stream_pos++];
    return env->NewStringUTF(tok.c_str());
}

// ---- Stream: end ----------------------------------------------------------
JNIEXPORT void JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaMultimodalPlugin_nativeStreamEnd(
    JNIEnv* env,
    jobject thiz
) {
    std::lock_guard<std::mutex> lock(mm_mutex);
    LOGI("Ending stream generation");
    mm_stream_tokens.clear();
    mm_stream_pos = 0;
}

// ---- Model info -----------------------------------------------------------
JNIEXPORT jobject JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaMultimodalPlugin_nativeGetMultimodalModelInfo(
    JNIEnv* env,
    jobject thiz
) {
    std::lock_guard<std::mutex> lock(mm_mutex);

    if (!mm_model || !mm_context) return nullptr;

    jlong  n_params     = llama_model_n_params(mm_model);
    jint   n_layers     = llama_model_n_layer(mm_model);
    jint   context_size = llama_n_ctx(mm_context);
    jboolean supports_vision = mm_mtmd ? mtmd_support_vision(mm_mtmd) : false;
    jboolean supports_audio  = mm_mtmd ? mtmd_support_audio(mm_mtmd) : false;

    LOGI("Model info: params=%lld, layers=%d, ctx=%d, vision=%d, audio=%d",
         (long long)n_params, n_layers, context_size, supports_vision, supports_audio);

    jclass info_class = env->FindClass(
        "net/nativemind/flutter_llama/FlutterLlamaMultimodalPlugin$MultimodalModelInfo");
    if (!info_class) {
        LOGE("Failed to find MultimodalModelInfo class");
        return nullptr;
    }

    jmethodID constructor = env->GetMethodID(info_class, "<init>", "(JIIZZ)V");
    if (!constructor) {
        LOGE("Failed to find MultimodalModelInfo constructor");
        return nullptr;
    }

    return env->NewObject(info_class, constructor,
                          n_params, n_layers, context_size,
                          supports_vision, supports_audio);
}

// ---- Free everything — no-op if nothing loaded --------------------------
JNIEXPORT void JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaMultimodalPlugin_nativeFreeMultimodalModel(
    JNIEnv* env,
    jobject thiz
) {
    std::lock_guard<std::mutex> lock(mm_mutex);

    // Guard: don't run free logic when nothing is loaded
    if (!mm_model && !mm_context && !mm_sampler && !mm_mtmd) {
        LOGI("free multimodal called but nothing loaded — skipping");
        return;
    }

    LOGI("Freeing multimodal model — mm_model=%p mm_context=%p", (void*)mm_model, (void*)mm_context);

    // Null pointers BEFORE freeing to prevent use-after-free
    if (mm_sampler)  { llama_sampler* s = mm_sampler; mm_sampler = nullptr; llama_sampler_free(s); }
    if (mm_mtmd)     { mtmd_context* m  = mm_mtmd;    mm_mtmd    = nullptr; mtmd_free(m); }
    if (mm_context)  { llama_context* c = mm_context;  mm_context = nullptr; llama_free(c); }
    if (mm_model)    { llama_model* mdl = mm_model;    mm_model   = nullptr; llama_model_free(mdl); }
    mm_vocab = nullptr;

    mm_stream_tokens.clear();
    mm_stream_pos = 0;

    LOGI("Multimodal model freed OK");
}

// ---- Stop generation ------------------------------------------------------
JNIEXPORT void JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaMultimodalPlugin_nativeStopMultimodalGeneration(
    JNIEnv* env,
    jobject thiz
) {
    std::lock_guard<std::mutex> lock(mm_mutex);
    mm_should_stop = true;
}

} // extern "C"
