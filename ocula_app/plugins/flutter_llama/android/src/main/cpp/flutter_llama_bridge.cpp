/*
 * Flutter Llama - JNI Bridge for Android
 * 
 * This file provides JNI bindings between Kotlin and llama.cpp
 * Updated for latest llama.cpp API
 */

#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include <algorithm>
#include <cctype>
#include <android/log.h>

#define LOG_TAG "FlutterLlamaBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Include llama.cpp headers
#include "llama.h"

// Global state
static llama_model* g_model = nullptr;
static llama_context* g_context = nullptr;
static const llama_vocab* g_vocab = nullptr;
static llama_sampler* g_sampler = nullptr;
static std::mutex g_mutex;
static bool g_should_stop = false;
static std::vector<std::string> g_stream_tokens;
static size_t g_stream_pos = 0;
static llama_context* g_embed_ctx = nullptr;  // Embedding context (for RAG)
static std::string g_active_backend = "cpu";
static bool g_backends_loaded = false;

static std::string to_lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static bool str_contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

static ggml_backend_dev_t find_gpu_device_for_backend(const std::string& requested_backend) {
    const size_t n_devices = ggml_backend_dev_count();
    ggml_backend_dev_t first_gpu = nullptr;

    for (size_t i = 0; i < n_devices; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }

        const auto dev_type = ggml_backend_dev_type(dev);
        if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU && dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const std::string reg_name = to_lower_copy(reg && ggml_backend_reg_name(reg) ? ggml_backend_reg_name(reg) : "");
        const std::string dev_name = to_lower_copy(ggml_backend_dev_name(dev) ? ggml_backend_dev_name(dev) : "");

        if (!first_gpu) {
            first_gpu = dev;
        }

        if (requested_backend == "opencl" && (str_contains(reg_name, "opencl") || str_contains(dev_name, "opencl"))) {
            return dev;
        }
        if (requested_backend == "vulkan" && (str_contains(reg_name, "vulkan") || str_contains(dev_name, "vulkan"))) {
            return dev;
        }
    }

    return requested_backend == "auto" ? first_gpu : nullptr;
}

static std::string backend_name_for_device(ggml_backend_dev_t dev) {
    if (!dev) {
        return "cpu";
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char* reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
    if (!reg_name || reg_name[0] == '\0') {
        return "gpu";
    }
    return to_lower_copy(reg_name);
}

extern "C" {

// Initialize and load model
JNIEXPORT jboolean JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaPlugin_nativeInitModel(
    JNIEnv* env,
    jobject thiz,
    jstring model_path,
    jint n_threads,
    jint n_gpu_layers,
    jint context_size,
    jint batch_size,
    jstring preferred_backend,
    jboolean use_gpu,
    jboolean verbose
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    const char* path = env->GetStringUTFChars(model_path, nullptr);
    const char* preferred_backend_c = env->GetStringUTFChars(preferred_backend, nullptr);
    const std::string requested_backend = to_lower_copy(preferred_backend_c ? preferred_backend_c : "auto");
    
    LOGI("Initializing model: %s", path);
    LOGI("Threads: %d, GPU layers: %d, Context: %d", n_threads, n_gpu_layers, context_size);
    LOGI("Requested backend: %s, use_gpu=%d", requested_backend.c_str(), (int) use_gpu);
    
    // Free existing model if any
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }
    
    // Load backends once. On CPU fallback we intentionally avoid loading
    // GPU backends (Vulkan/OpenCL), which are unstable on Android emulators.
    if (!g_backends_loaded) {
        if (use_gpu) {
            ggml_backend_load_all();
        } else {
            LOGI("GPU disabled — loading CPU backend only");
        }
        g_backends_loaded = true;
    }
    
    // Set up model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = use_gpu ? n_gpu_layers : 0;

    std::vector<ggml_backend_dev_t> selected_devices;
    ggml_backend_dev_t cpu_devices[2] = { nullptr, nullptr };
    if (use_gpu && requested_backend != "cpu") {
        ggml_backend_dev_t selected_dev = find_gpu_device_for_backend(requested_backend);
        if (selected_dev) {
            selected_devices.push_back(selected_dev);
            selected_devices.push_back(nullptr); // Null-terminated list required by llama_model_params
            model_params.devices = selected_devices.data();
            g_active_backend = backend_name_for_device(selected_dev);
            LOGI("Selected GPU backend: %s", g_active_backend.c_str());
        } else {
            // Requested GPU backend is unavailable: fallback to CPU to avoid init crashes.
            model_params.n_gpu_layers = 0;
            model_params.devices = nullptr;
            g_active_backend = "cpu";
            LOGI("Requested backend unavailable, falling back to CPU");
        }
    } else {
        model_params.n_gpu_layers = 0;
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev) {
            cpu_devices[0] = cpu_dev;
            model_params.devices = cpu_devices;
            LOGI("Restricting to CPU-only device: %s", ggml_backend_dev_name(cpu_dev));
        } else {
            model_params.devices = nullptr;
            LOGI("CPU device not found, using default backend selection");
        }
        g_active_backend = "cpu";
    }
    
    // Load model
    g_model = llama_model_load_from_file(path, model_params);
    if (!g_model) {
        LOGE("Failed to load model from: %s", path);
        env->ReleaseStringUTFChars(preferred_backend, preferred_backend_c);
        env->ReleaseStringUTFChars(model_path, path);
        return JNI_FALSE;
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
        LOGE("Failed to create context");
        llama_free_model(g_model);
        g_model = nullptr;
        env->ReleaseStringUTFChars(preferred_backend, preferred_backend_c);
        env->ReleaseStringUTFChars(model_path, path);
        return JNI_FALSE;
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
    
    LOGI("Model loaded successfully");
    LOGI("Context size: %d", llama_n_ctx(g_context));
    
    env->ReleaseStringUTFChars(preferred_backend, preferred_backend_c);
    env->ReleaseStringUTFChars(model_path, path);
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaPlugin_nativeGetActiveBackend(
    JNIEnv* env,
    jobject thiz
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return env->NewStringUTF(g_active_backend.c_str());
}

// Generate text
JNIEXPORT jobject JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaPlugin_nativeGenerate(
    JNIEnv* env,
    jobject thiz,
    jstring prompt,
    jfloat temperature,
    jfloat top_p,
    jint top_k,
    jint max_tokens,
    jfloat repeat_penalty
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (!g_model || !g_context || !g_vocab) {
        LOGE("Model not loaded");
        return nullptr;
    }
    
    const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);
    LOGI("Generating with prompt: %.50s...", prompt_str);
    
    std::string prompt_text(prompt_str);
    env->ReleaseStringUTFChars(prompt, prompt_str);
    
    // Tokenize prompt
    const int n_prompt = -llama_tokenize(g_vocab, prompt_text.c_str(), prompt_text.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    
    if (llama_tokenize(g_vocab, prompt_text.c_str(), prompt_text.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        LOGE("Failed to tokenize prompt");
        return nullptr;
    }
    
    // Decode prompt in batches (prompt may exceed n_batch)
    const int32_t n_batch_size = llama_n_batch(g_context);
    llama_batch batch;
    for (size_t i = 0; i < prompt_tokens.size(); i += n_batch_size) {
        int n_eval = std::min((int)(prompt_tokens.size() - i), (int)n_batch_size);
        batch = llama_batch_get_one(prompt_tokens.data() + i, n_eval);
        if (llama_decode(g_context, batch) != 0) {
            LOGE("Failed to decode prompt at pos %zu", i);
            return nullptr;
        }
    }
    
    // Update sampler with new parameters
    llama_sampler_free(g_sampler);
    
    auto sparams = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(1234));
    
    // Generate tokens
    std::string result;
    int n_generated = 0;
    int n_pos = prompt_tokens.size();
    
    g_should_stop = false;
    
    for (int i = 0; i < max_tokens; i++) {
        if (g_should_stop) {
            LOGI("Generation stopped by user");
            break;
        }
        
        // Sample next token
        llama_token new_token = llama_sampler_sample(g_sampler, g_context, -1);
        
        // Check for EOS
        if (llama_vocab_is_eog(g_vocab, new_token)) {
            LOGI("EOS token reached");
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
            LOGE("Failed to decode token");
            break;
        }
        
        n_generated++;
    }
    
    LOGI("Generated %d tokens", n_generated);
    
    // Create GenerationResult object
    jclass result_class = env->FindClass("net/nativemind/flutter_llama/FlutterLlamaPlugin$GenerationResult");
    if (!result_class) {
        LOGE("Failed to find GenerationResult class");
        return nullptr;
    }
    
    jmethodID constructor = env->GetMethodID(result_class, "<init>", "(Ljava/lang/String;I)V");
    if (!constructor) {
        LOGE("Failed to find GenerationResult constructor");
        return nullptr;
    }
    
    jstring j_result = env->NewStringUTF(result.c_str());
    jobject generation_result = env->NewObject(result_class, constructor, j_result, n_generated);
    
    return generation_result;
}

// Initialize streaming generation
JNIEXPORT void JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaPlugin_nativeGenerateStreamInit(
    JNIEnv* env,
    jobject thiz,
    jstring prompt,
    jfloat temperature,
    jfloat top_p,
    jint top_k,
    jint max_tokens,
    jfloat repeat_penalty
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    LOGI("Initializing stream generation");
    
    if (!g_model || !g_context || !g_vocab) {
        LOGE("Model not loaded");
        return;
    }
    
    g_should_stop = false;
    g_stream_tokens.clear();
    g_stream_pos = 0;
    
    const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);
    std::string prompt_text(prompt_str);
    env->ReleaseStringUTFChars(prompt, prompt_str);
    
    // Tokenize prompt
    const int n_prompt = -llama_tokenize(g_vocab, prompt_text.c_str(), prompt_text.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    
    if (llama_tokenize(g_vocab, prompt_text.c_str(), prompt_text.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        LOGE("Failed to tokenize prompt");
        return;
    }
    
    // Decode prompt in batches (prompt may exceed n_batch)
    const int32_t n_batch_size = llama_n_batch(g_context);
    llama_batch batch;
    for (size_t i = 0; i < prompt_tokens.size(); i += n_batch_size) {
        int n_eval = std::min((int)(prompt_tokens.size() - i), (int)n_batch_size);
        batch = llama_batch_get_one(prompt_tokens.data() + i, n_eval);
        if (llama_decode(g_context, batch) != 0) {
            LOGE("Failed to decode prompt at pos %zu", i);
            return;
        }
    }
    
    // Update sampler
    if (g_sampler) {
        llama_sampler_free(g_sampler);
    }
    
    auto sparams = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sparams);
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
    
    LOGI("Pre-generated %zu tokens for streaming", g_stream_tokens.size());
}

// Get next token in stream
JNIEXPORT jstring JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaPlugin_nativeGenerateStreamNext(
    JNIEnv* env,
    jobject thiz
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (g_should_stop || g_stream_pos >= g_stream_tokens.size()) {
        return nullptr;
    }
    
    const std::string& token = g_stream_tokens[g_stream_pos++];
    return env->NewStringUTF(token.c_str());
}

// End streaming generation
JNIEXPORT void JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaPlugin_nativeGenerateStreamEnd(
    JNIEnv* env,
    jobject thiz
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    LOGI("Ending stream generation");
    g_stream_tokens.clear();
    g_stream_pos = 0;
}

// Get model information
JNIEXPORT jobject JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaPlugin_nativeGetModelInfo(
    JNIEnv* env,
    jobject thiz
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (!g_model || !g_context) {
        return nullptr;
    }
    
    jlong n_params = llama_model_n_params(g_model);
    jint n_layers = llama_model_n_layer(g_model);
    jint context_size = llama_n_ctx(g_context);
    
    LOGI("Model info: params=%lld, layers=%d, context=%d", 
         (long long)n_params, n_layers, context_size);
    
    // Create ModelInfo object
    jclass info_class = env->FindClass("net/nativemind/flutter_llama/FlutterLlamaPlugin$ModelInfo");
    if (!info_class) {
        LOGE("Failed to find ModelInfo class");
        return nullptr;
    }
    
    jmethodID constructor = env->GetMethodID(info_class, "<init>", "(JII)V");
    if (!constructor) {
        LOGE("Failed to find ModelInfo constructor");
        return nullptr;
    }
    
    jobject model_info = env->NewObject(info_class, constructor, n_params, n_layers, context_size);
    return model_info;
}

// Free model — no-op if nothing is loaded
JNIEXPORT void JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaPlugin_nativeFreeModel(
    JNIEnv* env,
    jobject thiz
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    // Guard: don't run any free logic when nothing is loaded
    if (!g_model && !g_context && !g_sampler && !g_embed_ctx) {
        LOGI("free_model called but nothing loaded — skipping");
        g_active_backend = "cpu";
        return;
    }
    
    LOGI("Freeing model — g_model=%p g_context=%p g_sampler=%p",
         (void*)g_model, (void*)g_context, (void*)g_sampler);
    
    // Null pointers BEFORE freeing to prevent use-after-free
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
        llama_free_model(mdl);
    } else {
        g_vocab = nullptr;
    }
    g_active_backend = "cpu";
    
    LOGI("Model freed successfully");
}

// Stop generation
JNIEXPORT void JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaPlugin_nativeStopGeneration(
    JNIEnv* env,
    jobject thiz
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    LOGI("Stopping generation");
    g_should_stop = true;
}

// ──────────────────────────────────────────
// EMBEDDING — for RAG pipeline
// ──────────────────────────────────────────

JNIEXPORT jfloatArray JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaPlugin_nativeGetEmbedding(
    JNIEnv* env,
    jobject thiz,
    jstring text
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (!g_model || !g_vocab) {
        LOGE("getEmbedding: model not loaded");
        return nullptr;
    }
    
    // All our models are decoder-based for text (even VLMs with vision encoders).
    // POOLING_TYPE_NONE ensures KV cache is allocated. llama_decode for decoder path.
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
            LOGE("getEmbedding: failed to create embedding context");
            return nullptr;
        }
        LOGI("Embedding context created (n_embd=%d)", llama_model_n_embd(g_model));
    }
    
    const char* text_str = env->GetStringUTFChars(text, nullptr);
    std::string text_cpp(text_str);
    env->ReleaseStringUTFChars(text, text_str);
    
    // Tokenize
    const int n_tokens = -llama_tokenize(g_vocab, text_cpp.c_str(), text_cpp.size(), NULL, 0, true, true);
    if (n_tokens <= 0) {
        LOGE("getEmbedding: tokenization failed");
        return nullptr;
    }
    
    int n_use = std::min(n_tokens, (int)llama_n_ctx(g_embed_ctx));
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(g_vocab, text_cpp.c_str(), text_cpp.size(), tokens.data(), tokens.size(), true, true);
    if (n_use < n_tokens) tokens.resize(n_use);
    
    // Clear KV cache (only exists for decoder-only contexts with POOLING_TYPE_NONE)
    llama_memory_t mem = llama_get_memory(g_embed_ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }
    
    // Always use llama_decode for text — our models are all decoder-based
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    if (llama_decode(g_embed_ctx, batch) != 0) {
        LOGE("getEmbedding: decode failed");
        return nullptr;
    }
    
    int n_embd = llama_model_n_embd(g_model);
    
    // With POOLING_TYPE_NONE, use last-token hidden state as embedding
    const float* embd = llama_get_embeddings_ith(g_embed_ctx, -1);
    if (!embd) {
        LOGE("getEmbedding: no embeddings available");
        return nullptr;
    }
    
    // L2-normalize
    std::vector<float> normalized(n_embd);
    float norm = 0.0f;
    for (int i = 0; i < n_embd; i++) norm += embd[i] * embd[i];
    norm = sqrtf(norm);
    if (norm > 0.0f) {
        for (int i = 0; i < n_embd; i++) normalized[i] = embd[i] / norm;
    } else {
        memcpy(normalized.data(), embd, n_embd * sizeof(float));
    }
    
    // Return as Java float array
    jfloatArray result = env->NewFloatArray(n_embd);
    env->SetFloatArrayRegion(result, 0, n_embd, normalized.data());
    
    LOGI("getEmbedding: returned %d-dim embedding", n_embd);
    return result;
}

JNIEXPORT jint JNICALL
Java_net_nativemind_flutter_1llama_FlutterLlamaPlugin_nativeGetEmbeddingDim(
    JNIEnv* env,
    jobject thiz
) {
    if (!g_model) return 0;
    return llama_model_n_embd(g_model);
}

} // extern "C"
