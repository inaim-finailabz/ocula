// Flutter Llama - Windows Plugin
// Implements the same method channel as the iOS/macOS Swift plugin.

#include "flutter_llama_plugin.h"
#include "llama_cpp_bridge_win.h"

#include <flutter/method_channel.h>
#include <flutter/event_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>
#include <flutter/event_stream_handler_functions.h>

#include <windows.h>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <mutex>
#include <functional>

// ─── helpers ─────────────────────────────────────────────────────────────────

static std::string GetStr(const flutter::EncodableMap& m, const std::string& key,
                           const std::string& def = "") {
    auto it = m.find(flutter::EncodableValue(key));
    if (it == m.end()) return def;
    if (auto* v = std::get_if<std::string>(&it->second)) return *v;
    return def;
}

static int GetInt(const flutter::EncodableMap& m, const std::string& key, int def = 0) {
    auto it = m.find(flutter::EncodableValue(key));
    if (it == m.end()) return def;
    if (auto* v = std::get_if<int>(&it->second)) return *v;
    return def;
}

static double GetDouble(const flutter::EncodableMap& m, const std::string& key, double def = 0.0) {
    auto it = m.find(flutter::EncodableValue(key));
    if (it == m.end()) return def;
    if (auto* v = std::get_if<double>(&it->second)) return *v;
    return def;
}

static bool GetBool(const flutter::EncodableMap& m, const std::string& key, bool def = false) {
    auto it = m.find(flutter::EncodableValue(key));
    if (it == m.end()) return def;
    if (auto* v = std::get_if<bool>(&it->second)) return *v;
    return def;
}

// ─── Registration ─────────────────────────────────────────────────────────────

// static
void FlutterLlamaPlugin::RegisterWithRegistrar(flutter::PluginRegistrarWindows* registrar) {
    auto plugin = std::make_unique<FlutterLlamaPlugin>();
    FlutterLlamaPlugin* plugin_ptr = plugin.get();

    // Method channel
    auto method_channel = std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
        registrar->messenger(), "flutter_llama",
        &flutter::StandardMethodCodec::GetInstance());

    method_channel->SetMethodCallHandler(
        [plugin_raw = plugin_ptr](const auto& call, auto result) {
            plugin_raw->HandleMethodCall(call, std::move(result));
        });

    // Event channel for streaming tokens
    auto event_channel = std::make_unique<flutter::EventChannel<flutter::EncodableValue>>(
        registrar->messenger(), "flutter_llama/stream",
        &flutter::StandardMethodCodec::GetInstance());

    auto handler = std::make_unique<flutter::StreamHandlerFunctions<flutter::EncodableValue>>(
        [plugin_raw = plugin_ptr](const flutter::EncodableValue* args,
                                   std::unique_ptr<flutter::EventSink<flutter::EncodableValue>>&& sink)
            -> std::unique_ptr<flutter::StreamHandlerError<flutter::EncodableValue>> {
            std::lock_guard<std::mutex> lk(plugin_raw->stream_sink_mutex_);
            plugin_raw->stream_sink_ = std::move(sink);
            return nullptr;
        },
        [plugin_raw = plugin_ptr](const flutter::EncodableValue* args)
            -> std::unique_ptr<flutter::StreamHandlerError<flutter::EncodableValue>> {
            std::lock_guard<std::mutex> lk(plugin_raw->stream_sink_mutex_);
            plugin_raw->stream_sink_ = nullptr;
            return nullptr;
        });

    event_channel->SetStreamHandler(std::move(handler));

    registrar->AddPlugin(std::move(plugin));
}

// ─── Constructor / Destructor ─────────────────────────────────────────────────

FlutterLlamaPlugin::FlutterLlamaPlugin() {}
FlutterLlamaPlugin::~FlutterLlamaPlugin() {}

// ─── Stream helpers ───────────────────────────────────────────────────────────

void FlutterLlamaPlugin::SendStreamToken(const std::string& token) {
    std::lock_guard<std::mutex> lk(stream_sink_mutex_);
    if (stream_sink_) {
        stream_sink_->Success(flutter::EncodableValue(token));
    }
}

void FlutterLlamaPlugin::SendStreamEnd() {
    std::lock_guard<std::mutex> lk(stream_sink_mutex_);
    if (stream_sink_) {
        stream_sink_->EndOfStream();
    }
}

void FlutterLlamaPlugin::SendStreamError(const std::string& code, const std::string& message) {
    std::lock_guard<std::mutex> lk(stream_sink_mutex_);
    if (stream_sink_) {
        stream_sink_->Error(code, message);
    }
}

// ─── Method handler ───────────────────────────────────────────────────────────

void FlutterLlamaPlugin::HandleMethodCall(
    const flutter::MethodCall<flutter::EncodableValue>& method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {

    const std::string& method = method_call.method_name();
    auto result_ptr = std::shared_ptr<flutter::MethodResult<flutter::EncodableValue>>(
        std::move(result));

    // ── loadModel ──────────────────────────────────────────────────────────────
    if (method == "loadModel") {
        auto* args = std::get_if<flutter::EncodableMap>(method_call.arguments());
        if (!args) {
            result_ptr->Error("INVALID_ARGS", "Missing arguments");
            return;
        }
        auto model_path = GetStr(*args, "modelPath");
        auto n_threads   = GetInt(*args, "nThreads", 4);
        auto n_gpu_layers = GetInt(*args, "nGpuLayers", 0);
        auto context_size = GetInt(*args, "contextSize", 2048);
        auto batch_size   = GetInt(*args, "batchSize", 512);
        auto use_gpu      = GetBool(*args, "useGpu", true);
        auto verbose      = GetBool(*args, "verbose", false);

        if (model_path.empty()) {
            result_ptr->Error("INVALID_ARGS", "Missing modelPath");
            return;
        }

        std::thread([=]() {
            bool ok = llama_init_model(model_path.c_str(), n_threads, n_gpu_layers,
                                       context_size, batch_size, use_gpu, verbose);
            if (ok) {
                result_ptr->Success(flutter::EncodableValue(true));
            } else {
                result_ptr->Error("INIT_FAILED", "Failed to initialize model");
            }
        }).detach();

    // ── generate ──────────────────────────────────────────────────────────────
    } else if (method == "generate") {
        auto* args = std::get_if<flutter::EncodableMap>(method_call.arguments());
        if (!args) { result_ptr->Error("INVALID_ARGS", "Missing arguments"); return; }

        auto prompt      = GetStr(*args, "prompt");
        auto temperature = (float)GetDouble(*args, "temperature", 0.8);
        auto top_p       = (float)GetDouble(*args, "topP", 0.95);
        auto top_k       = GetInt(*args, "topK", 40);
        auto max_tokens  = GetInt(*args, "maxTokens", 512);
        auto repeat_pen  = (float)GetDouble(*args, "repeatPenalty", 1.1);

        std::thread([=]() {
            std::vector<char> buf(16384, 0);
            int32_t tokens_gen = 0;
            auto t0 = std::chrono::steady_clock::now();
            bool ok = llama_generate(prompt.c_str(), temperature, top_p, top_k,
                                     max_tokens, repeat_pen,
                                     buf.data(), (int32_t)buf.size(), &tokens_gen);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();

            if (ok) {
                flutter::EncodableMap resp;
                resp[flutter::EncodableValue("text")]              = flutter::EncodableValue(std::string(buf.data()));
                resp[flutter::EncodableValue("tokensGenerated")]   = flutter::EncodableValue((int)tokens_gen);
                resp[flutter::EncodableValue("generationTimeMs")]  = flutter::EncodableValue((int)ms);
                result_ptr->Success(flutter::EncodableValue(resp));
            } else {
                result_ptr->Error("GENERATION_FAILED", "Failed to generate response");
            }
        }).detach();

    // ── generateStream ────────────────────────────────────────────────────────
    } else if (method == "generateStream") {
        auto* args = std::get_if<flutter::EncodableMap>(method_call.arguments());
        if (!args) { result_ptr->Error("INVALID_ARGS", "Missing arguments"); return; }

        auto prompt      = GetStr(*args, "prompt");
        auto temperature = (float)GetDouble(*args, "temperature", 0.8);
        auto top_p       = (float)GetDouble(*args, "topP", 0.95);
        auto top_k       = GetInt(*args, "topK", 40);
        auto max_tokens  = GetInt(*args, "maxTokens", 512);
        auto repeat_pen  = (float)GetDouble(*args, "repeatPenalty", 1.1);

        std::thread([=]() {
            llama_generate_stream_init(prompt.c_str(), temperature, top_p,
                                       top_k, max_tokens, repeat_pen);

            std::vector<char> token_buf(256, 0);
            while (llama_generate_stream_next(token_buf.data(), (int32_t)token_buf.size())) {
                SendStreamToken(std::string(token_buf.data()));
                std::fill(token_buf.begin(), token_buf.end(), 0);
            }

            llama_generate_stream_end();
            SendStreamEnd();
            result_ptr->Success(flutter::EncodableValue());
        }).detach();

    // ── unloadModel ───────────────────────────────────────────────────────────
    } else if (method == "unloadModel") {
        std::thread([=]() {
            llama_bridge_free_model();
            result_ptr->Success(flutter::EncodableValue());
        }).detach();

    // ── getModelInfo ──────────────────────────────────────────────────────────
    } else if (method == "getModelInfo") {
        int64_t n_params = 0; int32_t n_layers = 0, ctx_size = 0;
        llama_get_model_info(&n_params, &n_layers, &ctx_size);
        flutter::EncodableMap info;
        info[flutter::EncodableValue("nParams")]     = flutter::EncodableValue((int64_t)n_params);
        info[flutter::EncodableValue("nLayers")]     = flutter::EncodableValue((int)n_layers);
        info[flutter::EncodableValue("contextSize")] = flutter::EncodableValue((int)ctx_size);
        result_ptr->Success(flutter::EncodableValue(info));

    // ── stopGeneration ────────────────────────────────────────────────────────
    } else if (method == "stopGeneration") {
        llama_stop_generation();
        result_ptr->Success(flutter::EncodableValue());

    // ── getEmbedding ──────────────────────────────────────────────────────────
    } else if (method == "getEmbedding") {
        auto* args = std::get_if<flutter::EncodableMap>(method_call.arguments());
        auto text = args ? GetStr(*args, "text") : "";

        std::thread([=]() {
            std::vector<float> buf(8192, 0.0f);
            int32_t n = llama_get_embedding(text.c_str(), buf.data(), (int32_t)buf.size());
            if (n > 0) {
                flutter::EncodableList lst;
                for (int i = 0; i < n; i++) lst.push_back(flutter::EncodableValue((double)buf[i]));
                result_ptr->Success(flutter::EncodableValue(lst));
            } else {
                result_ptr->Error("EMBEDDING_FAILED", "Failed to compute embedding");
            }
        }).detach();

    // ── getEmbeddingDim ───────────────────────────────────────────────────────
    } else if (method == "getEmbeddingDim") {
        result_ptr->Success(flutter::EncodableValue((int)llama_get_embedding_dim()));

    // ── loadEmbeddingModel ────────────────────────────────────────────────────
    } else if (method == "loadEmbeddingModel") {
        auto* args = std::get_if<flutter::EncodableMap>(method_call.arguments());
        auto path = args ? GetStr(*args, "modelPath") : "";

        std::thread([=]() {
            bool ok = llama_load_embedding_model(path.c_str());
            if (ok) {
                result_ptr->Success(flutter::EncodableValue(true));
            } else {
                result_ptr->Error("LOAD_FAILED", "Failed to load embedding model");
            }
        }).detach();

    // ── getEmbeddingV2 ────────────────────────────────────────────────────────
    } else if (method == "getEmbeddingV2") {
        auto* args = std::get_if<flutter::EncodableMap>(method_call.arguments());
        auto text = args ? GetStr(*args, "text") : "";

        std::thread([=]() {
            std::vector<float> buf(8192, 0.0f);
            int32_t n = llama_get_embedding_v2(text.c_str(), buf.data(), (int32_t)buf.size());
            if (n > 0) {
                flutter::EncodableList lst;
                for (int i = 0; i < n; i++) lst.push_back(flutter::EncodableValue((double)buf[i]));
                result_ptr->Success(flutter::EncodableValue(lst));
            } else {
                result_ptr->Error("EMBEDDING_FAILED", "Failed to compute embedding v2");
            }
        }).detach();

    // ── unloadEmbeddingModel ──────────────────────────────────────────────────
    } else if (method == "unloadEmbeddingModel") {
        std::thread([=]() {
            llama_unload_embedding_model();
            result_ptr->Success(flutter::EncodableValue());
        }).detach();

    // ── isEmbeddingModelLoaded ────────────────────────────────────────────────
    } else if (method == "isEmbeddingModelLoaded") {
        result_ptr->Success(flutter::EncodableValue(llama_is_embedding_model_loaded()));

    // ── getEmbeddingModelDim ──────────────────────────────────────────────────
    } else if (method == "getEmbeddingModelDim") {
        result_ptr->Success(flutter::EncodableValue((int)llama_get_embedding_model_dim()));

    // ── multimodal stubs (not yet supported on Windows) ───────────────────────
    } else if (method == "loadMultimodalModel" || method == "generateMultimodal" ||
               method == "generateMultimodalStream" || method == "getMultimodalModelInfo" ||
               method == "unloadMultimodalModel" || method == "stopMultimodalGeneration") {
        result_ptr->Error("UNSUPPORTED", "Multimodal is not yet supported on Windows");

    } else {
        result_ptr->NotImplemented();
    }
}
