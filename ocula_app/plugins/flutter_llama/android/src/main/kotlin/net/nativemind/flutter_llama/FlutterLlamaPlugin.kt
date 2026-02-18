package net.nativemind.flutter_llama

import android.os.Handler
import android.os.Build
import android.os.Looper
import android.util.Log
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * FlutterLlamaPlugin - плагин для работы с llama.cpp моделями на Android
 * 
 * Поддерживает:
 * - Загрузку GGUF моделей
 * - GPU ускорение через Vulkan/OpenCL
 * - Потоковую и обычную генерацию
 */
class FlutterLlamaPlugin : FlutterPlugin, MethodCallHandler, EventChannel.StreamHandler {
    companion object {
        private const val TAG = "FlutterLlama"
        private const val CHANNEL_NAME = "flutter_llama"
        private const val EVENT_CHANNEL_NAME = "flutter_llama/stream"

        init {
            try {
                // Load llama.cpp libraries in correct order
                System.loadLibrary("c++_shared")
                System.loadLibrary("ggml")
                System.loadLibrary("ggml-base")
                System.loadLibrary("ggml-cpu")
                System.loadLibrary("llama")
                System.loadLibrary("flutter_llama_bridge")
                Log.d(TAG, "Native libraries loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native libraries: ${e.message}")
            }
        }
    }

    private lateinit var channel: MethodChannel
    private lateinit var eventChannel: EventChannel
    private var eventSink: EventChannel.EventSink? = null
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    
    private var modelLoaded = false
    private var modelPath: String? = null
    private var activeBackend: String = "cpu"
    private var shouldStop = false

    // Multimodal plugin (separate channels for image+text inference)
    private var multimodalPlugin: FlutterLlamaMultimodalPlugin? = null

    override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, CHANNEL_NAME)
        channel.setMethodCallHandler(this)
        
        eventChannel = EventChannel(flutterPluginBinding.binaryMessenger, EVENT_CHANNEL_NAME)
        eventChannel.setStreamHandler(this)
        
        // Register multimodal plugin on the same engine binding
        multimodalPlugin = FlutterLlamaMultimodalPlugin()
        multimodalPlugin!!.onAttachedToEngine(flutterPluginBinding)
        
        Log.d(TAG, "Plugin attached to engine (text + multimodal)")
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "loadModel" -> loadModel(call, result)
            "generate" -> generate(call, result)
            "generateStream" -> generateStream(call, result)
            "unloadModel" -> unloadModel(result)
            "getModelInfo" -> getModelInfo(result)
            "stopGeneration" -> stopGeneration(result)
            "getEmbedding" -> getEmbedding(call, result)
            "getEmbeddingDim" -> getEmbeddingDim(result)
            else -> result.notImplemented()
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        multimodalPlugin?.onDetachedFromEngine(binding)
        multimodalPlugin = null
        executor.shutdown()
    }

    // MARK: - EventChannel.StreamHandler

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        eventSink = events
    }

    override fun onCancel(arguments: Any?) {
        eventSink = null
        shouldStop = true
    }

    // MARK: - Load Model

    private fun loadModel(call: MethodCall, result: Result) {
        executor.execute {
            try {
                val modelPath = call.argument<String>("modelPath")
                if (modelPath == null) {
                    mainHandler.post {
                        result.error("INVALID_ARGS", "Missing modelPath", null)
                    }
                    return@execute
                }

                val nThreads = call.argument<Int>("nThreads") ?: 4
                val nGpuLayers = call.argument<Int>("nGpuLayers") ?: 0
                val contextSize = call.argument<Int>("contextSize") ?: 2048
                val batchSize = call.argument<Int>("batchSize") ?: 512
                val useGpu = call.argument<Boolean>("useGpu") ?: true
                val gpuBackend = call.argument<String>("gpuBackend") ?: "auto"
                val verbose = call.argument<Boolean>("verbose") ?: false

                // Check if model file exists
                val file = File(modelPath)
                if (!file.exists()) {
                    mainHandler.post {
                        result.error("MODEL_NOT_FOUND", "Model file not found: $modelPath", null)
                    }
                    return@execute
                }

                this.modelPath = modelPath
                val selectedBackend = resolveBackendPreference(gpuBackend, useGpu)

                // Initialize model through JNI
                val success = nativeInitModel(
                    modelPath,
                    nThreads,
                    nGpuLayers,
                    contextSize,
                    batchSize,
                    selectedBackend,
                    useGpu,
                    verbose
                )

                modelLoaded = success
                if (success) {
                    activeBackend = nativeGetActiveBackend() ?: "cpu"
                }

                mainHandler.post {
                    if (success) {
                        Log.d(TAG, "Model loaded: $modelPath")
                        Log.d(TAG, "GPU layers: $nGpuLayers, threads: $nThreads, context: $contextSize")
                        Log.d(TAG, "GPU backend requested=$gpuBackend resolved=$selectedBackend active=$activeBackend")
                        result.success(true)
                    } else {
                        result.error("INIT_FAILED", "Failed to initialize model", null)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading model", e)
                mainHandler.post {
                    result.error("EXCEPTION", "Error loading model: ${e.message}", null)
                }
            }
        }
    }

    // MARK: - Generate (blocking)

    private fun generate(call: MethodCall, result: Result) {
        if (!modelLoaded) {
            result.error("MODEL_NOT_LOADED", "Model not loaded", null)
            return
        }

        executor.execute {
            try {
                val prompt = call.argument<String>("prompt")
                if (prompt == null) {
                    mainHandler.post {
                        result.error("INVALID_ARGS", "Missing prompt", null)
                    }
                    return@execute
                }

                val temperature = call.argument<Double>("temperature")?.toFloat() ?: 0.8f
                val topP = call.argument<Double>("topP")?.toFloat() ?: 0.95f
                val topK = call.argument<Int>("topK") ?: 40
                val maxTokens = call.argument<Int>("maxTokens") ?: 512
                val repeatPenalty = call.argument<Double>("repeatPenalty")?.toFloat() ?: 1.1f

                shouldStop = false
                val startTime = System.currentTimeMillis()

                // Generate through JNI
                val generationResult = nativeGenerate(
                    prompt,
                    temperature,
                    topP,
                    topK,
                    maxTokens,
                    repeatPenalty
                )

                val generationTime = System.currentTimeMillis() - startTime

                mainHandler.post {
                    if (generationResult != null) {
                        val response = hashMapOf(
                            "text" to generationResult.text,
                            "tokensGenerated" to generationResult.tokensGenerated,
                            "generationTimeMs" to generationTime
                        )
                        Log.d(TAG, "Generated: ${generationResult.tokensGenerated} tokens in ${generationTime}ms")
                        result.success(response)
                    } else {
                        result.error("GENERATION_FAILED", "Failed to generate response", null)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error generating", e)
                mainHandler.post {
                    result.error("EXCEPTION", "Error generating: ${e.message}", null)
                }
            }
        }
    }

    // MARK: - Generate Stream

    private fun generateStream(call: MethodCall, result: Result) {
        if (!modelLoaded) {
            result.error("MODEL_NOT_LOADED", "Model not loaded", null)
            return
        }

        val sink = eventSink
        if (sink == null) {
            result.error("NO_EVENT_SINK", "Event channel not initialized", null)
            return
        }

        executor.execute {
            try {
                val prompt = call.argument<String>("prompt")
                if (prompt == null) {
                    mainHandler.post {
                        result.error("INVALID_ARGS", "Missing prompt", null)
                    }
                    return@execute
                }

                val temperature = call.argument<Double>("temperature")?.toFloat() ?: 0.8f
                val topP = call.argument<Double>("topP")?.toFloat() ?: 0.95f
                val topK = call.argument<Int>("topK") ?: 40
                val maxTokens = call.argument<Int>("maxTokens") ?: 512
                val repeatPenalty = call.argument<Double>("repeatPenalty")?.toFloat() ?: 1.1f

                shouldStop = false

                // Initialize streaming generation
                nativeGenerateStreamInit(prompt, temperature, topP, topK, maxTokens, repeatPenalty)

                // Stream tokens one by one
                while (!shouldStop) {
                    val token = nativeGenerateStreamNext()
                    if (token != null) {
                        mainHandler.post {
                            sink.success(token)
                        }
                    } else {
                        break
                    }
                }

                nativeGenerateStreamEnd()

                mainHandler.post {
                    sink.endOfStream()
                    result.success(null)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error in streaming generation", e)
                mainHandler.post {
                    sink.error("EXCEPTION", "Error in streaming: ${e.message}", null)
                    result.error("EXCEPTION", "Error in streaming: ${e.message}", null)
                }
            }
        }
    }

    // MARK: - Unload Model

    private fun unloadModel(result: Result) {
        if (modelLoaded) {
            nativeFreeModel()
            modelLoaded = false
            modelPath = null
            Log.d(TAG, "Model unloaded")
        }
        result.success(null)
    }

    // MARK: - Get Model Info

    private fun getModelInfo(result: Result) {
        if (!modelLoaded || modelPath == null) {
            result.success(null)
            return
        }

        try {
            val info = nativeGetModelInfo()
            if (info != null) {
                val infoMap = hashMapOf(
                    "modelPath" to modelPath!!,
                    "nParams" to info.nParams,
                    "nLayers" to info.nLayers,
                    "contextSize" to info.contextSize,
                    "backend" to activeBackend
                )
                result.success(infoMap)
            } else {
                result.success(null)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting model info", e)
            result.success(null)
        }
    }

    // MARK: - Stop Generation

    private fun stopGeneration(result: Result) {
        shouldStop = true
        nativeStopGeneration()
        result.success(null)
    }

    // MARK: - Get Embedding (for RAG)

    private fun getEmbedding(call: MethodCall, result: Result) {
        if (!modelLoaded) {
            result.error("MODEL_NOT_LOADED", "Model not loaded", null)
            return
        }

        executor.execute {
            try {
                val text = call.argument<String>("text")
                if (text == null) {
                    mainHandler.post {
                        result.error("INVALID_ARGS", "Missing text parameter", null)
                    }
                    return@execute
                }

                val embedding = nativeGetEmbedding(text)

                mainHandler.post {
                    if (embedding != null && embedding.isNotEmpty()) {
                        // Convert float[] to List<Double> for Dart
                        val doubleList = embedding.map { it.toDouble() }
                        Log.d(TAG, "Embedding: ${embedding.size}-dim")
                        result.success(doubleList)
                    } else {
                        result.error("EMBEDDING_FAILED", "Failed to compute embedding", null)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error computing embedding", e)
                mainHandler.post {
                    result.error("EXCEPTION", "Error computing embedding: ${e.message}", null)
                }
            }
        }
    }

    // MARK: - Get Embedding Dimension

    private fun getEmbeddingDim(result: Result) {
        val dim = nativeGetEmbeddingDim()
        result.success(dim)
    }

    private fun resolveBackendPreference(requested: String, useGpu: Boolean): String {
        if (!useGpu) return "cpu"

        val normalized = requested.trim().lowercase()
        if (normalized == "opencl" || normalized == "vulkan" || normalized == "cpu") {
            return normalized
        }

        val fingerprint = buildHardwareFingerprint()
        val preferOpenCL = fingerprint.any {
            it.contains("qcom") ||
            it.contains("qualcomm") ||
            it.contains("adreno") ||
            it.contains("snapdragon") ||
            it.startsWith("sm") ||
            it.startsWith("sdm") ||
            it.startsWith("msm")
        }
        if (preferOpenCL) return "opencl"

        val preferVulkan = fingerprint.any {
            it.contains("mali") ||
            it.contains("exynos") ||
            it.contains("mediatek") ||
            it.contains("mt") ||
            it.contains("kirin") ||
            it.contains("tensor") ||
            it.contains("immortalis")
        }
        return if (preferVulkan) "vulkan" else "auto"
    }

    private fun buildHardwareFingerprint(): List<String> {
        val values = mutableListOf(
            Build.MANUFACTURER.orEmpty(),
            Build.BRAND.orEmpty(),
            Build.MODEL.orEmpty(),
            Build.HARDWARE.orEmpty(),
            Build.BOARD.orEmpty(),
            Build.DEVICE.orEmpty(),
            Build.PRODUCT.orEmpty()
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            values.add(Build.SOC_MODEL.orEmpty())
            values.add(Build.SOC_MANUFACTURER.orEmpty())
        }
        return values.map { it.lowercase() }.filter { it.isNotBlank() }
    }

    // MARK: - Native Methods (JNI)

    private external fun nativeInitModel(
        modelPath: String,
        nThreads: Int,
        nGpuLayers: Int,
        contextSize: Int,
        batchSize: Int,
        preferredBackend: String,
        useGpu: Boolean,
        verbose: Boolean
    ): Boolean

    private external fun nativeGenerate(
        prompt: String,
        temperature: Float,
        topP: Float,
        topK: Int,
        maxTokens: Int,
        repeatPenalty: Float
    ): GenerationResult?

    private external fun nativeGenerateStreamInit(
        prompt: String,
        temperature: Float,
        topP: Float,
        topK: Int,
        maxTokens: Int,
        repeatPenalty: Float
    )

    private external fun nativeGenerateStreamNext(): String?

    private external fun nativeGenerateStreamEnd()

    private external fun nativeGetModelInfo(): ModelInfo?

    private external fun nativeFreeModel()

    private external fun nativeStopGeneration()

    private external fun nativeGetEmbedding(text: String): FloatArray?

    private external fun nativeGetEmbeddingDim(): Int

    private external fun nativeGetActiveBackend(): String?

    // Data classes for JNI results
    data class GenerationResult(
        val text: String,
        val tokensGenerated: Int
    )

    data class ModelInfo(
        val nParams: Long,
        val nLayers: Int,
        val contextSize: Int
    )
}
