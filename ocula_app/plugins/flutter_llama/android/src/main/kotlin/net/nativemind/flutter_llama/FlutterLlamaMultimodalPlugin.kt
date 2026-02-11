package net.nativemind.flutter_llama

import android.os.Handler
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
 * FlutterLlamaMultimodalPlugin - multimodal (image+text) inference for Android
 *
 * Port of FlutterLlamaMultimodalHandler (iOS/macOS Swift) to Kotlin.
 * Registers on channel "flutter_llama_multimodal" and event channel
 * "flutter_llama_multimodal/stream", matching the Dart side exactly.
 *
 * Native library "flutter_llama_multimodal_bridge" is loaded alongside the
 * text-only bridge in FlutterLlamaPlugin's companion init block.
 */
class FlutterLlamaMultimodalPlugin : FlutterPlugin, MethodCallHandler, EventChannel.StreamHandler {
    companion object {
        private const val TAG = "FlutterLlamaMultimodal"
        private const val CHANNEL_NAME = "flutter_llama_multimodal"
        private const val EVENT_CHANNEL_NAME = "flutter_llama_multimodal/stream"

        init {
            try {
                System.loadLibrary("flutter_llama_multimodal_bridge")
                Log.d(TAG, "Multimodal native library loaded")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load multimodal native library: ${e.message}")
            }
        }
    }

    private lateinit var channel: MethodChannel
    private lateinit var eventChannel: EventChannel
    private var eventSink: EventChannel.EventSink? = null
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    private var modelLoaded = false
    private var shouldStop = false

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(binding.binaryMessenger, CHANNEL_NAME)
        channel.setMethodCallHandler(this)

        eventChannel = EventChannel(binding.binaryMessenger, EVENT_CHANNEL_NAME)
        eventChannel.setStreamHandler(this)

        Log.d(TAG, "Multimodal plugin attached to engine")
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "loadMultimodalModel"       -> loadMultimodalModel(call, result)
            "generateMultimodal"        -> generateMultimodal(call, result)
            "generateMultimodalStream"  -> generateMultimodalStream(call, result)
            "getMultimodalModelInfo"    -> getMultimodalModelInfo(result)
            "unloadMultimodalModel"     -> unloadMultimodalModel(result)
            "stopMultimodalGeneration"  -> stopMultimodalGeneration(result)
            else -> result.notImplemented()
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        eventChannel.setStreamHandler(null)
        executor.shutdown()
    }

    // -- EventChannel.StreamHandler --

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        eventSink = events
    }

    override fun onCancel(arguments: Any?) {
        eventSink = null
        shouldStop = true
    }

    // ===================================================================
    // Load Multimodal Model
    // ===================================================================

    private fun loadMultimodalModel(call: MethodCall, result: Result) {
        executor.execute {
            try {
                val args = call.arguments as? Map<*, *>
                val textModelPath = args?.get("textModelPath") as? String
                if (textModelPath == null) {
                    mainHandler.post { result.error("INVALID_ARGS", "Missing textModelPath", null) }
                    return@execute
                }

                val mmprojPath  = (args["mmprojPath"] as? String) ?: ""
                val useGpu      = (args["useGpuForMultimodal"] as? Boolean) ?: true
                val extraParams = args["extraParams"] as? Map<*, *>
                val nThreads    = (extraParams?.get("nThreads") as? Int) ?: 4
                val nGpuLayers  = (extraParams?.get("nGpuLayers") as? Int) ?: 99
                val contextSize = (extraParams?.get("contextSize") as? Int) ?: 2048
                val batchSize   = (extraParams?.get("batchSize") as? Int) ?: 512

                // Verify files exist
                if (!File(textModelPath).exists()) {
                    mainHandler.post { result.error("MODEL_NOT_FOUND", "Text model not found: $textModelPath", null) }
                    return@execute
                }
                if (mmprojPath.isNotEmpty() && !File(mmprojPath).exists()) {
                    mainHandler.post { result.error("MODEL_NOT_FOUND", "MMProj not found: $mmprojPath", null) }
                    return@execute
                }

                val success = nativeLoadMultimodalModel(
                    textModelPath, mmprojPath,
                    nThreads, nGpuLayers, contextSize, batchSize, useGpu
                )

                modelLoaded = success

                mainHandler.post {
                    if (success) {
                        Log.d(TAG, "Multimodal model loaded: $textModelPath")
                        result.success(true)
                    } else {
                        result.error("INIT_FAILED", "Failed to load multimodal model", null)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading multimodal model", e)
                mainHandler.post { result.error("EXCEPTION", "Error: ${e.message}", null) }
            }
        }
    }

    // ===================================================================
    // Generate Multimodal (blocking)
    // ===================================================================

    private fun generateMultimodal(call: MethodCall, result: Result) {
        if (!modelLoaded) {
            result.error("MODEL_NOT_LOADED", "Multimodal model not loaded", null)
            return
        }

        executor.execute {
            try {
                val args     = call.arguments as? Map<*, *>
                val inputMap = args?.get("input") as? Map<*, *>
                val paramsMap = args?.get("params") as? Map<*, *>
                if (inputMap == null || paramsMap == null) {
                    mainHandler.post { result.error("INVALID_ARGS", "Missing input or params", null) }
                    return@execute
                }

                val prompt      = (paramsMap["prompt"] as? String) ?: (inputMap["text"] as? String) ?: ""
                val imagePath   = inputMap["imagePath"] as? String
                val audioPath   = inputMap["audioPath"] as? String
                val inputType   = (inputMap["type"] as? String) ?: "text"

                val temperature   = (paramsMap["temperature"] as? Double)?.toFloat() ?: 0.8f
                val topP          = (paramsMap["topP"] as? Double)?.toFloat() ?: 0.95f
                val topK          = (paramsMap["topK"] as? Int) ?: 40
                val maxTokens     = (paramsMap["maxTokens"] as? Int) ?: 512
                val repeatPenalty = (paramsMap["repeatPenalty"] as? Double)?.toFloat() ?: 1.1f

                shouldStop = false
                val startTime = System.currentTimeMillis()

                // Insert media marker if needed
                var fullPrompt = prompt
                if (!imagePath.isNullOrEmpty() || !audioPath.isNullOrEmpty()) {
                    val marker = "<__media__>"
                    if (!fullPrompt.contains(marker)) {
                        fullPrompt = "$marker\n$prompt"
                    }
                }

                val genResult = nativeGenerateMultimodal(
                    fullPrompt,
                    imagePath ?: "",
                    audioPath ?: "",
                    temperature, topP, topK, maxTokens, repeatPenalty
                )

                val generationTime = System.currentTimeMillis() - startTime

                mainHandler.post {
                    if (genResult != null) {
                        val processedModalities = mutableListOf("text")
                        if (!imagePath.isNullOrEmpty()) processedModalities.add("image")
                        if (!audioPath.isNullOrEmpty()) processedModalities.add("audio")

                        val response = hashMapOf(
                            "text" to genResult.text,
                            "metadata" to emptyMap<String, Any>(),
                            "generationTimeMs" to generationTime,
                            "tokensGenerated" to genResult.tokensGenerated,
                            "inputType" to inputType,
                            "processedModalities" to processedModalities
                        )
                        Log.d(TAG, "Generated ${genResult.tokensGenerated} tokens in ${generationTime}ms")
                        result.success(response)
                    } else {
                        result.error("GENERATION_FAILED", "Multimodal generation failed", null)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error in multimodal generation", e)
                mainHandler.post { result.error("EXCEPTION", "Error: ${e.message}", null) }
            }
        }
    }

    // ===================================================================
    // Generate Stream
    // ===================================================================

    private fun generateMultimodalStream(call: MethodCall, result: Result) {
        if (!modelLoaded) {
            result.error("MODEL_NOT_LOADED", "Multimodal model not loaded", null)
            return
        }
        val sink = eventSink
        if (sink == null) {
            result.error("NO_EVENT_SINK", "Event channel not initialized", null)
            return
        }

        executor.execute {
            try {
                val args      = call.arguments as? Map<*, *>
                val inputMap  = args?.get("input") as? Map<*, *>
                val paramsMap = args?.get("params") as? Map<*, *>
                if (inputMap == null || paramsMap == null) {
                    mainHandler.post { result.error("INVALID_ARGS", "Missing input or params", null) }
                    return@execute
                }

                val prompt      = (paramsMap["prompt"] as? String) ?: (inputMap["text"] as? String) ?: ""
                val imagePath   = inputMap["imagePath"] as? String
                val audioPath   = inputMap["audioPath"] as? String

                val temperature   = (paramsMap["temperature"] as? Double)?.toFloat() ?: 0.8f
                val topP          = (paramsMap["topP"] as? Double)?.toFloat() ?: 0.95f
                val topK          = (paramsMap["topK"] as? Int) ?: 40
                val maxTokens     = (paramsMap["maxTokens"] as? Int) ?: 512
                val repeatPenalty = (paramsMap["repeatPenalty"] as? Double)?.toFloat() ?: 1.1f

                shouldStop = false

                // Insert media marker if needed
                var fullPrompt = prompt
                if (!imagePath.isNullOrEmpty() || !audioPath.isNullOrEmpty()) {
                    val marker = "<__media__>"
                    if (!fullPrompt.contains(marker)) {
                        fullPrompt = "$marker\n$prompt"
                    }
                }

                nativeStreamInit(
                    fullPrompt,
                    imagePath ?: "",
                    audioPath ?: "",
                    temperature, topP, topK, maxTokens, repeatPenalty
                )

                // Stream tokens one by one
                while (!shouldStop) {
                    val token = nativeStreamNext()
                    if (token != null) {
                        mainHandler.post { sink.success(token) }
                    } else {
                        break
                    }
                }

                nativeStreamEnd()

                mainHandler.post {
                    sink.endOfStream()
                    result.success(null)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error in multimodal streaming", e)
                mainHandler.post {
                    sink.error("EXCEPTION", "Error: ${e.message}", null)
                    result.error("EXCEPTION", "Error: ${e.message}", null)
                }
            }
        }
    }

    // ===================================================================
    // Get Model Info
    // ===================================================================

    private fun getMultimodalModelInfo(result: Result) {
        if (!modelLoaded) {
            result.success(null)
            return
        }
        try {
            val info = nativeGetMultimodalModelInfo()
            if (info != null) {
                val map = hashMapOf(
                    "nParams" to info.nParams,
                    "nLayers" to info.nLayers,
                    "contextSize" to info.contextSize,
                    "supportsVision" to info.supportsVision,
                    "supportsAudio" to info.supportsAudio
                )
                result.success(map)
            } else {
                result.success(null)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting model info", e)
            result.success(null)
        }
    }

    // ===================================================================
    // Unload / Stop
    // ===================================================================

    private fun unloadMultimodalModel(result: Result) {
        executor.execute {
            if (modelLoaded) {
                nativeFreeMultimodalModel()
                modelLoaded = false
                Log.d(TAG, "Multimodal model unloaded")
            }
            mainHandler.post { result.success(null) }
        }
    }

    private fun stopMultimodalGeneration(result: Result) {
        shouldStop = true
        nativeStopMultimodalGeneration()
        result.success(null)
    }

    // ===================================================================
    // Native Methods (JNI)
    // ===================================================================

    private external fun nativeLoadMultimodalModel(
        modelPath: String, mmprojPath: String,
        nThreads: Int, nGpuLayers: Int,
        contextSize: Int, batchSize: Int,
        useGpu: Boolean
    ): Boolean

    private external fun nativeGenerateMultimodal(
        prompt: String, imagePath: String, audioPath: String,
        temperature: Float, topP: Float, topK: Int,
        maxTokens: Int, repeatPenalty: Float
    ): MultimodalGenerationResult?

    private external fun nativeStreamInit(
        prompt: String, imagePath: String, audioPath: String,
        temperature: Float, topP: Float, topK: Int,
        maxTokens: Int, repeatPenalty: Float
    )

    private external fun nativeStreamNext(): String?

    private external fun nativeStreamEnd()

    private external fun nativeGetMultimodalModelInfo(): MultimodalModelInfo?

    private external fun nativeFreeMultimodalModel()

    private external fun nativeStopMultimodalGeneration()

    // ===================================================================
    // Data classes for JNI results
    // ===================================================================

    data class MultimodalGenerationResult(
        val text: String,
        val tokensGenerated: Int
    )

    data class MultimodalModelInfo(
        val nParams: Long,
        val nLayers: Int,
        val contextSize: Int,
        val supportsVision: Boolean,
        val supportsAudio: Boolean
    )
}
