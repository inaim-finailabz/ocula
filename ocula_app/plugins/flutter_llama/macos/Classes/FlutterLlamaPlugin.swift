import FlutterMacOS
import Foundation

// ============================================================================
// MARK: - Multimodal C Bridge Declarations
// ============================================================================

@_silgen_name("mtmd_bridge_load")
func mtmd_bridge_load(
    _ modelPath: UnsafePointer<CChar>,
    _ mmprojPath: UnsafePointer<CChar>,
    _ nThreads: Int32,
    _ nGpuLayers: Int32,
    _ contextSize: Int32,
    _ batchSize: Int32,
    _ useGpu: Bool
) -> Bool

@_silgen_name("mtmd_bridge_generate")
func mtmd_bridge_generate(
    _ prompt: UnsafePointer<CChar>,
    _ imagePath: UnsafePointer<CChar>?,
    _ audioPath: UnsafePointer<CChar>?,
    _ temperature: Float,
    _ topP: Float,
    _ topK: Int32,
    _ maxTokens: Int32,
    _ repeatPenalty: Float,
    _ output: UnsafeMutablePointer<CChar>,
    _ outputSize: Int32,
    _ tokensGenerated: UnsafeMutablePointer<Int32>
) -> Bool

@_silgen_name("mtmd_bridge_stream_init")
func mtmd_bridge_stream_init(
    _ prompt: UnsafePointer<CChar>,
    _ imagePath: UnsafePointer<CChar>?,
    _ audioPath: UnsafePointer<CChar>?,
    _ temperature: Float,
    _ topP: Float,
    _ topK: Int32,
    _ maxTokens: Int32,
    _ repeatPenalty: Float
)

@_silgen_name("mtmd_bridge_stream_next")
func mtmd_bridge_stream_next(
    _ output: UnsafeMutablePointer<CChar>,
    _ outputSize: Int32
) -> Bool

@_silgen_name("mtmd_bridge_stream_end")
func mtmd_bridge_stream_end()

@_silgen_name("mtmd_bridge_get_info")
func mtmd_bridge_get_info(
    _ nParams: UnsafeMutablePointer<Int64>,
    _ nLayers: UnsafeMutablePointer<Int32>,
    _ contextSize: UnsafeMutablePointer<Int32>,
    _ supportsVision: UnsafeMutablePointer<Bool>,
    _ supportsAudio: UnsafeMutablePointer<Bool>
)

@_silgen_name("mtmd_bridge_free")
func mtmd_bridge_free()

@_silgen_name("mtmd_bridge_stop")
func mtmd_bridge_stop()

// ============================================================================
// MARK: - FlutterLlamaMultimodalHandler (handles flutter_llama_multimodal)
// ============================================================================

@available(macOS 10.14, *)
class FlutterLlamaMultimodalHandler: NSObject, FlutterPlugin, FlutterStreamHandler {
    private var modelLoaded = false
    private let queue = DispatchQueue(label: "net.nativemind.flutter_llama_multimodal", qos: .userInitiated)
    private var eventSink: FlutterEventSink?
    private var shouldStop = false
    
    static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "flutter_llama_multimodal",
            binaryMessenger: registrar.messenger
        )
        let eventChannel = FlutterEventChannel(
            name: "flutter_llama_multimodal/stream",
            binaryMessenger: registrar.messenger
        )
        let instance = FlutterLlamaMultimodalHandler()
        registrar.addMethodCallDelegate(instance, channel: channel)
        eventChannel.setStreamHandler(instance)
        NSLog("[FlutterLlamaMultimodal] Channel registered")
    }
    
    func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "loadMultimodalModel":
            loadMultimodalModel(call: call, result: result)
        case "generateMultimodal":
            generateMultimodal(call: call, result: result)
        case "generateMultimodalStream":
            generateMultimodalStream(call: call, result: result)
        case "getMultimodalModelInfo":
            getMultimodalModelInfo(result: result)
        case "unloadMultimodalModel":
            unloadMultimodalModel(result: result)
        case "stopMultimodalGeneration":
            stopMultimodalGeneration(result: result)
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    // MARK: - FlutterStreamHandler
    
    func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        self.eventSink = events
        return nil
    }
    
    func onCancel(withArguments arguments: Any?) -> FlutterError? {
        self.eventSink = nil
        shouldStop = true
        return nil
    }
    
    // MARK: - Load Multimodal Model
    
    private func loadMultimodalModel(call: FlutterMethodCall, result: @escaping FlutterResult) {
        queue.async { [weak self] in
            guard let self = self else { return }
            guard let args = call.arguments as? [String: Any],
                  let textModelPath = args["textModelPath"] as? String else {
                DispatchQueue.main.async {
                    result(FlutterError(code: "INVALID_ARGS", message: "Missing textModelPath", details: nil))
                }
                return
            }
            
            let mmprojPath = args["mmprojPath"] as? String ?? ""
            let useGpu = args["useGpuForMultimodal"] as? Bool ?? true
            let nThreads = (args["extraParams"] as? [String: Any])?["nThreads"] as? Int ?? 4
            let nGpuLayers = (args["extraParams"] as? [String: Any])?["nGpuLayers"] as? Int ?? 99
            let contextSize = (args["extraParams"] as? [String: Any])?["contextSize"] as? Int ?? 2048
            let batchSize = (args["extraParams"] as? [String: Any])?["batchSize"] as? Int ?? 512
            
            let success = textModelPath.withCString { modelPtr in
                mmprojPath.withCString { mmprojPtr in
                    mtmd_bridge_load(modelPtr, mmprojPtr,
                                     Int32(nThreads), Int32(nGpuLayers),
                                     Int32(contextSize), Int32(batchSize), useGpu)
                }
            }
            
            self.modelLoaded = success
            
            DispatchQueue.main.async {
                if success {
                    NSLog("[FlutterLlamaMultimodal] Model loaded: \(textModelPath)")
                    result(true)
                } else {
                    result(FlutterError(code: "INIT_FAILED", message: "Failed to load multimodal model", details: nil))
                }
            }
        }
    }
    
    // MARK: - Generate Multimodal (blocking)
    
    private func generateMultimodal(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard modelLoaded else {
            result(FlutterError(code: "MODEL_NOT_LOADED", message: "Multimodal model not loaded", details: nil))
            return
        }
        
        queue.async { [weak self] in
            guard let self = self else { return }
            guard let args = call.arguments as? [String: Any],
                  let inputMap = args["input"] as? [String: Any],
                  let paramsMap = args["params"] as? [String: Any] else {
                DispatchQueue.main.async {
                    result(FlutterError(code: "INVALID_ARGS", message: "Missing input or params", details: nil))
                }
                return
            }
            
            let prompt = (paramsMap["prompt"] as? String) ?? (inputMap["text"] as? String) ?? ""
            let imagePath = inputMap["imagePath"] as? String
            let audioPath = inputMap["audioPath"] as? String
            let inputType = inputMap["type"] as? String ?? "text"
            
            let temperature = Float(paramsMap["temperature"] as? Double ?? 0.8)
            let topP = Float(paramsMap["topP"] as? Double ?? 0.95)
            let topK = Int32(paramsMap["topK"] as? Int ?? 40)
            let maxTokens = Int32(paramsMap["maxTokens"] as? Int ?? 512)
            let repeatPenalty = Float(paramsMap["repeatPenalty"] as? Double ?? 1.1)
            
            self.shouldStop = false
            let startTime = Date()
            
            var outputBuffer = [CChar](repeating: 0, count: 16384)
            var tokensGenerated: Int32 = 0
            
            // Build the prompt with media marker if needed
            var fullPrompt = prompt
            if (imagePath != nil && !imagePath!.isEmpty) || (audioPath != nil && !audioPath!.isEmpty) {
                let marker = "<__media__>"
                if !fullPrompt.contains(marker) {
                    fullPrompt = "\(marker)\n\(prompt)"
                }
            }
            
            let success: Bool = fullPrompt.withCString { promptPtr in
                if let imagePath = imagePath, !imagePath.isEmpty {
                    return imagePath.withCString { imgCStr in
                        if let audioPath = audioPath, !audioPath.isEmpty {
                            return audioPath.withCString { audCStr in
                                mtmd_bridge_generate(promptPtr, imgCStr, audCStr,
                                                     temperature, topP, topK, maxTokens,
                                                     repeatPenalty, &outputBuffer,
                                                     Int32(outputBuffer.count), &tokensGenerated)
                            }
                        } else {
                            return mtmd_bridge_generate(promptPtr, imgCStr, nil,
                                                       temperature, topP, topK, maxTokens,
                                                       repeatPenalty, &outputBuffer,
                                                       Int32(outputBuffer.count), &tokensGenerated)
                        }
                    }
                } else if let audioPath = audioPath, !audioPath.isEmpty {
                    return audioPath.withCString { audCStr in
                        mtmd_bridge_generate(promptPtr, nil, audCStr,
                                             temperature, topP, topK, maxTokens,
                                             repeatPenalty, &outputBuffer,
                                             Int32(outputBuffer.count), &tokensGenerated)
                    }
                } else {
                    return mtmd_bridge_generate(promptPtr, nil, nil,
                                               temperature, topP, topK, maxTokens,
                                               repeatPenalty, &outputBuffer,
                                               Int32(outputBuffer.count), &tokensGenerated)
                }
            }
            
            let generationTime = Int(Date().timeIntervalSince(startTime) * 1000)
            
            DispatchQueue.main.async {
                if success {
                    let responseText = String(cString: outputBuffer)
                    var processedModalities: [String] = ["text"]
                    if imagePath != nil && !imagePath!.isEmpty { processedModalities.append("image") }
                    if audioPath != nil && !audioPath!.isEmpty { processedModalities.append("audio") }
                    
                    let response: [String: Any] = [
                        "text": responseText,
                        "metadata": [:] as [String: Any],
                        "generationTimeMs": generationTime,
                        "tokensGenerated": Int(tokensGenerated),
                        "inputType": inputType,
                        "processedModalities": processedModalities,
                    ]
                    NSLog("[FlutterLlamaMultimodal] Generated \(tokensGenerated) tokens in \(generationTime)ms")
                    result(response)
                } else {
                    result(FlutterError(code: "GENERATION_FAILED", message: "Multimodal generation failed", details: nil))
                }
            }
        }
    }
    
    // MARK: - Generate Stream
    
    private func generateMultimodalStream(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard modelLoaded else {
            result(FlutterError(code: "MODEL_NOT_LOADED", message: "Multimodal model not loaded", details: nil))
            return
        }
        guard let eventSink = self.eventSink else {
            result(FlutterError(code: "NO_EVENT_SINK", message: "Event channel not initialized", details: nil))
            return
        }
        
        queue.async { [weak self] in
            guard let self = self else { return }
            guard let args = call.arguments as? [String: Any],
                  let inputMap = args["input"] as? [String: Any],
                  let paramsMap = args["params"] as? [String: Any] else {
                DispatchQueue.main.async {
                    result(FlutterError(code: "INVALID_ARGS", message: "Missing input or params", details: nil))
                }
                return
            }
            
            let prompt = (paramsMap["prompt"] as? String) ?? (inputMap["text"] as? String) ?? ""
            let imagePath = inputMap["imagePath"] as? String
            let audioPath = inputMap["audioPath"] as? String
            
            let temperature = Float(paramsMap["temperature"] as? Double ?? 0.8)
            let topP = Float(paramsMap["topP"] as? Double ?? 0.95)
            let topK = Int32(paramsMap["topK"] as? Int ?? 40)
            let maxTokens = Int32(paramsMap["maxTokens"] as? Int ?? 512)
            let repeatPenalty = Float(paramsMap["repeatPenalty"] as? Double ?? 1.1)
            
            self.shouldStop = false
            
            var fullPrompt = prompt
            if (imagePath != nil && !imagePath!.isEmpty) || (audioPath != nil && !audioPath!.isEmpty) {
                let marker = "<__media__>"
                if !fullPrompt.contains(marker) {
                    fullPrompt = "\(marker)\n\(prompt)"
                }
            }
            
            fullPrompt.withCString { promptPtr in
                if let imagePath = imagePath, !imagePath.isEmpty {
                    imagePath.withCString { imgCStr in
                        if let audioPath = audioPath, !audioPath.isEmpty {
                            audioPath.withCString { audCStr in
                                mtmd_bridge_stream_init(promptPtr, imgCStr, audCStr,
                                                        temperature, topP, topK, maxTokens, repeatPenalty)
                            }
                        } else {
                            mtmd_bridge_stream_init(promptPtr, imgCStr, nil,
                                                    temperature, topP, topK, maxTokens, repeatPenalty)
                        }
                    }
                } else if let audioPath = audioPath, !audioPath.isEmpty {
                    audioPath.withCString { audCStr in
                        mtmd_bridge_stream_init(promptPtr, nil, audCStr,
                                                temperature, topP, topK, maxTokens, repeatPenalty)
                    }
                } else {
                    mtmd_bridge_stream_init(promptPtr, nil, nil,
                                            temperature, topP, topK, maxTokens, repeatPenalty)
                }
            }
            
            var tokenBuffer = [CChar](repeating: 0, count: 256)
            while !self.shouldStop {
                let hasMore = mtmd_bridge_stream_next(&tokenBuffer, Int32(tokenBuffer.count))
                if hasMore {
                    let token = String(cString: tokenBuffer)
                    DispatchQueue.main.async {
                        eventSink(token)
                    }
                } else {
                    break
                }
            }
            
            mtmd_bridge_stream_end()
            
            DispatchQueue.main.async {
                eventSink(FlutterEndOfEventStream)
                result(nil)
            }
        }
    }
    
    // MARK: - Get Model Info
    
    private func getMultimodalModelInfo(result: @escaping FlutterResult) {
        guard modelLoaded else {
            result(nil)
            return
        }
        
        var nParams: Int64 = 0
        var nLayers: Int32 = 0
        var contextSize: Int32 = 0
        var supportsVision: Bool = false
        var supportsAudio: Bool = false
        
        mtmd_bridge_get_info(&nParams, &nLayers, &contextSize, &supportsVision, &supportsAudio)
        
        let info: [String: Any] = [
            "nParams": nParams,
            "nLayers": nLayers,
            "contextSize": contextSize,
            "supportsVision": supportsVision,
            "supportsAudio": supportsAudio
        ]
        result(info)
    }
    
    // MARK: - Unload
    
    private func unloadMultimodalModel(result: @escaping FlutterResult) {
        queue.async { [weak self] in
            guard let self = self else { return }
            if self.modelLoaded {
                mtmd_bridge_free()
                self.modelLoaded = false
                NSLog("[FlutterLlamaMultimodal] Model unloaded")
            }
            DispatchQueue.main.async { result(nil) }
        }
    }
    
    // MARK: - Stop
    
    private func stopMultimodalGeneration(result: @escaping FlutterResult) {
        shouldStop = true
        mtmd_bridge_stop()
        result(nil)
    }
}

// ============================================================================
// MARK: - FlutterLlamaPlugin (handles flutter_llama text-only channel)
// ============================================================================


@_silgen_name("llama_init_model")
func llama_init_model(
    _ modelPath: UnsafePointer<CChar>,
    _ nThreads: Int32,
    _ nGpuLayers: Int32,
    _ ctxSize: Int32,
    _ batchSize: Int32,
    _ useGpu: Bool,
    _ verbose: Bool
) -> Bool

@_silgen_name("llama_generate")
func llama_generate(
    _ prompt: UnsafePointer<CChar>,
    _ temperature: Float,
    _ topP: Float,
    _ topK: Int32,
    _ maxTokens: Int32,
    _ repeatPenalty: Float,
    _ output: UnsafeMutablePointer<CChar>,
    _ outputSize: Int32,
    _ tokensGenerated: UnsafeMutablePointer<Int32>
) -> Bool

@_silgen_name("llama_generate_stream_init")
func llama_generate_stream_init(
    _ prompt: UnsafePointer<CChar>,
    _ temperature: Float,
    _ topP: Float,
    _ topK: Int32,
    _ maxTokens: Int32,
    _ repeatPenalty: Float
)

@_silgen_name("llama_generate_stream_next")
func llama_generate_stream_next(
    _ output: UnsafeMutablePointer<CChar>,
    _ outputSize: Int32
) -> Bool

@_silgen_name("llama_generate_stream_end")
func llama_generate_stream_end()

@_silgen_name("llama_get_model_info")
func llama_get_model_info(
    _ nParams: UnsafeMutablePointer<Int64>,
    _ nLayers: UnsafeMutablePointer<Int32>,
    _ contextSize: UnsafeMutablePointer<Int32>
)

@_silgen_name("llama_bridge_free_model")
func llama_bridge_free_model()

@_silgen_name("llama_stop_generation")
func llama_stop_generation()

@_silgen_name("llama_get_embedding")
func llama_bridge_get_embedding(
    _ text: UnsafePointer<CChar>,
    _ output: UnsafeMutablePointer<Float>,
    _ outputSize: Int32
) -> Int32

@_silgen_name("llama_get_embedding_dim")
func llama_bridge_get_embedding_dim() -> Int32

/**
 * FlutterLlamaPlugin - плагин для работы с llama.cpp моделями на macOS
 * 
 * Поддерживает:
 * - Загрузку GGUF моделей
 * - GPU ускорение через Metal
 * - Потоковую и обычную генерацию
 */
@available(macOS 10.14, *)
public class FlutterLlamaPlugin: NSObject, FlutterPlugin, FlutterStreamHandler {
    private var modelLoaded = false
    private var modelPath: String?
    private let queue = DispatchQueue(label: "net.nativemind.flutter_llama", qos: .userInitiated)
    private var eventSink: FlutterEventSink?
    private var shouldStop = false
    
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "flutter_llama",
            binaryMessenger: registrar.messenger
        )
        
        let eventChannel = FlutterEventChannel(
            name: "flutter_llama/stream",
            binaryMessenger: registrar.messenger
        )
        
        let instance = FlutterLlamaPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
        eventChannel.setStreamHandler(instance)
        
        // --- Multimodal channel ---
        FlutterLlamaMultimodalHandler.register(with: registrar)
        
        NSLog("[FlutterLlama] Plugin registered (text + multimodal)")
    }
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "loadModel":
            loadModel(call: call, result: result)
        case "generate":
            generate(call: call, result: result)
        case "generateStream":
            generateStream(call: call, result: result)
        case "unloadModel":
            unloadModel(result: result)
        case "getModelInfo":
            getModelInfo(result: result)
        case "stopGeneration":
            stopGeneration(result: result)
        case "getEmbedding":
            getEmbedding(call: call, result: result)
        case "getEmbeddingDim":
            getEmbeddingDim(result: result)
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    // MARK: - FlutterStreamHandler
    
    public func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        self.eventSink = events
        return nil
    }
    
    public func onCancel(withArguments arguments: Any?) -> FlutterError? {
        self.eventSink = nil
        shouldStop = true
        return nil
    }
    
    // MARK: - Load Model
    
    private func loadModel(call: FlutterMethodCall, result: @escaping FlutterResult) {
        queue.async { [weak self] in
            guard let self = self else { return }
            guard let args = call.arguments as? [String: Any],
                  let modelPath = args["modelPath"] as? String else {
                DispatchQueue.main.async {
                    result(FlutterError(
                        code: "INVALID_ARGS",
                        message: "Missing required arguments",
                        details: nil
                    ))
                }
                return
            }
            
            let nThreads = args["nThreads"] as? Int ?? 4
            let nGpuLayers = args["nGpuLayers"] as? Int ?? 0
            let contextSize = args["contextSize"] as? Int ?? 2048
            let batchSize = args["batchSize"] as? Int ?? 512
            let useGpu = args["useGpu"] as? Bool ?? true
            let verbose = args["verbose"] as? Bool ?? false
            
            // Check if model file exists
            let fileManager = FileManager.default
            guard fileManager.fileExists(atPath: modelPath) else {
                DispatchQueue.main.async {
                    result(FlutterError(
                        code: "MODEL_NOT_FOUND",
                        message: "Model file not found: \(modelPath)",
                        details: nil
                    ))
                }
                return
            }
            
            self.modelPath = modelPath
            
            // Initialize model through llama.cpp C++ bridge
            let success = modelPath.withCString { modelPathPtr in
                llama_init_model(
                    modelPathPtr,
                    Int32(nThreads),
                    Int32(nGpuLayers),
                    Int32(contextSize),
                    Int32(batchSize),
                    useGpu,
                    verbose
                )
            }
            
            self.modelLoaded = success
            
            DispatchQueue.main.async {
                if success {
                    NSLog("[FlutterLlama] Model loaded: \(modelPath)")
                    NSLog("[FlutterLlama] GPU layers: \(nGpuLayers), threads: \(nThreads), context: \(contextSize)")
                    result(true)
                } else {
                    result(FlutterError(
                        code: "INIT_FAILED",
                        message: "Failed to initialize model",
                        details: nil
                    ))
                }
            }
        }
    }
    
    // MARK: - Generate (blocking)
    
    private func generate(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard modelLoaded else {
            result(FlutterError(
                code: "MODEL_NOT_LOADED",
                message: "Model not loaded",
                details: nil
            ))
            return
        }
        
        queue.async {
            guard let args = call.arguments as? [String: Any],
                  let prompt = args["prompt"] as? String else {
                DispatchQueue.main.async {
                    result(FlutterError(
                        code: "INVALID_ARGS",
                        message: "Missing prompt",
                        details: nil
                    ))
                }
                return
            }
            
            let temperature = (args["temperature"] as? Double) ?? 0.8
            let topP = (args["topP"] as? Double) ?? 0.95
            let topK = (args["topK"] as? Int) ?? 40
            let maxTokens = (args["maxTokens"] as? Int) ?? 512
            let repeatPenalty = (args["repeatPenalty"] as? Double) ?? 1.1
            
            self.shouldStop = false
            let startTime = Date()
            
            // Generate through llama.cpp C++ bridge
            var outputBuffer = [CChar](repeating: 0, count: 16384)
            var tokensGenerated: Int32 = 0
            
            let success = prompt.withCString { promptPtr in
                llama_generate(
                    promptPtr,
                    Float(temperature),
                    Float(topP),
                    Int32(topK),
                    Int32(maxTokens),
                    Float(repeatPenalty),
                    &outputBuffer,
                    Int32(outputBuffer.count),
                    &tokensGenerated
                )
            }
            
            let generationTime = Int(Date().timeIntervalSince(startTime) * 1000)
            
            DispatchQueue.main.async {
                if success {
                    let responseText = String(cString: outputBuffer)
                    let response: [String: Any] = [
                        "text": responseText,
                        "tokensGenerated": Int(tokensGenerated),
                        "generationTimeMs": generationTime
                    ]
                    NSLog("[FlutterLlama] Generated: \(tokensGenerated) tokens in \(generationTime)ms")
                    result(response)
                } else {
                    result(FlutterError(
                        code: "GENERATION_FAILED",
                        message: "Failed to generate response",
                        details: nil
                    ))
                }
            }
        }
    }
    
    // MARK: - Generate Stream
    
    private func generateStream(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard modelLoaded else {
            result(FlutterError(
                code: "MODEL_NOT_LOADED",
                message: "Model not loaded",
                details: nil
            ))
            return
        }
        
        guard let eventSink = self.eventSink else {
            result(FlutterError(
                code: "NO_EVENT_SINK",
                message: "Event channel not initialized",
                details: nil
            ))
            return
        }
        
        queue.async { [weak self] in
            guard let self = self else { return }
            guard let args = call.arguments as? [String: Any],
                  let prompt = args["prompt"] as? String else {
                DispatchQueue.main.async {
                    result(FlutterError(
                        code: "INVALID_ARGS",
                        message: "Missing prompt",
                        details: nil
                    ))
                }
                return
            }
            
            let temperature = (args["temperature"] as? Double) ?? 0.8
            let topP = (args["topP"] as? Double) ?? 0.95
            let topK = (args["topK"] as? Int) ?? 40
            let maxTokens = (args["maxTokens"] as? Int) ?? 512
            let repeatPenalty = (args["repeatPenalty"] as? Double) ?? 1.1
            
            self.shouldStop = false
            
            // Initialize streaming generation
            prompt.withCString { promptPtr in
                llama_generate_stream_init(
                    promptPtr,
                    Float(temperature),
                    Float(topP),
                    Int32(topK),
                    Int32(maxTokens),
                    Float(repeatPenalty)
                )
            }
            
            // Stream tokens one by one
            var tokenBuffer = [CChar](repeating: 0, count: 256)
            while !self.shouldStop {
                let hasMore = llama_generate_stream_next(&tokenBuffer, Int32(tokenBuffer.count))
                
                if hasMore {
                    let token = String(cString: tokenBuffer)
                    DispatchQueue.main.async {
                        eventSink(token)
                    }
                } else {
                    break
                }
            }
            
            llama_generate_stream_end()
            
            DispatchQueue.main.async {
                eventSink(FlutterEndOfEventStream)
                result(nil)
            }
        }
    }
    
    // MARK: - Unload Model
    
    private func unloadModel(result: @escaping FlutterResult) {
        // Dispatch onto the same serial queue used by loadModel / generate
        // to avoid freeing the model while another operation is in flight.
        queue.async { [weak self] in
            guard let self = self else { return }
            if self.modelLoaded {
                llama_bridge_free_model()
                self.modelLoaded = false
                self.modelPath = nil
                NSLog("[FlutterLlama] Model unloaded")
            }
            DispatchQueue.main.async {
                result(nil)
            }
        }
    }
    
    // MARK: - Get Model Info
    
    private func getModelInfo(result: @escaping FlutterResult) {
        guard modelLoaded, let modelPath = modelPath else {
            result(nil)
            return
        }
        
        var nParams: Int64 = 0
        var nLayers: Int32 = 0
        var contextSize: Int32 = 0
        
        llama_get_model_info(&nParams, &nLayers, &contextSize)
        
        let info: [String: Any] = [
            "modelPath": modelPath,
            "nParams": nParams,
            "nLayers": nLayers,
            "contextSize": contextSize
        ]
        
        result(info)
    }
    
    // MARK: - Stop Generation
    
    private func stopGeneration(result: @escaping FlutterResult) {
        shouldStop = true
        llama_stop_generation()
        result(nil)
    }
    
    // MARK: - Get Embedding (for RAG)
    
    private func getEmbedding(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard modelLoaded else {
            result(FlutterError(
                code: "MODEL_NOT_LOADED",
                message: "Model not loaded",
                details: nil
            ))
            return
        }
        
        queue.async {
            guard let args = call.arguments as? [String: Any],
                  let text = args["text"] as? String else {
                DispatchQueue.main.async {
                    result(FlutterError(
                        code: "INVALID_ARGS",
                        message: "Missing text parameter",
                        details: nil
                    ))
                }
                return
            }
            
            let maxDim = 8192
            var outputBuffer = [Float](repeating: 0.0, count: maxDim)
            
            let n_embd = text.withCString { textPtr -> Int32 in
                return llama_bridge_get_embedding(textPtr, &outputBuffer, Int32(maxDim))
            }
            
            DispatchQueue.main.async {
                if n_embd > 0 {
                    let embedding = Array(outputBuffer.prefix(Int(n_embd))).map { Double($0) }
                    result(embedding)
                } else {
                    result(FlutterError(
                        code: "EMBEDDING_FAILED",
                        message: "Failed to compute embedding",
                        details: nil
                    ))
                }
            }
        }
    }
    
    // MARK: - Get Embedding Dimension
    
    private func getEmbeddingDim(result: @escaping FlutterResult) {
        let dim = llama_bridge_get_embedding_dim()
        result(Int(dim))
    }
}

// MARK: - C++ Bridge Function Declarations
