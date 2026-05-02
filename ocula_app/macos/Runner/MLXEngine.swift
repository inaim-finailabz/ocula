import Foundation
import FlutterMacOS
import MLXLLM
import MLXLMCommon

/// Singleton MLX inference engine for macOS.
/// Exposed to Flutter via two channels:
///   Method  — com.finailabz.ai.ocula/mlx        (loadModel, unload, isLoaded, cancelGeneration)
///   Event   — com.finailabz.ai.ocula/mlx_stream  (streaming token output)
@MainActor
class MLXEngine: NSObject {

    static let shared = MLXEngine()

    private var modelContainer: ModelContainer?
    private var generationTask: Task<Void, Never>?

    // Flutter sinks — set by MainFlutterWindow after channel registration.
    var methodChannel: FlutterMethodChannel?
    var eventSink: FlutterEventSink?

    private override init() {}

    // MARK: - Method channel handler

    func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "loadModel":
            guard let args = call.arguments as? [String: Any],
                  let path = args["path"] as? String else {
                result(FlutterError(code: "BAD_ARGS", message: "path required", details: nil))
                return
            }
            Task { await self.loadModel(path: path, result: result) }

        case "unload":
            modelContainer = nil
            result(nil)

        case "isLoaded":
            result(modelContainer != nil)

        case "generate":
            guard let args = call.arguments as? [String: Any],
                  let prompt = args["prompt"] as? String else {
                result(FlutterError(code: "BAD_ARGS", message: "prompt required", details: nil))
                return
            }
            let maxTokens = args["maxTokens"] as? Int ?? 2048
            let temperature = args["temperature"] as? Double ?? 0.7
            result(nil) // acknowledge immediately; tokens arrive on event channel
            Task { await self.generate(prompt: prompt, maxTokens: maxTokens, temperature: Float(temperature)) }

        case "cancelGeneration":
            generationTask?.cancel()
            generationTask = nil
            result(nil)

        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // MARK: - Load

    private func loadModel(path: String, result: @escaping FlutterResult) async {
        do {
            let config = ModelConfiguration(directory: URL(fileURLWithPath: path))
            let container = try await LLMModelFactory.shared.loadContainer(configuration: config) { progress in
                // Loading progress (0-1) — sent back as a method channel event
                DispatchQueue.main.async {
                    self.methodChannel?.invokeMethod("loadProgress", arguments: Double(progress.fractionCompleted))
                }
            }
            self.modelContainer = container
            result(true)
        } catch {
            result(FlutterError(code: "LOAD_FAILED", message: error.localizedDescription, details: nil))
        }
    }

    // MARK: - Generate (streaming)

    private func generate(prompt: String, maxTokens: Int, temperature: Float) async {
        guard let container = modelContainer else {
            eventSink?(["error": "Model not loaded"])
            return
        }

        generationTask?.cancel()
        generationTask = Task {
            do {
                let parameters = GenerateParameters(temperature: temperature)

                // Prepare LMInput and run generation inside container.perform so
                // both steps share the same ModelContext.
                let output: String = try await container.perform { context in
                    let lmInput = try await context.processor.prepare(
                        input: UserInput(prompt: .text(prompt))
                    )
                    var tokenCount = 0
                    let result = try MLXLMCommon.generate(
                        input: lmInput,
                        parameters: parameters,
                        context: context
                    ) { tokens in
                        guard !Task.isCancelled else { return .stop }
                        tokenCount += tokens.count
                        return tokenCount >= maxTokens ? .stop : .more
                    }
                    return result.output
                }

                // Stream the full output character-by-character so the Flutter
                // side sees the same progressive token delivery as mobile.
                await MainActor.run {
                    for char in output {
                        guard !Task.isCancelled else { break }
                        self.eventSink?(["token": String(char), "done": false])
                    }
                    self.eventSink?(["token": "", "done": true])
                }
            } catch {
                await MainActor.run {
                    self.eventSink?(["error": error.localizedDescription])
                }
            }
        }
    }
}

// MARK: - FlutterStreamHandler for event channel

class MLXStreamHandler: NSObject, FlutterStreamHandler {
    func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        Task { @MainActor in
            MLXEngine.shared.eventSink = events
        }
        return nil
    }

    func onCancel(withArguments arguments: Any?) -> FlutterError? {
        Task { @MainActor in
            MLXEngine.shared.eventSink = nil
        }
        return nil
    }
}
