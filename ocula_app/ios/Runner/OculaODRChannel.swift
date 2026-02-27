import Foundation
import Flutter

/// Platform channel that exposes iOS On-Demand Resources (ODR) to Flutter.
///
/// Models are tagged in Xcode's Resource Tags editor and marked as
/// "Initial Install Tags" (auto-downloaded after App Store install) or
/// "Prefetched Tags" (downloaded in background shortly after install).
///
/// In dev / TestFlight builds where ODR tags are not served,
/// `beginAccessingResources` returns an error and this handler returns nil —
/// the Dart caller falls through transparently to the network download path.
///
/// ODR download progress is emitted on the EventChannel
/// "com.finailabz.ocula/odr_progress" as Map { "tag": String, "progress": Double }.
/// The Dart side subscribes before calling requestODRTag to receive 0.0–1.0 updates
/// and show a progress bar during large model downloads.
class OculaODRChannel: NSObject, FlutterStreamHandler {
  static let channelName         = "com.finailabz.ocula/odr"
  static let progressChannelName = "com.finailabz.ocula/odr_progress"

  /// Keep strong references so NSBundleResourceRequests are not deallocated
  /// while a download is in progress.
  private var activeRequests: [NSBundleResourceRequest] = []

  /// KVO observations — one per active ODR request.
  private var progressObservations: [NSKeyValueObservation] = []

  /// Flutter sink for the odr_progress EventChannel.
  private var eventSink: FlutterEventSink?

  func register(with binaryMessenger: FlutterBinaryMessenger) {
    let channel = FlutterMethodChannel(
      name: OculaODRChannel.channelName,
      binaryMessenger: binaryMessenger
    )
    channel.setMethodCallHandler { [weak self] call, result in
      switch call.method {
      case "requestODRTag":
        guard let args = call.arguments as? [String: Any],
              let tag = args["tag"] as? String,
              let fileName = args["fileName"] as? String
        else {
          result(FlutterMethodNotImplemented)
          return
        }
        self?.requestTag(tag, fileName: fileName, result: result)

      case "requestODRTagsBatch":
        // Request multiple ODR tags in a single coordinated download session.
        // args: { "tags": [String], "fileNames": [String] }
        // Returns: Map<fileName, filePath> or nil on error.
        guard let args = call.arguments as? [String: Any],
              let tags = args["tags"] as? [String],
              let fileNames = args["fileNames"] as? [String]
        else {
          result(FlutterMethodNotImplemented)
          return
        }
        self?.requestTagsBatch(tags, fileNames: fileNames, result: result)

      default:
        result(FlutterMethodNotImplemented)
      }
    }

    // Event channel — Dart subscribes to receive download-progress updates.
    let eventChannel = FlutterEventChannel(
      name: OculaODRChannel.progressChannelName,
      binaryMessenger: binaryMessenger
    )
    eventChannel.setStreamHandler(self)
  }

  // MARK: - FlutterStreamHandler

  func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
    self.eventSink = events
    return nil
  }

  func onCancel(withArguments arguments: Any?) -> FlutterError? {
    self.eventSink = nil
    return nil
  }

  // MARK: - Private

  private func requestTag(_ tag: String, fileName: String, result: @escaping FlutterResult) {
    let request = NSBundleResourceRequest(tags: [tag])
    activeRequests.append(request)

    let observation = request.progress.observe(\.fractionCompleted, options: [.new]) { [weak self] progress, _ in
      DispatchQueue.main.async {
        self?.eventSink?(["tag": tag, "progress": progress.fractionCompleted])
      }
    }
    progressObservations.append(observation)

    request.beginAccessingResources { [weak self] error in
      observation.invalidate()
      self?.progressObservations.removeAll { $0 === observation }
      defer { self?.activeRequests.removeAll { $0 === request } }

      if error != nil {
        result(nil)
        return
      }

      let base = (fileName as NSString).deletingPathExtension
      let ext  = (fileName as NSString).pathExtension
      let url  = Bundle.main.url(
        forResource: base,
        withExtension: ext.isEmpty ? "gguf" : ext
      )
      result(url?.path)
    }
  }

  /// Download all `tags` in a single coordinated ODR session (one progress bar),
  /// then resolve each fileName to its path in the main bundle.
  /// Returns a Map<fileName, filePath> on success, or nil if ODR is unavailable.
  private func requestTagsBatch(_ tags: [String], fileNames: [String], result: @escaping FlutterResult) {
    let request = NSBundleResourceRequest(tags: Set(tags))
    activeRequests.append(request)

    // Report combined progress using the first tag label so the Dart progress
    // listener (which filters by tag name) can display a unified progress bar.
    let progressTag = tags.first ?? "ocula-lite-p1"
    let observation = request.progress.observe(\.fractionCompleted, options: [.new]) { [weak self] progress, _ in
      DispatchQueue.main.async {
        self?.eventSink?(["tag": progressTag, "progress": progress.fractionCompleted])
      }
    }
    progressObservations.append(observation)

    request.beginAccessingResources { [weak self] error in
      observation.invalidate()
      self?.progressObservations.removeAll { $0 === observation }
      defer { self?.activeRequests.removeAll { $0 === request } }

      if error != nil {
        // ODR not configured (dev / TestFlight without App Store processing).
        // Return nil — Dart falls through to network download.
        result(nil)
        return
      }

      // Locate each file in the main bundle and build the result map.
      var pathMap: [String: String] = [:]
      for fileName in fileNames {
        let base = (fileName as NSString).deletingPathExtension
        let ext  = (fileName as NSString).pathExtension
        if let url = Bundle.main.url(
          forResource: base,
          withExtension: ext.isEmpty ? "gguf" : ext
        ) {
          pathMap[fileName] = url.path
        }
      }

      result(pathMap.isEmpty ? nil : pathMap)
    }
  }
}
