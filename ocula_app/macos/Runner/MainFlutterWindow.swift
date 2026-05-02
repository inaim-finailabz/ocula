import Cocoa
import FlutterMacOS

class MainFlutterWindow: NSWindow {
  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()
    let windowFrame = self.frame
    self.contentViewController = flutterViewController
    self.setFrame(windowFrame, display: true)

    RegisterGeneratedPlugins(registry: flutterViewController)

    let messenger = flutterViewController.engine.binaryMessenger

    // ── Camera channel ────────────────────────────────────────────────────────
    let cameraChannel = FlutterMethodChannel(
      name: "com.finailabz.ai.ocula/camera",
      binaryMessenger: messenger
    )
    cameraChannel.setMethodCallHandler { call, result in
      guard call.method == "capturePhoto" else {
        result(FlutterMethodNotImplemented)
        return
      }
      CameraCapture.present { path in
        DispatchQueue.main.async { result(path) }
      }
    }

    // ── MLX inference method channel ─────────────────────────────────────────
    let mlxMethodChannel = FlutterMethodChannel(
      name: "com.finailabz.ai.ocula/mlx",
      binaryMessenger: messenger
    )
    Task { @MainActor in
      MLXEngine.shared.methodChannel = mlxMethodChannel
    }
    mlxMethodChannel.setMethodCallHandler { call, result in
      Task { @MainActor in
        MLXEngine.shared.handle(call, result: result)
      }
    }

    // ── MLX token stream event channel ───────────────────────────────────────
    let mlxEventChannel = FlutterEventChannel(
      name: "com.finailabz.ai.ocula/mlx_stream",
      binaryMessenger: messenger
    )
    mlxEventChannel.setStreamHandler(MLXStreamHandler())

    super.awakeFromNib()
  }

  // Override performClose: (what the red-X button calls) so the window always
  // hides to the menu bar tray instead of closing/terminating the app.
  // This is the most reliable intercept point — it fires before windowShouldClose:
  // and doesn't depend on window_manager's Dart-side setPreventClose being ready.
  override func performClose(_ sender: Any?) {
    self.orderOut(nil)
  }

  // Prevent close() from destroying the window (e.g. if called programmatically).
  override func close() {
    self.orderOut(nil)
  }
}
