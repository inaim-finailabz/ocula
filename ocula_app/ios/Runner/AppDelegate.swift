import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  private let appGroupId = "group.com.finailabz.ai.ocula"

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)

    // Set up platform channel for Share Extension communication
    let controller = window?.rootViewController as! FlutterViewController
    let channel = FlutterMethodChannel(
      name: "com.finailabz.ai.ocula/share",
      binaryMessenger: controller.binaryMessenger
    )

    channel.setMethodCallHandler { [weak self] (call, result) in
      guard let self = self else { return }

      switch call.method {
      case "getPendingShare":
        let defaults = UserDefaults(suiteName: self.appGroupId)
        let content = defaults?.string(forKey: "pending_shared_content")
        result(content)

      case "clearPendingShare":
        let defaults = UserDefaults(suiteName: self.appGroupId)
        defaults?.removeObject(forKey: "pending_shared_content")
        defaults?.synchronize()
        result(nil)

      default:
        result(FlutterMethodNotImplemented)
      }
    }

    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}
