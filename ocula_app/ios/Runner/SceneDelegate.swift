import Flutter
import UIKit

class SceneDelegate: UIResponder, UIWindowSceneDelegate {
  var window: UIWindow?
  private let appGroupId = "group.com.finailabz.ai.ocula"
  private let odrChannel = OculaODRChannel()

  // Engine lives for the lifetime of the scene.
  lazy var flutterEngine: FlutterEngine = {
    let engine = FlutterEngine(name: "ocula_engine")
    engine.run()
    GeneratedPluginRegistrant.register(with: engine)
    return engine
  }()

  func scene(
    _ scene: UIScene,
    willConnectTo session: UISceneSession,
    options connectionOptions: UIScene.ConnectionOptions
  ) {
    guard let windowScene = scene as? UIWindowScene else { return }

    let flutterVC = FlutterViewController(engine: flutterEngine, nibName: nil, bundle: nil)

    let window = UIWindow(windowScene: windowScene)
    window.rootViewController = flutterVC
    window.makeKeyAndVisible()
    self.window = window

    // On-Demand Resources channel — provides ODR model paths to Flutter.
    odrChannel.register(with: flutterVC.binaryMessenger)

    // Share Extension channel — messenger is available now that FlutterViewController exists.
    let channel = FlutterMethodChannel(
      name: "com.finailabz.ai.ocula/share",
      binaryMessenger: flutterVC.binaryMessenger
    )
    channel.setMethodCallHandler { [weak self] (call, result) in
      guard let self = self else { return }
      switch call.method {
      case "getPendingShare":
        let content = UserDefaults(suiteName: self.appGroupId)?.string(forKey: "pending_shared_content")
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
  }
}
