import UIKit
import Social
import MobileCoreServices
import UniformTypeIdentifiers

/// Ocula Share Extension — receives shared content from any app.
/// Saves to a shared App Group container so the main Flutter app can pick it up.
class ShareViewController: SLComposeServiceViewController {

    private let appGroupId = "group.com.finailabz.ai.ocula"

    override func isContentValid() -> Bool {
        return true
    }

    override func didSelectPost() {
        handleSharedItems()
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        // Auto-post without showing the compose sheet for a faster experience
        handleSharedItems()
    }

    private func handleSharedItems() {
        guard let items = extensionContext?.inputItems as? [NSExtensionItem] else {
            completeRequest()
            return
        }

        let group = DispatchGroup()
        var sharedData: [[String: String]] = []

        for item in items {
            guard let attachments = item.attachments else { continue }

            for provider in attachments {
                // Text
                if provider.hasItemConformingToTypeIdentifier(UTType.plainText.identifier) {
                    group.enter()
                    provider.loadItem(forTypeIdentifier: UTType.plainText.identifier, options: nil) { data, _ in
                        if let text = data as? String {
                            sharedData.append(["type": "text", "content": text])
                        }
                        group.leave()
                    }
                }

                // URL
                if provider.hasItemConformingToTypeIdentifier(UTType.url.identifier) {
                    group.enter()
                    provider.loadItem(forTypeIdentifier: UTType.url.identifier, options: nil) { data, _ in
                        if let url = data as? URL {
                            sharedData.append(["type": "url", "content": url.absoluteString])
                        }
                        group.leave()
                    }
                }

                // Image
                if provider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
                    group.enter()
                    provider.loadItem(forTypeIdentifier: UTType.image.identifier, options: nil) { data, _ in
                        if let url = data as? URL {
                            sharedData.append(["type": "image", "content": url.path])
                        } else if let image = data as? UIImage,
                                  let imageData = image.jpegData(compressionQuality: 0.8) {
                            // Save to shared container
                            let filename = "shared_\(Int(Date().timeIntervalSince1970)).jpg"
                            if let containerURL = FileManager.default.containerURL(
                                forSecurityApplicationGroupIdentifier: self.appGroupId
                            ) {
                                let fileURL = containerURL.appendingPathComponent(filename)
                                try? imageData.write(to: fileURL)
                                sharedData.append(["type": "image", "content": fileURL.path])
                            }
                        }
                        group.leave()
                    }
                }

                // File (PDF, doc, etc.)
                if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
                    group.enter()
                    provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { data, _ in
                        if let url = data as? URL {
                            // Copy to shared container
                            if let containerURL = FileManager.default.containerURL(
                                forSecurityApplicationGroupIdentifier: self.appGroupId
                            ) {
                                let destURL = containerURL.appendingPathComponent(url.lastPathComponent)
                                try? FileManager.default.copyItem(at: url, to: destURL)
                                sharedData.append([
                                    "type": "file",
                                    "content": destURL.path,
                                    "name": url.lastPathComponent
                                ])
                            }
                        }
                        group.leave()
                    }
                }
            }
        }

        group.notify(queue: .main) {
            // Save to shared UserDefaults for the main app to pick up
            if let defaults = UserDefaults(suiteName: self.appGroupId) {
                // Encode as JSON array
                if let jsonData = try? JSONSerialization.data(withJSONObject: sharedData),
                   let jsonString = String(data: jsonData, encoding: .utf8) {
                    defaults.set(jsonString, forKey: "pending_shared_content")
                    defaults.synchronize()
                }
            }
            self.completeRequest()
        }
    }

    private func completeRequest() {
        extensionContext?.completeRequest(returningItems: nil, completionHandler: nil)
    }
}
