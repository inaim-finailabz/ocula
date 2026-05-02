import Cocoa
import AVFoundation

/// Native camera capture panel for macOS.
/// Opened via the "com.finailabz.ai.ocula/camera" method channel.
class CameraCapture: NSObject, AVCapturePhotoCaptureDelegate {

    private var session: AVCaptureSession?
    private var photoOutput: AVCapturePhotoOutput?
    private var panel: NSPanel?
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var completion: ((String?) -> Void)?
    private var tornDown = false

    // Strong self-reference kept until teardown.
    private static var active: CameraCapture?

    static func present(completion: @escaping (String?) -> Void) {
        // Request camera access first — AVCaptureDevice returns nil without it.
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                guard granted else {
                    let alert = NSAlert()
                    alert.messageText = "Camera Access Denied"
                    alert.informativeText = "Allow camera access in System Settings → Privacy & Security → Camera."
                    alert.alertStyle = .warning
                    alert.addButton(withTitle: "OK")
                    alert.runModal()
                    completion(nil)
                    return
                }
                let capture = CameraCapture()
                capture.completion = completion
                active = capture
                capture.buildUI()
            }
        }
    }

    // MARK: - UI

    private func buildUI() {
        let w: CGFloat = 680
        let h: CGFloat = 540
        let buttonBarH: CGFloat = 60

        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: w, height: h),
            styleMask: [.titled, .closable, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.title = "Take Photo"
        panel.center()
        panel.isReleasedWhenClosed = false
        panel.delegate = self
        self.panel = panel

        guard let content = panel.contentView else { return }

        // Preview container
        let previewContainer = NSView(frame: NSRect(x: 0, y: buttonBarH, width: w, height: h - buttonBarH))
        previewContainer.wantsLayer = true
        previewContainer.layer?.backgroundColor = NSColor.black.cgColor
        content.addSubview(previewContainer)

        // Bottom bar
        let bar = NSView(frame: NSRect(x: 0, y: 0, width: w, height: buttonBarH))
        bar.wantsLayer = true
        bar.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        content.addSubview(bar)

        let cancel = NSButton(frame: NSRect(x: 16, y: 12, width: 90, height: 36))
        cancel.title = "Cancel"
        cancel.bezelStyle = .rounded
        cancel.target = self
        cancel.action = #selector(onCancel)
        bar.addSubview(cancel)

        let captureBtn = NSButton(frame: NSRect(x: (w - 120) / 2, y: 10, width: 120, height: 40))
        captureBtn.title = "Capture"
        captureBtn.bezelStyle = .rounded
        captureBtn.keyEquivalent = "\r"
        captureBtn.target = self
        captureBtn.action = #selector(onCapture)
        bar.addSubview(captureBtn)

        guard setupSession(in: previewContainer) else {
            showNoCameraAlert()
            return
        }

        panel.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    // MARK: - Camera setup

    private func setupSession(in view: NSView) -> Bool {
        let session = AVCaptureSession()
        session.sessionPreset = .photo

        let device: AVCaptureDevice? =
            AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
            ?? AVCaptureDevice.default(for: .video)

        guard let device,
              let input = try? AVCaptureDeviceInput(device: device) else { return false }

        let output = AVCapturePhotoOutput()
        guard session.canAddInput(input), session.canAddOutput(output) else { return false }
        session.addInput(input)
        session.addOutput(output)

        let preview = AVCaptureVideoPreviewLayer(session: session)
        preview.frame = view.bounds
        preview.videoGravity = .resizeAspectFill
        preview.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
        view.layer?.addSublayer(preview)

        self.session = session
        self.photoOutput = output
        self.previewLayer = preview

        DispatchQueue.global(qos: .userInitiated).async { session.startRunning() }
        return true
    }

    // MARK: - Actions

    @objc private func onCapture() {
        guard let output = photoOutput else { return }
        if output.availablePhotoCodecTypes.contains(.jpeg) {
            output.capturePhoto(
                with: AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg]),
                delegate: self
            )
        } else {
            output.capturePhoto(with: AVCapturePhotoSettings(), delegate: self)
        }
    }

    @objc private func onCancel() {
        finish(path: nil)
    }

    // MARK: - AVCapturePhotoCaptureDelegate

    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        guard error == nil, let data = photo.fileDataRepresentation() else {
            finish(path: nil)
            return
        }
        let name = "ocula_capture_\(Int(Date().timeIntervalSince1970)).jpg"
        let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(name)
        do {
            try data.write(to: url)
            finish(path: url.path)
        } catch {
            finish(path: nil)
        }
    }

    // MARK: - Teardown

    /// Single exit point — called exactly once regardless of how the panel closes.
    private func finish(path: String?) {
        guard !tornDown else { return }
        tornDown = true

        // Grab and nil completion before closing the panel, so windowWillClose
        // (which fires synchronously inside panel.close()) cannot double-call it.
        let cb = completion
        completion = nil

        session?.stopRunning()
        session = nil

        let p = panel
        panel = nil          // nil before close so windowWillClose is a no-op
        CameraCapture.active = nil
        p?.close()

        cb?(path)
    }

    private func showNoCameraAlert() {
        let alert = NSAlert()
        alert.messageText = "No Camera Found"
        alert.informativeText = "No camera is available on this Mac."
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        alert.runModal()
        finish(path: nil)
    }
}

// MARK: - NSWindowDelegate — red-X close button
extension CameraCapture: NSWindowDelegate {
    func windowWillClose(_ notification: Notification) {
        // Only acts when the user clicks the red-X directly (panel is still set).
        // When finish() closes the panel, panel is already nil so this is a no-op.
        finish(path: nil)
    }
}
