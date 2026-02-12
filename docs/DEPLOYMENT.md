# Deploying Ocula

This document provides instructions for building and deploying the Ocula application to supported platforms: **macOS, iOS, and Android**.

Before proceeding, ensure you have completed all the steps in the **[QUICKSTART.md](QUICKSTART.md)** guide, including installing Flutter, setting up `llama.cpp`, and fetching the required AI models.

## Build Flavors & On-Demand Models

For a real-world deployment, you should not bundle all AI models with the app, as this would make the application size very large (2GB+). The recommended strategy is to bundle only the smallest "Free" tier model (`SmolVLM2`) and download the larger "Plus" and "Pro" models on-demand when a user purchases a higher tier.

This project is not yet configured with Flutter Flavors, but for a production app, you would typically have:
*   **`development` flavor:** Bundles all models for easy testing.
*   **`production` flavor:** Bundles only the free-tier model and includes code to download other models from a secure server.

The instructions below are for a single-flavor build that includes any models you have placed in the `assets/models/` directory.

## macOS

Building for macOS is the most straightforward way to test the app on a desktop environment.

**Requirements:**
*   A Mac with Apple Silicon is highly recommended for performance.
*   Xcode 16 or later.
*   The macOS deployment target is set to 11.0.

**Build and Run:**
To run the app in debug mode:
```bash
flutter run -d macos
```

**Build a Release `.app` Bundle:**
To create a distributable application bundle:
```bash
flutter build macos --release
```
The compiled app will be located at `build/macos/Build/Products/Release/ocula_app.app`. You can drag this file to your `Applications` folder.

## iOS

Deploying to a physical iOS device is essential, as the AI models are too resource-intensive to run on the iOS Simulator.

**Requirements:**
*   A physical iPhone or iPad.
*   An Apple Developer account with a valid signing certificate configured in Xcode.
*   **Permissions:** You must add usage descriptions to the `ios/Runner/Info.plist` file for the features Ocula uses.

Add the following keys to your `Info.plist`:
```xml
<key>NSMicrophoneUsageDescription</key>
<string>Ocula needs microphone access to enable voice input.</string>
<key>NSSpeechRecognitionUsageDescription</key>
<string>Ocula needs speech recognition access to transcribe your voice commands on-device.</string>
<key>NSCameraUsageDescription</key>
<string>Ocula needs camera access to analyze images and scenes in real-time.</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>Ocula needs photo library access to let you choose images for analysis.</string>
<key>NSContactsUsageDescription</key>
<string>Ocula needs contacts access to provide context-aware responses about your contacts.</string>
```

**Build and Run:**
To run the app on your connected device:
```bash
flutter run -d ios
```

**Build a Release `.ipa` for TestFlight/App Store:**
1.  **Prepare for Archiving:** Open the project's `ios` folder in Xcode (`open ios/Runner.xcworkspace`).
2.  **Configure Signing:** In the "Signing & Capabilities" tab, select your team and ensure the bundle identifier is unique.
3.  **Archive:** From the Xcode menu, select `Product > Archive`.
4.  **Distribute:** Once the archive is created, the "Organizer" window will appear. From here, you can "Distribute App" to upload it to App Store Connect for TestFlight or App Store submission.

## Android

Deploying to Android requires the Android SDK and NDK.

**Requirements:**
*   Android SDK Platform 24 (`minSdkVersion`) or higher.
*   Android NDK (for `llama.cpp` native libraries). The Flutter build process should handle this automatically if it's installed via Android Studio.
*   **Permissions:** Ensure the following permissions are present in `android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.READ_CONTACTS"/>
<uses-permission android:name="android.permission.INTERNET"/>
```

**Build and Run:**
To run the app on a connected Android device or emulator:
```bash
flutter run -d android
```

**Build a Release `.apk`:**
To create a shareable APK file:
```bash
flutter build apk --release
```
The compiled APK will be located at `build/app/outputs/flutter-apk/app-release.apk`.

**Build an App Bundle for Google Play:**
For submitting to the Google Play Store, you should build an Android App Bundle (`.aab`):
```bash
flutter build appbundle --release
```
The bundle will be located at `build/app/outputs/bundle/release/app-release.aab`. You can upload this file directly to the Play Console.
