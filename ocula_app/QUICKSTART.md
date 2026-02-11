# Ocula — Quick Start Guide

## Prerequisites

- Flutter 3.38+ (`flutter --version`)
- Xcode 16+ (macOS/iOS)
- Android Studio / Android SDK (Android)
- llama.cpp headers (see Setup below)

## Setup (One-time)

```bash
# 1. Clone the project
cd /Volumes/ExternalDisk/projects/Ocula
cd ocula_app

# 2. Clone llama.cpp for native headers (required by flutter_llama)
cd ..
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
ln -sf $(pwd)/llama.cpp ~/.pub-cache/hosted/pub.dev/flutter_llama-1.1.2/llama.cpp

# 3. Install Flutter dependencies
cd ocula_app
flutter pub get

# 4. Download a GGUF model for testing (SmolVLM ~150MB)
# Place .gguf files in assets/models/
# Example: huggingface-cli download HuggingFaceTB/SmolVLM-256M-Instruct --local-dir assets/models/
```

## macOS

```bash
# Build and run
flutter run -d macos

# Or build release
flutter build macos
open build/macos/Build/Products/Release/ocula_app.app
```

**Requirements:**
- macOS 11.0+ deployment target (already set)
- Podfile targets macOS 11.0

## iOS

```bash
# Run on connected iPhone
flutter run -d ios

# Build for release
flutter build ios
```

**Requirements:**
- Xcode with valid signing certificate
- Physical device recommended (AI models are too heavy for simulator)
- Add to `ios/Runner/Info.plist`:
  - `NSMicrophoneUsageDescription` — for voice input
  - `NSCameraUsageDescription` — for camera capture
  - `NSPhotoLibraryUsageDescription` — for gallery access
  - `NSContactsUsageDescription` — for contacts access
  - `NSSpeechRecognitionUsageDescription` — for STT

## Android

```bash
# Run on connected device / emulator
flutter run -d android

# Build APK
flutter build apk

# Build App Bundle (for Play Store)
flutter build appbundle
```

**Requirements:**
- Android SDK 24+ (minSdkVersion)
- NDK installed (for llama.cpp native libs)
- Add to `android/app/src/main/AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.READ_CONTACTS"/>
<uses-permission android:name="android.permission.INTERNET"/>
```

## HarmonyOS NEXT (OHOS)

```bash
# 1. Install the OpenHarmony Flutter fork
git clone https://gitee.com/openharmony-sig/flutter_flutter.git -b dev

# 2. Install DevEco Studio and set env vars
export DEVECO_SDK_HOME=/Applications/DevEco-Studio.app/Contents/sdk
export PATH=$PATH:$DEVECO_SDK_HOME/tools/ohpm/bin

# 3. Add OHOS platform to the project
flutter create --platforms ohos .

# 4. Build
flutter build ohos
```

## Project Structure

```
ocula_app/
├── lib/
│   ├── main.dart                    # App entry + routes + AssistantScreen
│   ├── screens/
│   │   ├── splash_screen.dart       # Lottie splash + AI pre-warming
│   │   └── onboarding_screen.dart   # First-launch privacy onboarding
│   ├── services/
│   │   ├── ai_manager.dart          # Model switching (Free/Plus/Pro tiers)
│   │   ├── orchestrator.dart        # Agent pipeline (Intent→RAG→Route→Generate→Log)
│   │   ├── rag_engine.dart          # On-device vector search
│   │   ├── indexer.dart             # Background data indexer
│   │   ├── local_data.dart          # Access emails, photos, files, contacts
│   │   ├── speech_service.dart      # STT + TTS
│   │   └── export_service.dart      # PDF generation + share
│   └── widgets/
│       └── paywall_gate.dart        # In-app purchase gate
├── assets/
│   ├── models/                      # .gguf model files (gitignored)
│   ├── animations/                  # Lottie JSON files
│   └── images/                      # Splash + icons
├── native/
│   ├── bridge.cpp                   # Raw C++ FFI bridge (optional)
│   └── CMakeLists.txt               # llama.cpp build config
├── ios/
├── android/
├── macos/
├── pubspec.yaml
├── MODEL_STRATEGY.md                # AI model tier strategy
└── QUICKSTART.md                    # This file
```

## Model Files

| Model | File | Size | Tier |
|-------|------|------|------|
| SmolVLM-256M | `smolvlm-256m.gguf` | ~150MB | Free (always loaded) |
| Moondream 0.5B | `moondream2.gguf` | ~300MB | Plus ($2.99) |
| Qwen2.5-VL-3B | `qwen2.5-vl-3b.gguf` | ~2GB | Pro ($4.99/mo) |
| Qwen Projector | `mmproj-model-f16.gguf` | ~600MB | Pro (vision) |

Place these in `assets/models/` for bundled builds, or use on-demand download for production.

## Key Commands

```bash
flutter pub get          # Install dependencies
flutter analyze          # Check for errors
flutter test             # Run tests
flutter run -d macos     # Run on macOS
flutter run -d ios       # Run on iOS device
flutter run -d android   # Run on Android
flutter build macos      # Release build macOS
flutter build ios        # Release build iOS
flutter build apk        # Release APK Android
```
