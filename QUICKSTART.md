# Ocula: Quick Start Guide

This guide provides all the necessary steps to set up your development environment, build, and run the Ocula application.

For a deeper understanding of the project, please refer to our other documentation:
*   **[README.md](README.md):** High-level project overview.
*   **[ARCHITECTURE.md](ARCHITECTURE.md):** Detailed explanation of the app's software architecture.
*   **[TECHNOLOGY.md](TECHNOLOGY.md):** A list of the core technologies and AI models used.
*   **[DEPLOYMENT.md](DEPLOYMENT.md):** Instructions for creating release builds.

## Prerequisites

*   Flutter `3.19` or higher.
*   Xcode `15` or higher (for iOS & macOS).
*   Android Studio & Android SDK `24` or higher (for Android).
*   Python `3.10` or higher (for downloading models).
*   A Hugging Face account and access token for downloading gated models.

## 1. Environment Setup

The setup process involves cloning the Ocula repository and the required `llama.cpp` dependency.

```bash
# 1. Clone the Ocula project
git clone https://github.com/your-username/ocula.git
cd ocula

# 2. Clone llama.cpp into the root directory
# The flutter_llama plugin requires this for its native headers.
git clone https://github.com/ggerganov/llama.cpp.git

# 3. Navigate to the Flutter app directory
cd ocula_app
```

## 2. Install Dependencies & Models

Next, install the Flutter package dependencies and download the AI models.

```bash
# 4. Install Flutter dependencies
flutter pub get

# 5. Download the AI Models using the helper script
# This script uses 'huggingface-cli' to download the required GGUF models.
# You will need to provide a Hugging Face token with access to the gated models.
HUGGING_FACE_HUB_TOKEN=<YOUR_TOKEN> ./fetch_ocula_stack.sh
```

If the script runs successfully, the `ocula_app/assets/models/` directory will be populated with the necessary `.gguf` files.

## 3. Build and Run

You are now ready to run the application.

### macOS
```bash
flutter run -d macos
```

### iOS (Physical Device Required)
```bash
flutter run -d ios
```
**Note:** The AI models are too resource-intensive for the iOS Simulator. You must run on a physical device. Remember to add the required privacy permissions to `ios/Runner/Info.plist` as detailed in the [DEPLOYMENT.md](DEPLOYMENT.md) guide.

### Android
```bash
flutter run -d android
```
**Note:** Remember to add the required permissions to `android/app/src/main/AndroidManifest.xml` as detailed in the [DEPLOYMENT.md](DEPLOYMENT.md) guide.

## Key Development Commands

Here are the most common commands you will use during development.

```bash
# Install all Flutter package dependencies
flutter pub get

# Run all unit and widget tests
flutter test

# Analyze the Dart code for errors and warnings
flutter analyze
```

## Project Structure Overview

The Ocula project is a monorepo containing the main Flutter app and its native dependencies.

```
/
├── ocula_app/             # The main Flutter application
│   ├── lib/
│   │   ├── main.dart      # App entry point and main screen
│   │   ├── services/      # Core application logic (AI, speech, etc.)
│   │   └── screens/       # Other UI screens
│   ├── assets/
│   │   └── models/        # Location for downloaded .gguf models
│   ├── pubspec.yaml
│   └── fetch_ocula_stack.sh # Script to download models
│
├── llama.cpp/             # Native C++ AI engine
│
├── README.md              # Project overview (this file)
├── QUICKSTART.md          # Setup and build guide
├── ARCHITECTURE.md        # System architecture details
├── TECHNOLOGY.md          # Tech stack and models list
└── DEPLOYMENT.md          # Release and deployment guide
```
