# Technology Stack

Ocula is built with a modern, cross-platform stack designed for high performance and privacy. This document outlines the key technologies, libraries, and AI models that power the application.

## Core Framework

*   **[Flutter](https://flutter.dev/):** The UI toolkit and application framework. Ocula uses Flutter to build a single, high-performance application for iOS, Android, and macOS from a single Dart codebase.
*   **[Dart](https://dart.dev/):** The programming language used for all application logic in the UI and Service layers.

## AI & Machine Learning

### Inference Engine

*   **[llama.cpp](https://github.com/ggerganov/llama.cpp):** The core AI inference engine. This high-performance C++ library is a key component, enabling Ocula to run large language models efficiently on consumer hardware. It provides optimizations for various hardware, including Apple Silicon (Metal).
*   **[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md):** The model file format used for all of Ocula's AI models. GGUF is a quantized format designed for `llama.cpp` that allows for smaller model sizes and faster inference with minimal performance loss.

### Flutter to Native Bridge

*   **[flutter_llama](https://pub.dev/packages/flutter_llama):** A Flutter plugin that provides a high-level Dart API for `llama.cpp`. Ocula uses a private, local fork of this plugin to ensure stability and implement necessary patches. It uses `dart:ffi` to communicate with the native `llama.cpp` code.

### AI Models

Ocula employs a tiered strategy of different models for different tasks.

| Tier / Role | Model | Format | Key Responsibility |
| :--- | :--- | :--- | :--- |
| **Sensor** | `SmolVLM2-256M` | GGUF (Q8_0) | The always-on, instant-response model for basic queries. |
| **Specialist**| `moondream3-preview` | GGUF (Q4_K_M)| The model for spatial reasoning, counting, and image-based questions. |
| **Thinker** | `Qwen3-VL-2B-Thinking` | GGUF (Q4_K_M)| The primary model for complex reasoning, OCR, and analysis. |

*   **Vision Projectors (`mmproj`):** Each vision-language model is accompanied by a separate multimodal projector file (e.g., `mmproj-Qwen3-VL-2B-Thinking-F16.gguf`). This file is loaded alongside the main model to enable image understanding.

## Application Services & Libraries

### Assistant Features

*   **[speech_to_text](https://pub.dev/packages/speech_to_text):** Enables voice input (STT) by using the platform's built-in speech recognition services.
*   **[flutter_tts](https://pub.dev/packages/flutter_tts):** Provides text-to-speech (TTS) capabilities for speaking responses aloud.

### Data & Communication

*   **[camera](https://pub.dev/packages/camera):** Provides access to the device's camera for real-time vision and capturing images.
*   **[image_picker](https://pub.dev/packages/image_picker):** Allows users to select images from the device's gallery.
*   **[pdf](https://pub.dev/packages/pdf):** Used by the `ExportService` to generate PDF reports of conversations.
*   **[share_plus](https://pub.dev/packages/share_plus):** Enables sharing of exported reports or text with other applications (e.g., Email, Slack).
*   **[path_provider](https://pub.dev/packages/path_provider):** Used to find standard locations on the filesystem for storing downloaded AI models.

### On-Device Search (RAG)

*   **[sqflite](https://pub.dev/packages/sqflite):** A Flutter plugin for SQLite, a self-contained, high-reliability, embedded, full-featured, public-domain, SQL database engine. It will be used to store embeddings for the Retrieval-Augmented Generation (RAG) system.

## Development & Tooling

*   **IDE:** Visual Studio Code or Android Studio.
*   **Build System:** Flutter CLI & Gradle (Android) / Xcode (iOS/macOS).
*   **Version Control:** Git.
*   **Model Downloader:** `huggingface-cli` (from the `huggingface_hub` Python library).
