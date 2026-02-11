# Ocula Architecture

Ocula is built on a modular, service-oriented architecture designed to handle the complexities of running multiple AI models on a mobile device. The core principles are separation of concerns, efficient resource management, and a high degree of privacy.

The application can be broken down into three main layers:
1.  **UI Layer (Flutter):** The user-facing interface.
2.  **Service Layer (Dart):** The "brains" of the application, managing state, business logic, and AI operations.
3.  **Native Layer (`llama.cpp`):** The high-performance C++ backend that runs the AI models.

```
┌─────────────────────────────┐
│          UI Layer           │
│    (Flutter Widgets)        │
│ ┌─────────────────────────┐ │
│ │     AssistantScreen     │ │
│ │       (main.dart)       │ │
│ └─────────────────────────┘ │
└─────────────┬─────────────┘
              │
┌─────────────▼─────────────┐
│      Service Layer        │
│          (Dart)           │
│ ┌─────────────────────────┐ │
│ │      Orchestrator       │ │
│ └───────────┬─────────────┘ │
│   ┌─────────▼─────────┐     │
│   │     AIManager     │     │
│   └─────────┬─────────┘     │
│ ┌───────────┴───────────┐   │
│ │ RAG, Speech, Export.. │   │
│ └───────────────────────┘   │
└─────────────┬─────────────┘
              │ (dart:ffi)
┌─────────────▼─────────────┐
│       Native Layer        │
│  (flutter_llama plugin)   │
│ ┌─────────────────────────┐ │
│ │       llama.cpp         │ │
│ └─────────────────────────┘ │
└─────────────────────────────┘
```

## 1. UI Layer

The UI is built entirely in Flutter and is kept as "dumb" as possible. Its primary responsibility is to display state provided by the Service Layer and forward user input events.

*   **`main.dart` (`AssistantScreen`):** The main application screen. It holds the state for the chat transcript, manages UI animations (like the Ocula Orb), and wires up the input controls (text field, camera button, microphone button) to the appropriate services.
*   **Widgets:** The UI is composed of standard Flutter widgets, with a focus on creating a responsive and fluid user experience. The `OculaOrb` is a custom-painted widget that provides visual feedback for the AI's state (idle, listening, thinking, speaking).
*   **Routing:** The app uses Flutter's built-in routing to manage navigation between the splash screen, onboarding, and the main assistant screen.

## 2. Service Layer

This is where the core application logic resides. All services are written in Dart.

### The Orchestrator (`orchestrator.dart`)

The `Orchestrator` is the central coordinator of the AI pipeline. When a user sends a query, the `Orchestrator` is responsible for:
1.  **Intent Detection:** Analyzing the user's prompt to understand the goal (e.g., simple chat, spatial query, complex analysis).
2.  **RAG (Retrieval-Augmented Generation):** (Future) Querying the `RAGEngine` to find relevant local data (contacts, files) to add to the prompt as context.
3.  **Auto-Routing:** Instructing the `AIManager` to load the appropriate AI model based on the detected intent and device hardware.
4.  **Execution:** Calling the `AIManager` to generate a response from the loaded model.
5.  **Logging:** (Future) Recording the interaction for debugging and local history.

### The AI Manager (`ai_manager.dart`)

The `AIManager` is the heart of the AI system. Its key responsibilities are:
*   **Model Management:** It manages a pool of available AI models, defined by `AITier` (Free, Plus, Pro).
*   **Engine Switching:** It handles the loading and unloading of models into memory via the `flutter_llama` plugin. This is the most critical and resource-intensive operation. The `AIManager` ensures that only one model is loaded at a time to conserve RAM.
*   **Hardware Awareness:** It checks the device's RAM to prevent loading large models on low-end devices, thus avoiding crashes.
*   **Prompt Formatting:** It takes the user's raw text and formats it into the specific ChatML template required by the underlying GGUF models.
*   **Vision Handling:** It orchestrates the "hot-swap" to a multimodal vision model when an image is attached to a prompt, loading the necessary vision projector model alongside the main model.

### Other Core Services

*   **`SpeechService`:** Integrates the `speech_to_text` and `flutter_tts` plugins to provide a seamless voice interface.
*   **`ExportService`:** Uses the `pdf` and `share_plus` plugins to generate PDF reports of conversations and share them with other apps.
*   **`Indexer` & `RAGEngine`:** These services will power the on-device search. `Indexer` runs in the background to process and embed local data (contacts, files) into a vector store (`ObjectBox`), and `RAGEngine` provides a search interface for the `Orchestrator`.
*   **`OculaModelManager`:** A utility service that knows where all the model files are located, tracks their download status, and provides paths to the `AIManager`.

## 3. Native Layer

The native layer provides the raw AI compute power.

*   **`llama.cpp`:** The high-performance inference engine written in C++. It is optimized for running Large Language Models (LLMs) on consumer hardware, including CPUs and GPUs via Metal (iOS/macOS) and other backends.
*   **`flutter_llama` Plugin:** This Flutter plugin acts as the bridge between the Dart world and the `llama.cpp` C++ world. It uses `dart:ffi` to call the underlying `llama.cpp` functions.
    *   **Local Fork:** Ocula uses a local, forked version of this plugin (`plugins/flutter_llama`). This allows for custom modifications and ensures stability, as seen with the `_silgen_name` fix mentioned in the `pubspec.yaml`.
*   **`native/bridge.cpp`:** This file appears to be unused scaffolding or a remnant of a previous FFI implementation. The primary native integration happens within the `flutter_llama` plugin itself.

## Data Flow: A User Query

1.  User types "How many cars are in this photo?" and attaches an image.
2.  The `AssistantScreen` captures the text and image file. It calls `_orchestrator.run()`.
3.  The `Orchestrator` sees the image and the keyword "how many," detecting a **spatial/counting intent**.
4.  It calls `_aiManager.autoRoute()` with the prompt and `hasImage: true`.
5.  The `AIManager` checks the device RAM. Assuming it's sufficient, it determines the best model is the **Specialist Tier** (`moondream3`).
6.  It calls `switchEngine(AITier.plus)`.
7.  The `AIManager` unloads the currently active model (e.g., the free `SmolVLM2`) and then loads the `moondream3-q4_k_m.gguf` model into memory via the `flutter_llama` plugin.
8.  The `Orchestrator` then calls `_aiManager.ask()`, providing the prompt and image path.
9.  The `AIManager` loads the corresponding vision projector for Moondream, formats the prompt, and passes everything to the `flutter_llama` plugin.
10. `flutter_llama` uses `dart:ffi` to call `llama.cpp`, which runs the multimodal inference.
11. The generated text is passed back up the chain to the `AssistantScreen`, where it is displayed in a new chat bubble and spoken aloud by the `SpeechService`.
12. Finally, the `AIManager` unloads the vision projector and may revert to a smaller text model to conserve resources.
