# Ocula: The On-Device AI Assistant

Ocula is a multi-agent, privacy-first mobile AI assistant that runs entirely on-device. It leverages a tiered architecture of local vision and language models to provide a range of capabilities, from instant object recognition to complex reasoning, without sending your data to the cloud.

This repository contains the full source code for the Ocula Flutter application.

<p align="center">
  <img src="https-pro-ocr.jpg" width="200" alt="Ocula performing OCR on a document">
  &nbsp; &nbsp; &nbsp;
  <img src="ocula-orb-animation.gif" width="200" alt="Ocula Orb Animation">
  &nbsp; &nbsp; &nbsp;
  <img src="plus-object-detail.jpg" width="200" alt="Ocula identifying an object">
</p>

## ✨ Key Features

*   **100% On-Device & Private:** All AI processing happens locally. Your photos, voice, and data never leave your device.
*   **Tiered AI System:** Ocula automatically switches between multiple AI models based on your query, balancing performance and capability.
    *   **Sensor (Free):** An always-on, tiny model for instant identification.
    *   **Specialist (Plus):** A mid-tier model for spatial reasoning, counting, and details.
    *   **Thinker (Pro):** A powerful model for document analysis, OCR, and complex reasoning.
*   **Multimodal Input:** Interact via text, voice, or your device's camera.
*   **Local Data Indexing (RAG):** Ocula can index your local files, contacts, and photos to provide context-aware answers (coming soon).
*   **Cross-Platform:** Built with Flutter for a native experience on iOS, Android, and macOS.

## 📚 Documentation

This project is documented across several files:

*   **[QUICKSTART.md](QUICKSTART.md):** The main guide for developers. Includes prerequisites, setup, build commands, and project structure.
*   **[ARCHITECTURE.md](ARCHITECTURE.md):** A detailed explanation of the app's architecture, including the tiered AI system and service layer.
*   **[TECHNOLOGY.md](TECHNOLOGY.md):** A list of the core technologies, libraries, and AI models used in the project.
*   **[DEPLOYMENT.md](DEPLOYMENT.md):** Instructions for building and deploying the application to different platforms.

## 🚀 Getting Started

To get started with Ocula development, please see the **[QUICKSTART.md](QUICKSTART.md)** guide.

A brief overview of the setup process:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ocula.git
    cd ocula
    ```
2.  **Set up `llama.cpp`:**
    The project relies on `llama.cpp` for its native AI engine. You'll need to clone it and link it.
    ```bash
    git clone https://github.com/ggerganov/llama.cpp.git
    # Detailed linking instructions in QUICKSTART.md
    ```
3.  **Download Models:**
    Run the fetch script to download the required GGUF models.
    ```bash
    cd ocula_app
    ./fetch_ocula_stack.sh
    ```
4.  **Run the App:**
    ```bash
    flutter pub get
    flutter run
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
