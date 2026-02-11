import 'package:flutter_llama/flutter_llama.dart';

/// Manages loading/unloading of GGUF models.
/// Only one model is in RAM at a time to prevent OOM crashes on mobile.
class AIManager {
  static final AIManager _instance = AIManager._();
  factory AIManager() => _instance;
  AIManager._();

  FlutterLlama? _engine;
  String? _activeFeature;

  String? get activeFeature => _activeFeature;
  bool get isModelLoaded => _engine != null;

  /// Model file mapping per feature type.
  static const _modelPaths = {
    'quick_scan': 'assets/models/smolvlm.gguf',       // SmolVLM-256M (Free)
    'detail': 'assets/models/moondream.gguf',          // Moondream 0.5B (Plus)
    'document': 'assets/models/qwen2.5-vl.gguf',      // Qwen2-VL-2B (Pro)
    'reasoning': 'assets/models/qwen2.5-vl.gguf',     // Qwen2-VL-2B (Pro)
  };

  /// Load the appropriate model for a given feature.
  /// Clears the previous model from RAM first.
  Future<void> loadFeature(String featureType) async {
    // 1. CLEAR RAM: Avoid crashes on mobile
    if (_engine != null) {
      await _engine!.unloadModel();
      _engine = null;
    }

    // 2. CHOOSE MODEL: Map feature to the right GGUF file
    final modelPath = _modelPaths[featureType] ?? _modelPaths['detail']!;
    final config = LlamaConfig(
      modelPath: modelPath,
      nGpuLayers: 99, // Use all available GPU/NPU layers
    );

    // 3. INITIALIZE: Load into Apple Silicon / Phone NPU
    _engine = FlutterLlama.instance;
    await _engine!.init(config);
    _activeFeature = featureType;
  }

  /// Run a text prompt against the currently loaded model.
  Future<String> ask(String prompt) async {
    if (_engine == null) {
      throw StateError('No model loaded. Call loadFeature() first.');
    }
    final response = await _engine!.complete(prompt);
    return response;
  }

  /// Unload the current model and free all native memory.
  Future<void> dispose() async {
    if (_engine != null) {
      await _engine!.unloadModel();
      _engine = null;
    }
    _activeFeature = null;
  }
}
