import 'dart:io';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_llama/flutter_llama.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'app_language.dart';
import 'model_manager.dart';

enum AITier { free, plus, pro, enterprise }

/// Intent detected from the user's query.
enum QueryIntent { chat, photo, email, file, contact, calendar, web }

class ModelNotReadyException implements Exception {
  final AITier tier;
  ModelNotReadyException(this.tier);

  @override
  String toString() {
    return 'Model for tier ${tier.toString().split('.').last} is not ready.';
  }
}

class AIManager {
  static final AIManager _instance = AIManager._internal();
  factory AIManager() => _instance;
  AIManager._internal();

  final FlutterLlama _textEngine = FlutterLlama.instance;
  final FlutterLlamaMultimodal _visionEngine = FlutterLlamaMultimodal.instance;
  final OculaModelManager _models = OculaModelManager();

  AITier? _activeTier;
  bool _isVisionMode = false;
  final AppLanguage _appLang = AppLanguage();
  int? _deviceRamMB;

  AITier? get activeTier => _activeTier;
  bool get isModelLoaded => _textEngine.isModelLoaded;

  /// Device RAM in MB. Cached after first call.
  Future<int> get deviceRamMB async {
    if (_deviceRamMB != null) return _deviceRamMB!;
    final info = DeviceInfoPlugin();
    if (Platform.isAndroid) {
      final android = await info.androidInfo;
      _deviceRamMB = android.systemFeatures.contains('android.hardware.ram.low')
          ? 2000
          : 6000; // Android doesn't expose exact RAM; heuristic
    } else if (Platform.isIOS) {
      // iOS doesn't expose RAM directly; use model heuristic
      final ios = await info.iosInfo;
      final model = ios.utsname.machine;
      // iPhones with < 4GB: iPhone SE, iPhone 8 and earlier
      _deviceRamMB = model.contains('iPhone9') || model.contains('iPhone8')
          ? 3000
          : 6000;
    } else {
      _deviceRamMB = 8000; // Desktop — assume plenty of RAM
    }
    return _deviceRamMB!;
  }

  /// Clear all models from memory (for advanced users).
  /// This will dispose loaded models and free up RAM.
  Future<void> clearMemory() async {
    try {
      if (_textEngine.isModelLoaded) {
        await _textEngine.unloadModel();
      }
      // Note: FlutterLlamaMultimodal may not have unloadModel method
      // We reset the state and let garbage collection handle it
      _activeTier = null;
      _isVisionMode = false;
    } catch (e) {
      // Ignore disposal errors, just ensure we reset state
      _activeTier = null;
      _isVisionMode = false;
    }
  }

  /// Switch to enterprise model based on configured settings.
  Future<void> _switchToEnterpriseModel() async {
    final prefs = await SharedPreferences.getInstance();
    final isEnabled = prefs.getBool('enterprise_enabled') ?? false;
    
    if (!isEnabled) {
      throw Exception('Enterprise mode is not enabled');
    }

    final useLocal = prefs.getBool('enterprise_use_local') ?? true;
    
    // 1. Flush RAM — unload whatever is currently loaded
    if (_activeTier != null) {
      if (_isVisionMode) {
        await _visionEngine.unloadMultimodalModel();
      } else {
        await _textEngine.unloadModel();
      }
    }

    if (useLocal) {
      // Load local enterprise model
      final modelPath = prefs.getString('enterprise_model_path') ?? '';
      if (modelPath.isEmpty) {
        throw Exception('Enterprise model path not configured');
      }
      
      final modelFile = File(modelPath);
      if (!await modelFile.exists()) {
        throw Exception('Enterprise model file not found: $modelPath');
      }

      // Load the enterprise model (assuming it's a text model for now)
      await _textEngine.loadModel(LlamaConfig(
        modelPath: modelPath,
        nGpuLayers: -1,
        useGpu: true,
      ));
      _isVisionMode = false;
    } else {
      // For remote API models, we don't load anything into memory
      // The actual API calls would be handled in the query methods
      // This is just to mark that enterprise mode is active
    }

    _activeTier = AITier.enterprise;
  }

  /// Check if a tier's model is downloaded and ready.
  Future<bool> isTierReady(AITier tier) async {
    if (tier == AITier.enterprise) {
      final prefs = await SharedPreferences.getInstance();
      final isEnabled = prefs.getBool('enterprise_enabled') ?? false;
      if (!isEnabled) return false;
      
      final useLocal = prefs.getBool('enterprise_use_local') ?? true;
      
      if (useLocal) {
        final modelPath = prefs.getString('enterprise_model_path') ?? '';
        if (modelPath.isEmpty) return false;
        
        final file = File(modelPath);
        return await file.exists();
      } else {
        // For remote API, check if URL and API key are configured
        final modelUrl = prefs.getString('enterprise_model_url') ?? '';
        final apiKey = prefs.getString('enterprise_api_key') ?? '';
        return modelUrl.isNotEmpty && apiKey.isNotEmpty;
      }
    }
    
    final path = await _models.mainModelPath(tier);
    return path != null;
  }

  Future<void> switchEngine(AITier tier) async {
    if (_activeTier == tier) return;

    if (tier == AITier.enterprise) {
      await _switchToEnterpriseModel();
      return;
    }

    // Check if model is downloaded
    final mainPath = await _models.mainModelPath(tier);
    if (mainPath == null) {
      throw ModelNotReadyException(tier);
    }

    // Remember what was loaded so we can restore on failure
    final previousTier = _activeTier;

    // 1. Flush RAM — unload whatever is currently loaded
    if (_activeTier != null) {
      try {
        if (_isVisionMode) {
          await _visionEngine.unloadMultimodalModel();
        } else {
          await _textEngine.unloadModel();
        }
      } catch (e) {
        debugPrint('[AIManager] unload failed (non-fatal): $e');
      }
      // Model is gone from native side regardless
      _activeTier = null;
      _isVisionMode = false;
    }

    // 2. Always load via text engine for chat.
    //    Vision engine is loaded on-demand in ask() when an image is attached.
    try {
      await _textEngine.loadModel(LlamaConfig(
        modelPath: mainPath,
        nGpuLayers: -1,
        useGpu: true,
      ));
      _isVisionMode = false;
      _activeTier = tier;
    } catch (e) {
      debugPrint('[AIManager] loadModel($tier) failed: $e');
      // Try to recover by reloading the free model
      if (tier != AITier.free && previousTier != null) {
        final freePath = await _models.mainModelPath(AITier.free);
        if (freePath != null) {
          try {
            await _textEngine.loadModel(LlamaConfig(
              modelPath: freePath,
              nGpuLayers: -1,
              useGpu: true,
            ));
            _activeTier = AITier.free;
            _isVisionMode = false;
            return; // recovered to free tier
          } catch (_) {}
        }
      }
      rethrow;
    }
  }

  /// Auto-route: pick the right model based on hardware + intent.
  ///
  /// Routing order:
  /// 1. HARDWARE CHECK — low-RAM devices stay on Sensor (free).
  /// 2. INTENT CHECK:
  ///    - Reasoning (why, how, explain, analyze) → Thinker (pro / Qwen3)
  ///    - Spatial  (where, count, find, point)   → Specialist (plus / Moondream 2)
  ///    - Default                                → Sensor (free / SmolVLM2)
  Future<void> autoRoute(String prompt, {bool hasImage = false}) async {
    // 1. Hardware gate — don't crash low-end phones
    final ram = await deviceRamMB;
    if (ram < 4000) {
      await switchEngine(AITier.free);
      return;
    }

    final lower = prompt.toLowerCase();

    // 2. Reasoning intent → Thinker (Qwen3-VL-2B)
    final isReasoning = lower.contains('why') ||
        lower.contains('how') ||
        lower.contains('explain') ||
        lower.contains('analyze') ||
        lower.contains('compare') ||
        lower.contains('summarize') ||
        lower.contains('contract');

    // 3. Spatial intent → Specialist (Moondream 2)
    final isSpatial = lower.contains('where') ||
        lower.contains('count') ||
        lower.contains('find') ||
        lower.contains('point') ||
        lower.contains('total') ||
        lower.contains('receipt') ||
        lower.contains('label');

    if (isReasoning) {
      await switchEngine(AITier.pro);
    } else if (isSpatial || hasImage) {
      await switchEngine(AITier.plus);
    }
    // else: stay on free (Sensor) — already loaded at startup
  }

  /// The main entry point. Full pipeline:
  /// 1. Build ChatML-formatted prompt with system + context + user message
  /// 2. If image attached, hot-swap to vision engine
  /// 3. Generate response
  Future<String> ask(String prompt, {
    String context = '',
    bool hasImage = false,
    String? imagePath,
  }) async {
    // Build ChatML-formatted prompt — SmolVLM2 and Qwen3 both use this template
    final langPrefix = _appLang.promptPrefix;
    final systemMsg = '${langPrefix}You are Ocula, a private AI assistant that runs '
        'entirely on-device. Be concise and helpful. Answer only the user\'s question.';

    final userMsg = context.isNotEmpty
        ? 'Context:\n$context\n\n$prompt'
        : prompt;

    final fullPrompt = '<|im_start|>system\n$systemMsg<|im_end|>\n'
        '<|im_start|>user\n$userMsg<|im_end|>\n'
        '<|im_start|>assistant\n';

    // Generate — if image is attached, try to hot-swap to vision engine
    if (imagePath != null) {
      final projPath = await _models.visionProjectorPath(_activeTier ?? AITier.free);
      if (projPath != null) {
        try {
          final mainPath = await _models.mainModelPath(_activeTier ?? AITier.free) ?? '';
          // Unload text engine, load vision engine for this query
          await _textEngine.unloadModel();
          final loaded = await _visionEngine.loadMultimodalModel(
            MultimodalConfig.textAndImage(mainPath, projPath),
          );
          if (loaded) {
            final response = await _visionEngine.describeImage(imagePath, fullPrompt);
            // Swap back to text engine for next query
            await _visionEngine.unloadMultimodalModel();
            await _textEngine.loadModel(LlamaConfig(
              modelPath: mainPath,
              nGpuLayers: -1,
              useGpu: true,
            ));
            return response.text;
          }
          // Vision engine failed to load — reload text engine and fall through
          await _textEngine.loadModel(LlamaConfig(
            modelPath: mainPath,
            nGpuLayers: -1,
            useGpu: true,
          ));
        } catch (e) {
          // Vision engine not available on this platform — fall through to text-only
          final mainPath = await _models.mainModelPath(_activeTier ?? AITier.free) ?? '';
          if (!_textEngine.isModelLoaded) {
            await _textEngine.loadModel(LlamaConfig(
              modelPath: mainPath,
              nGpuLayers: -1,
              useGpu: true,
            ));
          }
        }
      }
    }

    // Text-only generation
    final response = await _textEngine.generate(GenerationParams(
      prompt: fullPrompt,
      maxTokens: 512,
      temperature: 0.7,
      repeatPenalty: 1.3,
      stopSequences: ['<|im_end|>', '<|im_start|>'],
    ));

    // Strip any trailing template tokens the model might emit
    var text = response.text;
    text = text.replaceAll('<|im_end|>', '').replaceAll('<|im_start|>', '').trim();
    return text;
  }

  /// Unload everything and free all native memory.
  Future<void> dispose() async {
    if (_textEngine.isModelLoaded) {
      await _textEngine.unloadModel();
    }
    _activeTier = null;
  }
}
