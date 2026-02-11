import 'dart:io';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter_llama/flutter_llama.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'local_data.dart';
import 'rag_engine.dart';
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
  final LocalData _localData = LocalData();
  final RAGEngine _rag = RAGEngine();
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

    // 1. Flush RAM — unload whatever is currently loaded
    if (_activeTier != null) {
      if (_isVisionMode) {
        await _visionEngine.unloadMultimodalModel();
      } else {
        await _textEngine.unloadModel();
      }
    }

    // 2. Always load via text engine for chat.
    //    Vision engine is loaded on-demand in ask() when an image is attached.
    //    This avoids the problem where FlutterLlamaMultimodal cannot do
    //    text-only generation.
    await _textEngine.loadModel(LlamaConfig(
      modelPath: mainPath,
      nGpuLayers: -1,
      useGpu: true,
    ));
    _isVisionMode = false;

    _activeTier = tier;
  }

  /// Detect what the user wants from their message.
  QueryIntent _detectIntent(String prompt) {
    final lower = prompt.toLowerCase();

    if (lower.contains('search') || lower.contains('google') || lower.contains('look up')) {
      return QueryIntent.web;
    }
    if (lower.contains('email') || lower.contains('inbox') || lower.contains('mail')) {
      return QueryIntent.email;
    }
    if (lower.contains('photo') || lower.contains('picture') || lower.contains('screenshot')) {
      return QueryIntent.photo;
    }
    if (lower.contains('file') || lower.contains('document') || lower.contains('pdf')) {
      return QueryIntent.file;
    }
    if (lower.contains('contact') || lower.contains('phone number') || lower.contains('call')) {
      return QueryIntent.contact;
    }
    if (lower.contains('schedule') || lower.contains('calendar') || lower.contains('meeting')) {
      return QueryIntent.calendar;
    }
    return QueryIntent.chat;
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
  /// 1. Detect intent
  /// 2. RAG search for relevant local data
  /// 3. Build prompt with context → generate response
  Future<String> ask(String prompt, {bool hasImage = false, String? imagePath}) async {
    final intent = _detectIntent(prompt);
    String context = '';

    // RAG search across all indexed local data
    final ragContext = await _rag.getContext(prompt);
    if (ragContext.isNotEmpty) {
      context = ragContext;
    }

    // Web search — ONLY intent that touches the internet
    if (intent == QueryIntent.web) {
      final webResult = await _localData.webSearch(prompt);
      if (webResult.isNotEmpty) {
        context += '\n\n[web] $webResult';
      }
    }

    // Build the full prompt
    final langPrefix = _appLang.promptPrefix;
    final fullPrompt = context.isNotEmpty
        ? '${langPrefix}You are Ocula, a private AI assistant. '
          'Answer based on the user\'s personal data below. '
          'Be concise and helpful.\n\n'
          'Retrieved context:\n$context\n\n'
          'User: $prompt'
        : '${langPrefix}You are Ocula, a private AI assistant. '
          'Be concise and helpful.\n\n'
          'User: $prompt';

    // Generate — if image is attached, hot-swap to vision engine
    if (imagePath != null) {
      final projPath = await _models.visionProjectorPath(_activeTier ?? AITier.free);
      if (projPath != null) {
        // Unload text engine, load vision engine for this query
        await _textEngine.unloadModel();
        await _visionEngine.loadMultimodalModel(
          MultimodalConfig.textAndImage(
            await _models.mainModelPath(_activeTier ?? AITier.free) ?? '',
            projPath,
          ),
        );
        final response = await _visionEngine.describeImage(imagePath, fullPrompt);
        // Swap back to text engine for next query
        await _visionEngine.unloadMultimodalModel();
        await _textEngine.loadModel(LlamaConfig(
          modelPath: await _models.mainModelPath(_activeTier ?? AITier.free) ?? '',
          nGpuLayers: -1,
          useGpu: true,
        ));
        return response.text;
      }
    }

    // Text-only generation
    final response = await _textEngine.generate(GenerationParams(
      prompt: fullPrompt,
      maxTokens: 512,
      temperature: 0.7,
    ));
    return response.text;
  }

  /// Unload everything and free all native memory.
  Future<void> dispose() async {
    if (_textEngine.isModelLoaded) {
      await _textEngine.unloadModel();
    }
    _activeTier = null;
  }
}
