import 'package:flutter_llama/flutter_llama.dart';
import 'local_data.dart';
import 'rag_engine.dart';
import 'model_manager.dart';

enum AITier { free, plus, pro }

/// Intent detected from the user's query.
enum QueryIntent { chat, photo, email, file, contact, calendar, web }

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

  AITier? get activeTier => _activeTier;
  bool get isModelLoaded => _textEngine.isModelLoaded;

  /// Check if a tier's model is downloaded and ready.
  Future<bool> isTierReady(AITier tier) async {
    final path = await _models.mainModelPath(tier);
    return path != null;
  }

  Future<void> switchEngine(AITier tier) async {
    if (_activeTier == tier) return;

    // Check if model is downloaded
    final mainPath = await _models.mainModelPath(tier);
    if (mainPath == null) {
      // Model not downloaded — can't switch
      return;
    }

    // 1. Flush RAM — unload whatever is currently loaded
    if (_activeTier != null) {
      if (_isVisionMode) {
        await _visionEngine.unloadMultimodalModel();
      } else {
        await _textEngine.unloadModel();
      }
    }

    // 2. Load from downloaded path
    switch (tier) {
      case AITier.free:
        await _textEngine.loadModel(LlamaConfig(
          modelPath: mainPath,
          nGpuLayers: -1,
          useGpu: true,
        ));
        _isVisionMode = false;
        break;

      case AITier.plus:
        await _textEngine.loadModel(LlamaConfig(
          modelPath: mainPath,
          nGpuLayers: -1,
          useGpu: true,
        ));
        _isVisionMode = false;
        break;

      case AITier.pro:
        final projPath = await _models.visionProjectorPath(tier);
        if (projPath == null) return; // Vision projector not downloaded
        await _visionEngine.loadMultimodalModel(
          MultimodalConfig.textAndImage(mainPath, projPath),
        );
        _isVisionMode = true;
        break;
    }

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

  /// Auto-route: pick the right model based on the question.
  Future<void> autoRoute(String prompt, {bool hasImage = false}) async {
    final lower = prompt.toLowerCase();

    final needsPro = lower.contains('why') ||
        lower.contains('how') ||
        lower.contains('explain') ||
        lower.contains('analyze') ||
        lower.contains('compare') ||
        lower.contains('summarize') ||
        lower.contains('contract');

    final needsPlus = hasImage ||
        lower.contains('read') ||
        lower.contains('count') ||
        lower.contains('total') ||
        lower.contains('receipt') ||
        lower.contains('label');

    if (needsPro) {
      await switchEngine(AITier.pro);
    } else if (needsPlus) {
      await switchEngine(AITier.plus);
    }
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
    final fullPrompt = context.isNotEmpty
        ? 'You are Ocula, a private AI assistant. '
          'Answer based on the user\'s personal data below. '
          'Be concise and helpful.\n\n'
          'Retrieved context:\n$context\n\n'
          'User: $prompt'
        : 'You are Ocula, a private AI assistant. '
          'Be concise and helpful.\n\n'
          'User: $prompt';

    // Generate — use vision engine if image is attached and Pro tier
    if (_isVisionMode && imagePath != null) {
      final response = await _visionEngine.describeImage(
        imagePath,
        fullPrompt,
      );
      return response.text;
    }

    final response = await _textEngine.generate(GenerationParams(
      prompt: fullPrompt,
      maxTokens: 512,
      temperature: 0.7,
    ));
    return response.text;
  }

  /// Unload everything and free all native memory.
  Future<void> dispose() async {
    if (_isVisionMode) {
      await _visionEngine.unloadMultimodalModel();
    } else {
      await _textEngine.unloadModel();
    }
    _activeTier = null;
  }
}
