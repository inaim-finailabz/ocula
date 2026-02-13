import 'dart:async';
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
  AIManager._internal() {
    // Listen for tiers that finish downloading in the background.
    // When a better tier becomes available, mark it as pending.
    // The actual switch happens between queries via applyPendingUpgrade().
    _tierReadySub = _models.tierReadyStream.listen(_onTierReady);
  }

  final FlutterLlama _textEngine = FlutterLlama.instance;
  final FlutterLlamaMultimodal _visionEngine = FlutterLlamaMultimodal.instance;
  final OculaModelManager _models = OculaModelManager();

  AITier? _activeTier;
  bool _isSwitching = false;
  final AppLanguage _appLang = AppLanguage();
  int? _deviceRamMB;

  /// The best tier that finished downloading but hasn't been loaded yet.
  /// Set by the tierReadyStream listener, consumed by applyPendingUpgrade().
  AITier? _pendingUpgradeTier;
  StreamSubscription<AITier>? _tierReadySub;

  AITier? get activeTier => _activeTier;
  AITier? get pendingUpgradeTier => _pendingUpgradeTier;
  bool get isModelLoaded => _textEngine.isModelLoaded;

  /// Tier priority: higher index = better model
  static const _tierRank = {
    AITier.free: 0,
    AITier.plus: 1,
    AITier.pro: 2,
    AITier.enterprise: 3,
  };

  /// Minimum RAM (MB) required to safely run each tier.
  /// Includes model weight + KV cache + system overhead headroom.
  static const _tierRamRequirementMB = {
    AITier.free: 1500,       // 175 MB model + context → ~1 GB total, 500 MB headroom
    AITier.plus: 3500,       // 900 MB model + context → ~2.5 GB, 1 GB headroom
    AITier.pro: 5000,        // 1.1 GB model + context → ~3.5 GB, 1.5 GB headroom
    AITier.enterprise: 4000, // variable, conservative default
  };

  /// Check if this device has enough RAM for the given tier.
  Future<bool> canDeviceRunTier(AITier tier) async {
    final ram = await deviceRamMB;
    final required = _tierRamRequirementMB[tier] ?? 8000;
    return ram >= required;
  }

  /// Called when a tier finishes downloading in the background.
  /// Only records it as pending if it's better than what's currently loaded
  /// AND the device has enough RAM to run it.
  void _onTierReady(AITier tier) async {
    final currentRank = _tierRank[_activeTier] ?? -1;
    final pendingRank = _tierRank[_pendingUpgradeTier] ?? -1;
    final newRank = _tierRank[tier] ?? -1;

    if (newRank > currentRank && newRank > pendingRank) {
      // Hardware gate: don't queue upgrades the device can't handle
      if (!await canDeviceRunTier(tier)) {
        debugPrint('[AIManager] ⚠ ${tier.name} downloaded but device RAM '
            'too low (${await deviceRamMB} MB < ${_tierRamRequirementMB[tier]} MB) — skipping');
        return;
      }
      _pendingUpgradeTier = tier;
      debugPrint('[AIManager] 🏁 Pending upgrade queued: ${tier.name} '
          '(current: ${_activeTier?.name ?? "none"}, RAM: ${await deviceRamMB} MB)');
    }
  }

  /// Apply a pending model upgrade if one is ready.
  /// Call this BETWEEN queries (at the top of orchestrator.run())
  /// so we never switch mid-generation.
  ///
  /// Returns true if the model was upgraded, false if no upgrade was pending.
  Future<bool> applyPendingUpgrade() async {
    final target = _pendingUpgradeTier;
    if (target == null) return false;

    // Clear the flag immediately so we don't retry on every query
    _pendingUpgradeTier = null;

    // Already on this tier (or a better one)
    final currentRank = _tierRank[_activeTier] ?? -1;
    final targetRank = _tierRank[target] ?? -1;
    if (targetRank <= currentRank) return false;

    debugPrint('[AIManager] ⬆ Applying pending upgrade: '
        '${_activeTier?.name ?? "none"} → ${target.name}');

    // Hardware gate: double-check RAM before committing to the switch.
    // RAM availability may have changed since the upgrade was queued.
    if (!await canDeviceRunTier(target)) {
      debugPrint('[AIManager] ⬆ ${target.name} upgrade cancelled — device RAM '
          'too low (${await deviceRamMB} MB < ${_tierRamRequirementMB[target]} MB)');
      return false;
    }

    try {
      await switchEngine(target);
      debugPrint('[AIManager] ⬆ Upgrade complete: now on ${target.name}');
      return true;
    } catch (e) {
      debugPrint('[AIManager] ⬆ Upgrade to ${target.name} failed: $e — staying on ${_activeTier?.name}');
      return false;
    }
  }

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

  // ════════════════════════════════════════════════════════════════════
  // CENTRALIZED UNLOAD — every path that needs to release the current
  // model MUST go through this method. Direct unloadModel() /
  // unloadMultimodalModel() calls are forbidden outside of here.
  // ════════════════════════════════════════════════════════════════════

  /// Safely unload the text model if one is loaded.
  ///
  /// - Only acts if a model is actually loaded (isModelLoaded check)
  /// - Swallows errors so callers never crash on unload
  /// - Resets _activeTier
  /// - Waits for Metal GPU drain on iOS/macOS if [waitForMetal] is true
  Future<void> _safeUnload({bool waitForMetal = true}) async {
    if (!_textEngine.isModelLoaded) {
      debugPrint('[AIManager] _safeUnload: no model loaded — skipping');
      _activeTier = null;
      return;
    }
    try {
      await _textEngine.unloadModel();
    } catch (e) {
      debugPrint('[AIManager] _safeUnload failed (non-fatal): $e');
    }
    _activeTier = null;

    if (waitForMetal && (Platform.isIOS || Platform.isMacOS)) {
      debugPrint('[AIManager] Waiting for Metal GPU cleanup...');
      await Future.delayed(const Duration(milliseconds: 1500));
    }
  }

  /// Clear all models from memory (for advanced users).
  /// This will dispose loaded models and free up RAM.
  Future<void> clearMemory() async {
    await _safeUnload();
  }

  /// Switch to enterprise model based on configured settings.
  Future<void> _switchToEnterpriseModel() async {
    final prefs = await SharedPreferences.getInstance();
    final isEnabled = prefs.getBool('enterprise_enabled') ?? false;

    if (!isEnabled) {
      throw Exception('Enterprise mode is not enabled');
    }

    final useLocal = prefs.getBool('enterprise_use_local') ?? true;

    if (useLocal) {
      final modelPath = prefs.getString('enterprise_model_path') ?? '';
      if (modelPath.isEmpty) {
        throw Exception('Enterprise model path not configured');
      }
      final modelFile = File(modelPath);
      if (!await modelFile.exists()) {
        throw Exception('Enterprise model file not found: $modelPath');
      }

      // Native llama_init_model handles atomic free+load
      await _textEngine.loadModel(LlamaConfig(
        modelPath: modelPath,
        nGpuLayers: -1,
        useGpu: true,
      ));
    } else {
      // Remote API — unload any loaded model to free RAM
      await _safeUnload();
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

  /// Check if a tier's main model file is downloaded and on disk.
  /// Does NOT require loading — just checks the file exists.
  Future<bool> isTierDownloaded(AITier tier) async {
    final path = await _models.mainModelPath(tier);
    return path != null;
  }

  Future<void> switchEngine(AITier tier) async {
    if (_activeTier == tier) return;

    // Prevent re-entrant calls — if a switch is already in progress
    // (e.g. from a concurrent applyPendingUpgrade or autoRoute), bail out.
    if (_isSwitching) {
      debugPrint('[AIManager] switchEngine: already switching — ignoring ${tier.name}');
      return;
    }

    _isSwitching = true;
    try {
      if (tier == AITier.enterprise) {
        await _switchToEnterpriseModel();
        return;
      }

      // ══════════════════════════════════════════════════════════════════
      // VIRTUAL-MEMORY PRINCIPLE: validate the replacement model FULLY
      // before touching the currently-loaded model. Never evict a working
      // model unless the replacement is downloaded, on-disk, and valid.
      // ══════════════════════════════════════════════════════════════════

      // 1. Resolve path — null means "not downloaded at all"
      final mainPath = await _models.mainModelPath(tier);
      if (mainPath == null) {
        debugPrint('[AIManager] switchEngine: ${tier.name} model path is null — not downloaded');
        throw ModelNotReadyException(tier);
      }

      // 2. Look up the expected file size from the model registry
      final modelInfo = OculaModelManager.models
          .where((m) => m.tier == tier && !m.isVisionProjector)
          .firstOrNull;
      final expectedSize = modelInfo?.sizeBytes;

      // 3. Full GGUF validation: exists + no .partial + size + magic header
      final isValid = await _models.isValidLocalModel(
        mainPath,
        expectedSizeBytes: expectedSize,
      );
      if (!isValid) {
        debugPrint('[AIManager] switchEngine: ${tier.name} model at $mainPath '
            'failed GGUF validation — not switching');
        throw ModelNotReadyException(tier);
      }

      debugPrint('[AIManager] switchEngine: ${_activeTier?.name ?? "none"} → ${tier.name} '
          '(model validated ✓)');

      // 4. Hardware gate — check RAM before committing to the switch.
      // This prevents unloading a working model only to fail loading the new one.
      if (!await canDeviceRunTier(tier)) {
        final ram = await deviceRamMB;
        final required = _tierRamRequirementMB[tier] ?? 8000;
        debugPrint('[AIManager] ⚠ ${tier.name} rejected — device RAM '
            'too low ($ram MB < $required MB)');
        throw ModelNotReadyException(tier);
      }

      // ── ATOMIC SWAP: let native handle cleanup + load in one call ──
      // The native llama_init_model() handles:
      //   1. Frees old contexts in @autoreleasepool (Metal objects drain)
      //   2. Validates g_model with is_plausible_heap_ptr before freeing
      //   3. Waits 500ms for Metal GPU drain between context and model free
      //   4. Loads the new model
      //
      // We NEVER call unloadModel() from Dart for text→text switches.
      // The native code does it atomically and safely.

      // ── Load the new model ──
      // Tier-appropriate settings:
      //   free/plus: full GPU (-1) + 2048 context — small models, fit easily
      //   pro:       CPU only (0) + 2048 context — avoids Metal OOM on 2B model
      // Using CPU for pro is safe: A17 Pro handles Qwen3-VL-2B Q4_K_M at
      // ~8-12 tok/s on CPU, which is perfectly usable.
      final int gpuLayers;
      final int contextSize;
      if (tier == AITier.pro) {
        gpuLayers = 0;       // CPU-only — no Metal contention
        contextSize = 2048;
      } else {
        gpuLayers = -1;      // all layers on GPU
        contextSize = 2048;
      }

      try {
        await _textEngine.loadModel(LlamaConfig(
          modelPath: mainPath,
          nGpuLayers: gpuLayers,
          useGpu: gpuLayers != 0,
          contextSize: contextSize,
        ));
        _activeTier = tier;
        debugPrint('[AIManager] switchEngine: ${tier.name} loaded ✓');
      } catch (e) {
        debugPrint('[AIManager] loadModel(${tier.name}) failed: $e');
        // Recovery: try to reload free model so we're never stuck with nothing
        if (tier != AITier.free) {
          final freePath = await _models.mainModelPath(AITier.free);
          if (freePath != null) {
            try {
              await _textEngine.loadModel(LlamaConfig(
                modelPath: freePath,
                nGpuLayers: -1,
                useGpu: true,
              ));
              _activeTier = AITier.free;
              debugPrint('[AIManager] Recovered to free tier');
              return;
            } catch (_) {}
          }
        }
        rethrow;
      }
    } finally {
      _isSwitching = false;
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
  /// 2. If image attached, use vision engine (separate native bridge, no
  ///    need to unload text engine — they have independent global state)
  /// 3. Generate response
  ///
  /// [intent] helps tailor the system prompt to the data type (contact,
  /// file, photo, etc.) so the model grounds its answer in phone assets.
  Future<String> ask(String prompt, {
    String context = '',
    bool hasImage = false,
    String? imagePath,
    QueryIntent intent = QueryIntent.chat,
  }) async {
    // Build ChatML-formatted prompt — SmolVLM2 and Qwen3 both use this template
    final langPrefix = _appLang.promptPrefix;

    // ── System prompt: grounded in phone assets when context exists ──
    final String systemMsg;

    if (hasImage && imagePath != null) {
      // Image analysis mode — tell the model to describe what it sees
      systemMsg = '${langPrefix}You are Ocula, a private AI assistant running on the user\'s phone. '
          'The user has shared an image from their device. '
          'Carefully analyze and describe what you see in the image. '
          'Identify objects, text, people, scenes, colors, and any notable details. '
          'If it looks like a document or receipt, extract the key information. '
          'If it looks like a screenshot, describe the content shown. '
          'If it contains text, read and transcribe it. '
          'Be specific and detailed in your description.\n'
          'After describing, ask a relevant follow-up: '
          '"Want me to save this info?", "Should I find related contacts?", '
          '"Need me to extract the text?", "Want to share this?"';
    } else if (context.isNotEmpty) {
      // RAG-grounded mode — phone assets are available
      final intentGuide = _intentGuide(intent);
      systemMsg = '${langPrefix}You are Ocula, a private AI assistant running on the user\'s phone. '
          'You have access to the user\'s REAL phone data below (contacts, files, photos, '
          'calendar events, emails, and previous conversations). '
          'CRITICAL RULES:\n'
          '1. ALWAYS answer using the provided phone data first. Never make up information.\n'
          '2. Include specific details: phone numbers, email addresses, dates, file names, '
          'event times, and contact names exactly as they appear in the data.\n'
          '3. If the data contains the answer, use it directly — do NOT give a generic response.\n'
          '4. If a document or file is in the data, summarize its key points and mention the file name.\n'
          '5. If a contact is found, show their name, phone, and email.\n'
          '6. If the data does NOT answer the question, say so honestly and suggest what to try.\n'
          '$intentGuide'
          'After answering, end with a contextual follow-up question:\n'
          '- For contacts: "Want me to call them?" or "Should I draft a message?"\n'
          '- For files/docs: "Want me to find similar files?" or "Need a summary of another document?"\n'
          '- For calendar: "Should I set a reminder?" or "Want to reschedule?"\n'
          '- For emails: "Want to reply?" or "Should I find related emails?"\n'
          '- For photos: "Want to see more photos from that day?" or "Should I share this?"\n'
          'Be concise, specific, and actionable.';
    } else {
      // No context — general assistant mode
      systemMsg = '${langPrefix}You are Ocula, a private AI assistant running entirely on-device. '
          'The user\'s phone data (contacts, files, calendar, photos, emails) has been indexed '
          'but no matching data was found for this query. '
          'Answer helpfully, but remind the user they can ask about their phone data: '
          '"I can also search your contacts, files, calendar, or photos if needed." '
          'After answering, end with a brief follow-up question or suggestion. '
          'Be concise and helpful.';
    }

    // ── User message: structured for clarity ──
    final String userMsg;
    if (context.isNotEmpty) {
      userMsg = '--- USER\'S PHONE DATA ---\n$context\n--- END DATA ---\n\n'
          'Based on the phone data above, answer this: $prompt';
    } else {
      userMsg = prompt;
    }

    final fullPrompt = '<|im_start|>system\n$systemMsg<|im_end|>\n'
        '<|im_start|>user\n$userMsg<|im_end|>\n'
        '<|im_start|>assistant\n';

    // ── Vision path ──
    // The multimodal bridge (mm_model) is separate from the text bridge
    // (g_model). They don't share Metal buffers. So we can load the vision
    // model without touching the text model. After describing the image we
    // unload only the vision model to free RAM.
    if (imagePath != null) {
      final projPath = await _models.visionProjectorPath(_activeTier ?? AITier.free);
      if (projPath != null) {
        final mainPath = await _models.mainModelPath(_activeTier ?? AITier.free) ?? '';
        try {
          final loaded = await _visionEngine.loadMultimodalModel(
            MultimodalConfig.textAndImage(mainPath, projPath),
          );
          if (loaded) {
            final response = await _visionEngine.describeImage(imagePath, fullPrompt);
            // Free vision model RAM — text engine is untouched
            try { await _visionEngine.unloadMultimodalModel(); } catch (_) {}
            return response.text;
          }
        } catch (e) {
          debugPrint('[AIManager] Vision failed: $e — falling through to text');
          try { await _visionEngine.unloadMultimodalModel(); } catch (_) {}
        }
      }
    }

    // ── Text path ──
    final response = await _textEngine.generate(GenerationParams(
      prompt: fullPrompt,
      maxTokens: 512,
      temperature: 0.7,
      repeatPenalty: 1.3,
      stopSequences: ['<|im_end|>', '<|im_start|>'],
    ));

    var text = response.text;
    text = text.replaceAll('<|im_end|>', '').replaceAll('<|im_start|>', '').trim();
    return text;
  }

  /// Build intent-specific guidance to narrow down the model's response.
  String _intentGuide(QueryIntent intent) {
    switch (intent) {
      case QueryIntent.contact:
        return '7. The user is looking for CONTACT information. '
            'Prioritize showing names, phone numbers, and emails from the data. '
            'Ask: "Is this the contact you were looking for?"\n';
      case QueryIntent.file:
        return '7. The user is asking about a FILE or DOCUMENT on their phone. '
            'Summarize the key content of the file. Mention the file name and type. '
            'If it\'s a long document, provide the main points.\n';
      case QueryIntent.calendar:
        return '7. The user is asking about their CALENDAR or SCHEDULE. '
            'Show exact dates, times, locations, and event names from the data. '
            'Mention if events conflict or are upcoming soon.\n';
      case QueryIntent.email:
        return '7. The user is asking about their EMAILS. '
            'Show sender, subject, date, and key content. '
            'Mention any action items or attachments.\n';
      case QueryIntent.photo:
        return '7. The user is asking about their PHOTOS. '
            'Describe matching photos by date, content, and location if available.\n';
      default:
        return '';
    }
  }

  /// Stop any in-flight generation (text or vision).
  /// Safe to call even if nothing is generating — the native side just
  /// sets g_should_stop / mm_should_stop to true and returns.
  Future<void> stopGeneration() async {
    try {
      await _textEngine.stopGeneration();
    } catch (_) {}
    try {
      await _visionEngine.stopMultimodalGeneration();
    } catch (_) {}
    debugPrint('[AIManager] stopGeneration requested');
  }

  /// Unload everything and free all native memory.
  Future<void> dispose() async {
    await _tierReadySub?.cancel();
    _tierReadySub = null;
    await _safeUnload(waitForMetal: false);
    _pendingUpgradeTier = null;
  }
}
