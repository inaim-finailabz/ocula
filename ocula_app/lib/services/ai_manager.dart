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
  bool _isVisionMode = false;
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

      // ── SAFETY: Verify the target model is downloaded BEFORE touching anything ──
      final mainPath = await _models.mainModelPath(tier);
      if (mainPath == null) {
        throw ModelNotReadyException(tier);
      }

      // Verify the file actually exists and is > 1MB (not a corrupt stub)
      final targetFile = File(mainPath);
      if (!targetFile.existsSync() || targetFile.lengthSync() < 1024 * 1024) {
        throw ModelNotReadyException(tier);
      }

      debugPrint('[AIManager] switchEngine: ${_activeTier?.name ?? "none"} → ${tier.name}');
      // ── Hardware gate: check RAM before committing to the switch ──
      // If the device doesn't have enough RAM, refuse the switch entirely.
      // This prevents unloading a working model only to fail loading the new one.
      if (!await canDeviceRunTier(tier)) {
        final ram = await deviceRamMB;
        final required = _tierRamRequirementMB[tier] ?? 8000;
        debugPrint('[AIManager] ⚠ ${tier.name} rejected — device RAM '
            'too low ($ram MB < $required MB)');
        throw ModelNotReadyException(tier);
      }

      // ── Step 1: Explicitly unload the current model ──
      // We MUST separate unload from load so Metal GPU has time to fully
      // reclaim its buffer pool between the two operations. Doing both
      // inside a single native call (llama_init_model) causes the new KV
      // cache allocation to fail because Metal hasn't freed the old buffers.
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
        _activeTier = null;
        _isVisionMode = false;

        // ── Step 2: Wait for Metal GPU to reclaim its buffer pool ──
        // The native bridge uses @autoreleasepool + null-before-free, but
        // Metal command queues drain asynchronously. 1.5s is enough for
        // the A17 Pro to finish in-flight GPU work and reclaim all buffers.
        if (Platform.isIOS || Platform.isMacOS) {
          debugPrint('[AIManager] Waiting for Metal GPU cleanup...');
          await Future.delayed(const Duration(milliseconds: 1500));
        }
      }

      // ── Step 3: Load the new model ──
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
        _isVisionMode = false;
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
              _isVisionMode = false;
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

    // Generate — if image is attached, try to hot-swap to vision engine.
    // Hold _isSwitching to prevent switchEngine / applyPendingUpgrade from
    // racing with this unload→load cycle.
    if (imagePath != null) {
      final projPath = await _models.visionProjectorPath(_activeTier ?? AITier.free);
      if (projPath != null) {
        _isSwitching = true;
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
        } finally {
          _isSwitching = false;
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
    await _tierReadySub?.cancel();
    _tierReadySub = null;
    if (_textEngine.isModelLoaded) {
      await _textEngine.unloadModel();
    }
    _activeTier = null;
    _pendingUpgradeTier = null;
  }
}
