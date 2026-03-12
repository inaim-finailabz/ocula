import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_llama/flutter_llama.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'app_language.dart';
import 'model_manager.dart';
import 'rag_engine.dart';
import 'rag_config.dart';
import 'text_utils.dart';

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
  bool? _isEmulator;
  int _loadedTextContextSize = 2048;

  /// The best tier that finished downloading but hasn't been loaded yet.
  /// Set by the tierReadyStream listener, consumed by applyPendingUpgrade().
  AITier? _pendingUpgradeTier;

  /// Prevents concurrent embedding model load attempts.
  bool _isLoadingEmbedding = false;
  StreamSubscription<AITier>? _tierReadySub;

  /// Stream that fires whenever the active tier changes (for UI updates).
  final StreamController<AITier> _activeTierController =
      StreamController<AITier>.broadcast();
  Stream<AITier> get activeTierStream => _activeTierController.stream;

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
    AITier.free: 2000, // 1.28 GB model + 25 MB embed + context → ~2 GB total
    AITier.plus: 4000, // 1.1 GB model + 819 MB mmproj + context → ~3.5 GB
    AITier.pro: 7500,  // 2.5 GB model + 836 MB mmproj + context → needs 8 GB device
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
    // Enterprise and Pro are never auto-upgraded.
    // Enterprise: paid feature — must be explicitly enabled in Settings.
    // Pro: requires 8 GB device; crashes lower-RAM devices if auto-loaded.
    //      User must explicitly activate it from Settings after download.
    if (tier == AITier.enterprise || tier == AITier.pro) {
      debugPrint(
        '[AIManager] ⚠ ${tier.name} tier ready — auto-upgrade disabled '
        '(${tier == AITier.enterprise ? "paid feature" : "on-demand only: requires 8 GB device"})',
      );
      return;
    }

    final currentRank = _tierRank[_activeTier] ?? -1;
    final pendingRank = _tierRank[_pendingUpgradeTier] ?? -1;
    final newRank = _tierRank[tier] ?? -1;

    if (newRank > currentRank && newRank > pendingRank) {
      // Hardware gate: don't queue upgrades the device can't handle
      if (!await canDeviceRunTier(tier)) {
        debugPrint(
          '[AIManager] ⚠ ${tier.name} downloaded but device RAM '
          'too low (${await deviceRamMB} MB < ${_tierRamRequirementMB[tier]} MB) — skipping',
        );
        return;
      }
      _pendingUpgradeTier = tier;
      debugPrint(
        '[AIManager] 🏁 Pending upgrade queued: ${tier.name} '
        '(current: ${_activeTier?.name ?? "none"}, RAM: ${await deviceRamMB} MB)',
      );
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

    debugPrint(
      '[AIManager] ⬆ Applying pending upgrade: '
      '${_activeTier?.name ?? "none"} → ${target.name}',
    );

    // Hardware gate: double-check RAM before committing to the switch.
    // RAM availability may have changed since the upgrade was queued.
    if (!await canDeviceRunTier(target)) {
      debugPrint(
        '[AIManager] ⬆ ${target.name} upgrade cancelled — device RAM '
        'too low (${await deviceRamMB} MB < ${_tierRamRequirementMB[target]} MB)',
      );
      return false;
    }

    try {
      await switchEngine(target);
      debugPrint('[AIManager] ⬆ Upgrade complete: now on ${target.name}');
      return true;
    } catch (e) {
      debugPrint(
        '[AIManager] ⬆ Upgrade to ${target.name} failed: $e — staying on ${_activeTier?.name}',
      );
      return false;
    }
  }

  /// Device RAM in MB. Cached after first call.
  Future<int> get deviceRamMB async {
    if (_deviceRamMB != null) return _deviceRamMB!;
    final info = DeviceInfoPlugin();
    if (Platform.isAndroid) {
      // Read actual RAM from /proc/meminfo (works on real devices + emulators)
      try {
        final meminfo = await File('/proc/meminfo').readAsString();
        final match = RegExp(r'MemTotal:\s+(\d+)\s+kB').firstMatch(meminfo);
        if (match != null) {
          _deviceRamMB = int.parse(match.group(1)!) ~/ 1024;
          debugPrint(
            '[AIManager] Actual device RAM: $_deviceRamMB MB (from /proc/meminfo)',
          );
        }
      } catch (_) {
        // Fall through to heuristic
      }
      if (_deviceRamMB == null) {
        final android = await info.androidInfo;
        _deviceRamMB =
            android.systemFeatures.contains('android.hardware.ram.low')
            ? 2000
            : 6000;
      }
    } else if (Platform.isIOS) {
      // iOS doesn't expose RAM directly; derive from model identifier.
      // Format: iPhone<gen>,<sub> — gen increases with each iPhone generation.
      //
      // RAM by generation (as of 2026):
      //   gen ≥ 17  = iPhone 16 series         → 8 GB
      //   gen == 16 = iPhone 15 Pro/Max         → 8 GB
      //   gen == 15 = iPhone 14 Pro/Max + 15/+  → 6 GB
      //   gen == 14 = iPhone 13/14 mix:
      //     sub 2,3     = iPhone 13 Pro/Max     → 6 GB
      //     sub 7,8     = iPhone 14/Plus        → 6 GB
      //     sub 4,5,6   = iPhone 13/mini/SE3    → 4 GB
      //   gen == 13 = iPhone 12:
      //     sub ≥ 3     = 12 Pro/Max            → 6 GB
      //     sub 1,2     = 12/mini               → 4 GB
      //   gen ≤ 12    = iPhone 11 and older     → 4 GB
      //
      // iPads get 8 GB (conservative — all modern iPads are 8–16 GB).
      final ios = await info.iosInfo;
      final machine = ios.utsname.machine;
      final iPhoneMatch = RegExp(r'^iPhone(\d+),(\d+)$').firstMatch(machine);
      if (iPhoneMatch != null) {
        final gen = int.parse(iPhoneMatch.group(1)!);
        final sub = int.parse(iPhoneMatch.group(2)!);
        if (gen >= 17) {
          _deviceRamMB = 8000; // iPhone 16+
        } else if (gen == 16) {
          _deviceRamMB = 8000; // iPhone 15 Pro / 15 Pro Max
        } else if (gen == 15) {
          _deviceRamMB = 6000; // iPhone 14 Pro/Max, iPhone 15, iPhone 15 Plus
        } else if (gen == 14) {
          // sub 2,3 = 13 Pro/Max (6 GB); sub 7,8 = 14/Plus (6 GB); rest = 4 GB
          _deviceRamMB = (sub == 2 || sub == 3 || sub >= 7) ? 6000 : 4000;
        } else if (gen == 13) {
          // sub 3,4 = 12 Pro/Max (6 GB); sub 1,2 = 12/mini (4 GB)
          _deviceRamMB = (sub >= 3) ? 6000 : 4000;
        } else {
          _deviceRamMB = 4000; // iPhone 12 (gen ≤ 12) and older
        }
      } else {
        _deviceRamMB = 8000; // iPad or simulator — assume plenty
      }
    } else {
      _deviceRamMB = 8000; // Desktop — assume plenty of RAM
    }
    return _deviceRamMB!;
  }

  /// True if running on an Android emulator (goldfish/ranchu kernel, x86 ABI).
  Future<bool> get isEmulator async {
    if (_isEmulator != null) return _isEmulator!;
    if (!Platform.isAndroid) {
      _isEmulator = false;
      return false;
    }
    final info = DeviceInfoPlugin();
    final android = await info.androidInfo;
    _isEmulator = !android.isPhysicalDevice;
    if (_isEmulator!) {
      debugPrint(
        '[AIManager] Running on Android EMULATOR — will use reduced settings',
      );
    }
    return _isEmulator!;
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

  /// Validate enterprise API key format.
  /// Keys must start with 'ocula-ent-' and have a valid base64 payload (min 32 chars).
  static bool isValidEnterpriseKey(String key) {
    if (key.length < 32) return false;
    if (!key.startsWith('ocula-ent-')) return false;
    final payload = key.substring('ocula-ent-'.length);
    if (payload.isEmpty) return false;
    try {
      base64.decode(base64.normalize(payload));
      return true;
    } catch (_) {
      return false;
    }
  }

  /// Switch to enterprise model based on configured settings.
  Future<void> _switchToEnterpriseModel() async {
    final prefs = await SharedPreferences.getInstance();
    final isEnabled = prefs.getBool('enterprise_enabled') ?? false;

    if (!isEnabled) {
      throw Exception('Enterprise mode is not enabled');
    }

    // Gate: require a valid enterprise API key
    final apiKey = prefs.getString('enterprise_api_key') ?? '';
    if (!isValidEnterpriseKey(apiKey)) {
      throw Exception('Invalid or missing enterprise API key');
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
      final ok = await _textEngine.loadModel(
        LlamaConfig(modelPath: modelPath, nGpuLayers: -1, useGpu: true),
      );
      if (!ok) {
        throw ModelNotReadyException(AITier.enterprise);
      }
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

      // Gate: require a valid enterprise API key
      final apiKey = prefs.getString('enterprise_api_key') ?? '';
      if (!isValidEnterpriseKey(apiKey)) return false;

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
      debugPrint(
        '[AIManager] switchEngine: already switching — ignoring ${tier.name}',
      );
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
        debugPrint(
          '[AIManager] switchEngine: ${tier.name} model path is null — not downloaded',
        );
        throw ModelNotReadyException(tier);
      }

      // 2. Look up the expected file size from the model registry
      final modelInfo = OculaModelManager.models
          .where(
            (m) =>
                m.tier == tier && !m.isVisionProjector && !m.isEmbeddingModel,
          )
          .firstOrNull;
      final registryName = modelInfo?.fileName;
      final actualName = p.basename(mainPath);
      // For split GGUF models, model.sizeBytes is the TOTAL across all parts.
      // mainModelPath returns part 1, which is ~1/N of total, so use per-part size.
      int? expectedSize;
      if (registryName != null && registryName == actualName && modelInfo != null) {
        final splitM = RegExp(r'-\d{5}-of-(\d{5})\.gguf$').firstMatch(registryName);
        final parts = splitM != null ? int.parse(splitM.group(1)!) : 1;
        expectedSize = modelInfo.sizeBytes ~/ parts;
      }

      // 3. Full GGUF validation: exists + no .partial + size + magic header
      final isValid = await _models.isValidLocalModel(
        mainPath,
        expectedSizeBytes: expectedSize,
      );
      if (!isValid) {
        debugPrint(
          '[AIManager] switchEngine: ${tier.name} model at $mainPath '
          'failed GGUF validation — not switching',
        );
        throw ModelNotReadyException(tier);
      }

      debugPrint(
        '[AIManager] switchEngine: ${_activeTier?.name ?? "none"} → ${tier.name} '
        '(model validated ✓)',
      );

      // 4. Hardware gate — check RAM before committing to the switch.
      // This prevents unloading a working model only to fail loading the new one.
      if (!await canDeviceRunTier(tier)) {
        final ram = await deviceRamMB;
        final required = _tierRamRequirementMB[tier] ?? 8000;
        debugPrint(
          '[AIManager] ⚠ ${tier.name} rejected — device RAM '
          'too low ($ram MB < $required MB)',
        );
        throw Exception(
          '${OculaModelManager.featureLabel(tier)} requires '
          '${(required / 1024).round()} GB RAM. '
          'This device has ~${(ram / 1024).round()} GB — '
          'an iPhone 15 Pro or newer is recommended.',
        );
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
      // Build platform/tier-specific load attempts.
      final bool emulator = await isEmulator;
      final int nThreads = emulator
          ? 2
          : Platform.numberOfProcessors.clamp(4, 6);

      final List<LlamaConfig> attempts = [];
      if (emulator) {
        final contextSize = tier == AITier.pro ? 2048 : 1024;
        debugPrint(
          '[AIManager] Emulator mode: gpuLayers=0, contextSize=$contextSize',
        );
        attempts.add(
          LlamaConfig(
            modelPath: mainPath,
            nGpuLayers: 0,
            useGpu: false,
            contextSize: contextSize,
            batchSize: 256,
            nThreads: nThreads,
          ),
        );
      } else if (Platform.isAndroid) {
        final ram = await deviceRamMB;
        final proCtx = _proContextSizeForRam(ram);
        final int gpuLayers;
        if (tier == AITier.pro) {
          gpuLayers = 12;
        } else if (tier == AITier.plus) {
          gpuLayers = 16;
        } else {
          gpuLayers = 20;
        }

        // Attempt 1: GPU (Vulkan/OpenCL) — preferred for performance.
        attempts.add(
          LlamaConfig(
            modelPath: mainPath,
            nGpuLayers: gpuLayers,
            useGpu: true,
            contextSize: tier == AITier.pro ? proCtx : 2048,
            batchSize: 256,
            nThreads: nThreads,
          ),
        );

        // Attempt 2: Fewer GPU layers — helps on devices with limited VRAM.
        attempts.add(
          LlamaConfig(
            modelPath: mainPath,
            nGpuLayers: (gpuLayers / 2).floor(),
            useGpu: true,
            contextSize: tier == AITier.pro ? proCtx : 2048,
            batchSize: 128,
            nThreads: nThreads,
          ),
        );

        // Attempt 3: CPU-only — fallback for Nothing OS, HarmonyOS, or any
        // device where the Vulkan/OpenCL driver crashes llama.cpp natively.
        attempts.add(
          LlamaConfig(
            modelPath: mainPath,
            nGpuLayers: 0,
            useGpu: false,
            contextSize: tier == AITier.pro ? 1536 : 2048,
            batchSize: 128,
            nThreads: nThreads,
          ),
        );
      } else {
        final ram = await deviceRamMB;
        final proCtx = _proContextSizeForRam(ram);
        // iOS/macOS: for Pro tier, try progressively safer configs to avoid
        // transient Metal/OOM fallbacks that bounce back to Lite.
        if (tier == AITier.pro) {
          // High-RAM devices: prefer larger context first for better retrieval fidelity.
          if (ram >= 7000) {
            attempts.add(
              LlamaConfig(
                modelPath: mainPath,
                nGpuLayers: -1,
                useGpu: true,
                contextSize: proCtx,
                batchSize: 256,
                nThreads: nThreads.clamp(4, 6),
              ),
            );
            attempts.add(
              LlamaConfig(
                modelPath: mainPath,
                nGpuLayers: 24,
                useGpu: true,
                contextSize: 3072,
                batchSize: 256,
                nThreads: nThreads.clamp(3, 5),
              ),
            );
          }

          // Safe fallbacks.
          attempts.add(
            LlamaConfig(
              modelPath: mainPath,
              nGpuLayers: 12,
              useGpu: true,
              contextSize: 1536,
              batchSize: 128,
              nThreads: nThreads.clamp(3, 4),
            ),
          );
          attempts.add(
            LlamaConfig(
              modelPath: mainPath,
              nGpuLayers: 24,
              useGpu: true,
              contextSize: 2048,
              batchSize: 256,
              nThreads: nThreads.clamp(3, 4),
            ),
          );
          attempts.add(
            LlamaConfig(
              modelPath: mainPath,
              nGpuLayers: 0,
              useGpu: false,
              contextSize: 1536,
              batchSize: 128,
              nThreads: nThreads.clamp(3, 4),
            ),
          );
          attempts.add(
            LlamaConfig(
              modelPath: mainPath,
              nGpuLayers: -1,
              useGpu: true,
              contextSize: 2048,
              batchSize: 256,
              nThreads: nThreads,
            ),
          );
        } else {
          attempts.add(
            LlamaConfig(
              modelPath: mainPath,
              nGpuLayers: -1,
              useGpu: true,
              contextSize: 2048,
              batchSize: 512,
              nThreads: nThreads,
            ),
          );
        }
      }

      Object? lastLoadError;
      var loaded = false;
      LlamaConfig? loadedConfig;
      for (var i = 0; i < attempts.length; i++) {
        final cfg = attempts[i];
        try {
          debugPrint(
            '[AIManager] loadModel(${tier.name}) attempt ${i + 1}/${attempts.length} '
            'gpu=${cfg.nGpuLayers} ctx=${cfg.contextSize} batch=${cfg.batchSize}',
          );
          loaded = await _textEngine.loadModel(cfg);
          if (loaded) {
            loadedConfig = cfg;
            break;
          }
          lastLoadError = ModelNotReadyException(tier);
        } catch (e) {
          lastLoadError = e;
          debugPrint(
            '[AIManager] loadModel(${tier.name}) attempt ${i + 1} failed: $e',
          );
        }
      }

      if (!loaded) {
        debugPrint('[AIManager] loadModel(${tier.name}) failed after retries');
        // Recovery: try to reload free model so we're never stuck with nothing.
        // Important: still throw for the requested tier so callers know switch
        // did not succeed.
        if (tier != AITier.free) {
          final freePath = await _models.mainModelPath(AITier.free);
          if (freePath != null) {
            try {
              final recovered = await _textEngine.loadModel(
                LlamaConfig(modelPath: freePath, nGpuLayers: -1, useGpu: true),
              );
              if (recovered) {
                _activeTier = AITier.free;
                _activeTierController.add(AITier.free);
                debugPrint('[AIManager] Recovered to free tier');
              }
            } catch (_) {}
          }
        }
        if (tier == AITier.pro) {
          throw Exception(
            'Thinker model is incompatible with this iOS llama runtime '
            '(qwen3vl architecture not supported by text engine). '
            'Update flutter_llama/llama.xcframework or use a non-qwen3vl Pro model.',
          );
        }
        throw lastLoadError ?? ModelNotReadyException(tier);
      }

      _activeTier = tier;
      _loadedTextContextSize = loadedConfig?.contextSize ?? 2048;
      _activeTierController.add(tier);
      debugPrint('[AIManager] switchEngine: ${tier.name} loaded ✓');

      // Load dedicated embedding model if available (non-blocking).
      // This runs alongside the text model — separate native instance.
      _loadEmbeddingModelIfAvailable();
    } finally {
      _isSwitching = false;
    }
  }

  /// Load the dedicated embedding model if it's downloaded.
  /// Fire-and-forget — doesn't block text generation.
  /// Called after every tier switch so embedding stays active regardless of
  /// which text model is loaded (lite, plus, or pro).
  void _loadEmbeddingModelIfAvailable() async {
    // Prevent concurrent calls — only one load attempt at a time.
    if (_isLoadingEmbedding) return;
    // Skip if already loaded AND still usable (separate native instance).
    if (_textEngine.isEmbeddingModelLoaded) return;
    _isLoadingEmbedding = true;
    try {
      final path = await _models.embeddingModelPath();
      if (path != null) {
        final ok = await _textEngine.loadEmbeddingModel(path);
        debugPrint('[AIManager] Embedding model loaded: $ok');

        if (ok) {
          // Check if we need to re-index (embedding model changed)
          final rag = RAGEngine();
          await rag.init();
          if (await rag.needsReindex()) {
            debugPrint(
              '[AIManager] Embedding model changed — clearing old vectors for re-index',
            );
            await rag.clear();
            await rag.markEmbeddingModel();
            // The indexer's periodic timer will pick up the re-index automatically
          } else {
            await rag.markEmbeddingModel();
          }
        }
      }
    } catch (e) {
      debugPrint('[AIManager] Embedding model load failed (non-fatal): $e');
    } finally {
      _isLoadingEmbedding = false;
    }
  }

  /// Auto-route: pick the right model based on hardware + intent.
  ///
  /// Routing order:
  /// 1. HARDWARE CHECK — low-RAM devices stay on Ocula Lite (free).
  /// 2. INTENT CHECK:
  ///    - Reasoning (why, how, explain, analyze) → Ocula Pro (Qwen3-VL-2B)
  ///    - Spatial  (where, count, find, point)   → Ocula Plus (Moondream 2)
  ///    - Default                                → Ocula Lite (SmolVLM2-500M)
  Future<void> autoRoute(String prompt, {bool hasImage = false}) async {
    // 1. Hardware gate — don't crash low-end phones
    final ram = await deviceRamMB;
    if (ram < 4000) {
      await switchEngine(AITier.free);
      return;
    }

    final lower = prompt.toLowerCase();

    // 2. Reasoning intent → Ocula Pro (Qwen3-VL-2B)
    final isReasoning =
        lower.contains('why') ||
        lower.contains('how') ||
        lower.contains('explain') ||
        lower.contains('analyze') ||
        lower.contains('compare') ||
        lower.contains('summarize') ||
        lower.contains('contract');

    // 3. Spatial intent → Ocula Plus (Moondream 2)
    final isSpatial =
        lower.contains('where') ||
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
    // else: stay on free (Ocula Lite) — already loaded at startup
  }

  /// The main entry point. Full pipeline:
  /// 1. Build ChatML-formatted prompt with system + context + user message
  /// 2. If image attached, use vision engine (separate native bridge, no
  ///    need to unload text engine — they have independent global state)
  /// 3. Generate response
  ///
  /// [intent] helps tailor the system prompt to the data type (contact,
  /// file, photo, etc.) so the model grounds its answer in phone assets.
  Future<String> ask(
    String prompt, {
    String context = '',
    bool hasImage = false,
    String? imagePath,
    QueryIntent intent = QueryIntent.chat,
  }) async {
    // ── Prompt strategy per model tier ──
    //
    // Ocula Lite (free): 2048 ctx, ~500M params. Better instruction following.
    //   → Ultra-short system prompt, pre-summarized context, low temperature.
    //
    // Ocula Plus (plus): 2048 ctx, ~1.8B params. Better instruction following.
    //   → Short system prompt, more context allowed.
    //
    // Ocula Pro (pro): 4096 ctx, ~2B params. Good instruction following.
    //   → Richer system prompt, full context, higher token budget.
    final langPrefix = _appLang.promptPrefix;
    // Ocula Lite (free/Qwen2.5-1.5B): capable text model — not "small".
    // Ocula Plus (plus/Qwen3-VL-2B) and Pro (pro/Qwen2.5-VL-7B): rich prompts + vision.
    const bool isSmallModel = false;
    final bool isProModel = (_activeTier == AITier.plus ||
        _activeTier == AITier.pro ||
        _activeTier == AITier.enterprise);

    // Pre-summarize context for small models: keep only the most relevant lines
    // to avoid overwhelming the tiny context window.
    final ragConfig = RagConfig();
    final budgetChars = ragConfig.contextBudgetChars;
    String compactContext = _compactContextForTier(
      context,
      query: prompt,
      isSmallModel: isSmallModel,
      isProModel: isProModel,
      budgetChars: budgetChars,
    );

    final String systemMsg;
    String userMsg;
    final rolePrefix =
        '${langPrefix}You are Ocula, an AI assistant with access to the user\'s phone assets via local RAG context. '
        'Use available phone data to help the user, and never invent missing phone data. '
        'When context is available, start with where you found the answer (document, image, contact, email, calendar). '
        'Include key metadata when available: date/time, location, sender, author, file name. '
        'If context contains [Ambiguity], ask one clarifying question instead of guessing. ';

    // Intent-specific data label for context sections
    final String dataLabel = _intentDataLabel(intent);

    if (hasImage && imagePath != null) {
      systemMsg = '${rolePrefix}Describe this image. Be specific and brief.';
      userMsg = prompt;
    } else if (compactContext.isNotEmpty) {
      if (isSmallModel) {
        // Free tier: question-first format so the tiny model knows what to find
        // BEFORE reading the data — dramatically improves relevance on <1B models.
        systemMsg =
            '${rolePrefix}'
            'Answer ONLY from DATA — never invent names, numbers, dates or details. '
            'Check: does DATA answer the QUESTION? '
            'If yes, give the direct factual answer from DATA. '
            'If no, name the source you found (file name, contact, etc.) and say what specific detail is missing.';
        userMsg = 'QUESTION: $prompt\n\nDATA:\n$compactContext\n\nANSWER:';
      } else if (isProModel) {
        // Ocula Pro: can handle richer instructions and structured output
        systemMsg =
            '$rolePrefix'
            'Answer using ONLY the user\'s phone data provided below. '
            'NEVER invent or guess information not in the data.\n'
            'Rules:\n'
            '- Start with "Where I found it:" then cite the source type and reference.\n'
            '- Include specific details: names, phone numbers, dates, times, file names.\n'
            '- Include sender/author/location when present in the data.\n'
            '- Use bullet points for lists of items (contacts, events, files).\n'
            '- For calendar events, always include date and time.\n'
            '- For contacts, include phone number or email when available.\n'
            '- If the data doesn\'t directly answer the question, describe what you DID find (source name, date, type) and what specific information is missing.\n'
            '- End with one short follow-up question the user might ask next.';
        userMsg = '$dataLabel:\n$compactContext\n\nQuestion: $prompt';
      } else {
        // Plus tier: moderate instructions with light structure
        systemMsg =
            '$rolePrefix'
            'Answer using ONLY the data provided. Never invent or guess. '
            'Include specific values: names, numbers, dates, file names. '
            'If the data does not contain the answer, name the source you found and say what specific detail is missing. '
            'Use bullet points for lists. Cite the source type (document, contact, calendar, etc). '
            'Be concise — no filler phrases, no repetition.';
        userMsg = '$dataLabel:\n$compactContext\n\nQ: $prompt';
      }
    } else {
      // No RAG context — general chat.
      // Without data to ground on, models hallucinate more easily.
      // Tell the model explicitly it has no phone data available.
      if (isProModel) {
        systemMsg =
            '$rolePrefix'
            'You currently have no indexed data matching this query. '
            'Do NOT invent contacts, events, or files. '
            'If the user references a specific file or item by name, acknowledge it by name and explain it hasn\'t been indexed yet — suggest they open Settings > Your Data to index it. '
            'For general knowledge questions answer helpfully, concisely and accurately. '
            'If the user is searching phone info, ask one clarifying question: '
            'which source to search — docs, images, contacts, calendar, or email.';
      } else {
        systemMsg =
            '$rolePrefix'
            'No indexed data found for this query. Do not make up information. Be brief. '
            'If the user mentions a specific file or item by name, acknowledge it and suggest indexing it via Settings > Your Data. '
            'Otherwise ask whether to search docs, images, contacts, calendar, or email.';
      }
      userMsg = prompt;
    }

    String fullPrompt =
        '<|im_start|>system\n$systemMsg<|im_end|>\n'
        '<|im_start|>user\n$userMsg<|im_end|>\n'
        '<|im_start|>assistant\n';

    // Hard guard: keep prompt within a safe range for native decode slots.
    // This protects against "failed to find a memory slot for batch".
    final maxPromptChars = _safePromptCharBudget(
      isSmallModel: isSmallModel,
      isProModel: isProModel,
    );
    if (fullPrompt.length > maxPromptChars && compactContext.isNotEmpty) {
      final staticOverhead = fullPrompt.length - compactContext.length;
      final targetCtx = (maxPromptChars - staticOverhead - 120).clamp(
        350,
        1800,
      );
      compactContext = _summarizeContext(compactContext, maxChars: targetCtx);
      userMsg = isProModel
          ? '$dataLabel:\n$compactContext\n\nQuestion: $prompt'
          : isSmallModel
          ? 'QUESTION: $prompt\n\nDATA:\n$compactContext\n\nANSWER:'
          : '$dataLabel:\n$compactContext\n\nQ: $prompt';
      fullPrompt =
          '<|im_start|>system\n$systemMsg<|im_end|>\n'
          '<|im_start|>user\n$userMsg<|im_end|>\n'
          '<|im_start|>assistant\n';
      debugPrint(
        '[AIManager] Prompt clamped: ${fullPrompt.length} chars (ctx=${compactContext.length})',
      );
    }

    // ── Vision path ──
    // The multimodal bridge (mm_model) is separate from the text bridge
    // (g_model). They don't share Metal buffers. So we can load the vision
    // model without touching the text model. After describing the image we
    // unload only the vision model to free RAM.
    if (imagePath != null) {
      final tier = _activeTier ?? AITier.free;
      String unavailableReason = 'The vision model may still be loading.';

      // Free-tier vision is expected to be available at startup.
      // Force a one-time readiness check before reporting a failure.
      if (tier == AITier.free) {
        try {
          await _models.ensureFreeModelReady();
        } catch (_) {}
      }

      final mainPath = await _models.mainModelPath(tier);
      final projPath = await _models.visionProjectorPath(tier);

      debugPrint(
        '[AIManager] Vision path: tier=${tier.name}, main=${mainPath != null ? "found" : "MISSING"}, proj=${projPath != null ? "found" : "MISSING"}',
      );

      if (mainPath == null || projPath == null) {
        if (projPath == null) {
          final visionModel = OculaModelManager.models
              .where((m) => m.tier == tier && m.isVisionProjector)
              .firstOrNull;
          if (visionModel != null) {
            try {
              final status = await _models.getStatus(visionModel.fileName);
              if (status == ModelStatus.notDownloaded) {
                debugPrint(
                  '[AIManager] Vision projector missing for ${tier.name}. Starting background download...',
                );
                unawaited(_models.download(visionModel));
              }
              unavailableReason =
                  'The vision add-on is downloading in the background.';
            } catch (_) {
              unavailableReason = 'The vision model files are not ready yet.';
            }
          } else {
            unavailableReason =
                'This model tier has no vision projector configured.';
          }
        } else {
          unavailableReason = 'The vision model files are not ready yet.';
        }
      } else {
        final mainValid = await _models.isValidLocalModel(mainPath);
        final projValid = await _models.isValidLocalModel(projPath);

        if (!mainValid || !projValid) {
          unavailableReason = 'The vision model is still preparing.';
          debugPrint(
            '[AIManager] Vision invalid GGUF: mainValid=$mainValid projValid=$projValid',
          );
        } else {
          final emulator = await isEmulator;
          final isQwenVision = mainPath.toLowerCase().contains('qwen');
          final extraParams = <String, dynamic>{
            if (emulator) ...{
              'nThreads': 2,
              'nGpuLayers': 0,
              'contextSize': 1024,
              'batchSize': 256,
            },
            if (isQwenVision) 'imageMinTokens': 1024,
          };
          final cfg = MultimodalConfig(
            textModelPath: mainPath,
            mmprojPath: projPath,
            enableVision: true,
            enableAudio: false,
            useGpuForMultimodal: !emulator,
            extraParams: extraParams.isEmpty ? null : extraParams,
          );

          // On some devices the first load can race with native cleanup.
          // Retry once after a short delay before surfacing an error.
          for (var attempt = 1; attempt <= 2; attempt++) {
            try {
              if (_visionEngine.isModelLoaded) {
                await _visionEngine.unloadMultimodalModel();
              }
              debugPrint(
                '[AIManager] Loading vision model attempt $attempt: ${mainPath.split('/').last}',
              );
              final loaded = await _visionEngine.loadMultimodalModel(cfg);
              debugPrint('[AIManager] Vision model loaded: $loaded');
              if (loaded) {
                final primaryParams = GenerationParams(
                  prompt: fullPrompt,
                  maxTokens: 160,
                  temperature: 0.2,
                  repeatPenalty: 1.15,
                  stopSequences: const ['<|im_end|>'],
                );
                var response = await _visionEngine.describeImage(
                  imagePath,
                  fullPrompt,
                  params: primaryParams,
                );
                var visionText = _cleanVisionText(response.text);

                // Some multimodal models return 0 tokens with ChatML wrappers.
                // Retry once with a plain prompt format before failing.
                if (_isLowSignalVisionOutput(visionText, prompt) ||
                    response.tokensGenerated == 0) {
                  debugPrint(
                    '[AIManager] Vision empty output; retrying with plain prompt',
                  );
                  final plainPrompt =
                      'Describe this image. Be specific and brief.\n'
                      'User request: $prompt';
                  final fallbackParams = GenerationParams(
                    prompt: plainPrompt,
                    maxTokens: 160,
                    temperature: 0.2,
                    repeatPenalty: 1.15,
                  );
                  response = await _visionEngine.describeImage(
                    imagePath,
                    plainPrompt,
                    params: fallbackParams,
                  );
                  visionText = _cleanVisionText(response.text);
                }
                debugPrint(
                  '[AIManager] Vision generated ${response.text.length} chars',
                );
                try {
                  await _visionEngine.unloadMultimodalModel();
                } catch (_) {}
                if (!_isLowSignalVisionOutput(visionText, prompt)) {
                  return visionText;
                }
                unavailableReason =
                    'The vision model returned an empty output.';
              } else {
                unavailableReason = 'The vision model failed to initialize.';
              }
            } catch (e) {
              debugPrint('[AIManager] Vision failed (attempt $attempt): $e');
              unavailableReason = 'The vision engine failed to initialize.';
              try {
                await _visionEngine.unloadMultimodalModel();
              } catch (_) {}
            }

            if (attempt == 1) {
              await Future.delayed(const Duration(milliseconds: 700));
            }
          }
        }
      }

      // Vision failed or no projector — DON'T fall through to text-only
      // (text model can't see the image, it would produce garbage).
      debugPrint(
        '[AIManager] Vision unavailable — returning error message: $unavailableReason',
      );
      return 'I couldn\'t process this image right now. '
          '$unavailableReason Please try again in a moment.';
    }

    // ── Text path ──
    // Tier-tuned generation: small models need low temperature to stay coherent,
    // pro models can handle more creativity and longer output.
    // 2026-02 update: increased token budgets, tightened temperatures.
    //   free: 220 tokens max, temp 0.15, repPen 1.4  (was 150/0.1/1.8)
    //   plus: 400 tokens max, temp 0.25, repPen 1.2  (was 300/0.3/1.3)
    //   pro:  512 tokens max, temp 0.35, repPen 1.2  (was 384/0.5/1.3)
    // Bigger budgets prevent answer truncation; lower temps improve accuracy.
    final configMaxTok = ragConfig.maxResponseTokens;
    final int maxTok = isSmallModel
        ? (configMaxTok * 0.25).round().clamp(80, 220)
        : (isProModel
              ? (configMaxTok * 1.33).round().clamp(256, 512)
              : (configMaxTok * 0.6).round().clamp(140, 400));
    final double temp = isSmallModel ? 0.15 : (isProModel ? 0.35 : 0.25);
    final double repPen = isSmallModel ? 1.4 : 1.2;

    LlamaResponse response;
    try {
      response = await _textEngine.generate(
        GenerationParams(
          prompt: fullPrompt,
          maxTokens: maxTok,
          temperature: temp,
          repeatPenalty: repPen,
          // Some models emit <|im_start|> as their first token, which can
          // prematurely stop generation at 0 tokens.
          stopSequences: ['<|im_end|>'],
        ),
      );
    } on PlatformException catch (e) {
      final msg = e.toString().toLowerCase();
      final likelyContextOverflow =
          msg.contains('memory slot') ||
          msg.contains('batch') ||
          msg.contains('context');
      if (!likelyContextOverflow || compactContext.isEmpty) rethrow;

      debugPrint(
        '[AIManager] Generation memory-slot failure; retrying with tighter context',
      );
      final tighterContext = _summarizeContext(
        compactContext,
        maxChars: isProModel ? 1200 : 800,
      );
      final retryUserMsg = tighterContext.isNotEmpty
          ? (isSmallModel
              ? 'QUESTION: $prompt\n\nDATA:\n$tighterContext\n\nANSWER:'
              : '${_intentDataLabel(intent)}:\n$tighterContext\n\nQ: $prompt')
          : prompt;
      final retryPrompt =
          '<|im_start|>system\n$systemMsg<|im_end|>\n'
          '<|im_start|>user\n$retryUserMsg<|im_end|>\n'
          '<|im_start|>assistant\n';

      response = await _textEngine.generate(
        GenerationParams(
          prompt: retryPrompt,
          maxTokens: maxTok,
          temperature: temp,
          repeatPenalty: repPen,
          stopSequences: ['<|im_end|>'],
        ),
      );
      compactContext = tighterContext;
    }

    var text = response.text;
    text = text
        .replaceAll('<|im_end|>', '')
        .replaceAll('<|im_start|>', '')
        .trim();

    // Strip <think>...</think> blocks — all Qwen3 models output them (Lite, Plus, Pro).
    text = text.replaceAll(RegExp(r'<think>.*?</think>', dotAll: true), '').trim();
    // Also strip unclosed <think> blocks (model stopped mid-think).
    final thinkStart = text.indexOf('<think>');
    if (thinkStart >= 0) {
      text = text.substring(0, thinkStart).trim();
    }

    // Strip question-first prompt format artifacts that the model may echo back.
    // e.g. "ANSWER: The answer is..." → "The answer is..."
    text = text.replaceFirst(RegExp(r'^ANSWER:\s*', caseSensitive: false), '');
    text = text.replaceFirst(RegExp(r'^Q:\s*', caseSensitive: false), '');
    text = text.replaceFirst(RegExp(r'^QUESTION:.*\n+DATA:.*\n+', caseSensitive: false, dotAll: true), '');

    // Post-process: strip leaked prompt/context.
    // Some models (especially on CPU) echo the user's data/context verbatim.
    text = stripLeakedContext(text);

    // Recovery for silent outputs (0 tokens / empty text).
    // Some non-ChatML models can stop immediately on the templated prompt.
    if (text.isEmpty) {
      debugPrint(
        '[AIManager] Empty generation detected; retrying with plain prompt format',
      );
      final fallbackPrompt =
          'System: $systemMsg\n\nUser: $userMsg\n\nAssistant:';
      final retry = await _textEngine.generate(
        GenerationParams(
          prompt: fallbackPrompt,
          maxTokens: maxTok,
          temperature: temp,
          repeatPenalty: repPen,
          stopSequences: ['<|im_end|>'],
        ),
      );
      text = retry.text
          .replaceAll('<|im_end|>', '')
          .replaceAll('<|im_start|>', '')
          .trim();
      text = stripLeakedContext(text);
    }

    // Post-process: truncate rambling output.
    // Small models: aggressive truncation at 150 chars.
    // Plus models: truncate at 300 chars.
    // Pro models: truncate at 800 chars.
    if (isSmallModel && text.length > 320) {
      text = _truncateToFirstParagraph(text, maxChars: 320);
    } else if (!isProModel && text.length > 560) {
      text = _truncateToFirstParagraph(text, maxChars: 560);
    } else if (isProModel && text.length > 1200) {
      text = _truncateToFirstParagraph(text, maxChars: 1200);
    }

    // Anti-hallucination: detect and reject off-topic or nonsensical output.
    text = _guardHallucination(text, prompt, compactContext, isSmallModel);
    text = _enforceFactGrounding(text, prompt, compactContext);

    return text;
  }

  /// Streaming version of [ask]. Yields partial text as tokens arrive.
  /// The final yield is the complete, post-processed response.
  /// Falls back to non-streaming [ask] for vision requests.
  Stream<String> askStream(
    String prompt, {
    String context = '',
    bool hasImage = false,
    String? imagePath,
    QueryIntent intent = QueryIntent.chat,
  }) async* {
    // Vision path — no streaming support, fall back to blocking
    if (hasImage && imagePath != null) {
      final result = await ask(
        prompt,
        context: context,
        hasImage: true,
        imagePath: imagePath,
        intent: intent,
      );
      yield result;
      return;
    }

    // Build the prompt identically to ask()
    // Ocula Lite (free): standard prompts. Plus/Pro: rich prompts + think block stripping.
    const bool isSmallModel = false;
    final bool isProModel = (_activeTier == AITier.plus ||
        _activeTier == AITier.pro ||
        _activeTier == AITier.enterprise);
    final ragConfig = RagConfig();
    final langPrefix = _appLang.promptPrefix;
    final budgetChars = ragConfig.contextBudgetChars;

    // Compact context (same logic as ask())
    String compactContext = _compactContextForTier(
      context,
      query: prompt,
      isSmallModel: isSmallModel,
      isProModel: isProModel,
      budgetChars: budgetChars,
    );

    // Build system/user messages (same logic as ask())
    String systemMsg;
    String userMsg;
    final rolePrefix =
        '${langPrefix}You are Ocula, an AI assistant with access to the user\'s phone assets via local RAG context. '
        'Use available phone data to help the user, and never invent missing phone data. '
        'When context is available, start with where you found the answer (document, image, contact, email, calendar). '
        'Include key metadata when available: date/time, location, sender, author, file name. '
        'If context contains [Ambiguity], ask one clarifying question instead of guessing. ';
    if (compactContext.isNotEmpty) {
      final dataLabel = _intentDataLabel(intent);
      if (isSmallModel) {
        // Same question-first format as non-streaming path for consistency
        systemMsg =
            '${rolePrefix}'
            'Answer ONLY from DATA — never invent names, numbers, dates or details. '
            'Check: does DATA answer the QUESTION? '
            'If yes, give the direct factual answer from DATA. '
            'If no, name the source you found (file name, contact, etc.) and say what specific detail is missing.';
        userMsg = 'QUESTION: $prompt\n\nDATA:\n$compactContext\n\nANSWER:';
      } else if (isProModel) {
        systemMsg =
            '$rolePrefix'
            'Answer using ONLY the user\'s phone data provided below. '
            'NEVER invent or guess information not in the data.\n'
            'Rules:\n'
            '- Start with "Where I found it:" then cite the source type and reference.\n'
            '- Include specific details: names, phone numbers, dates, times, file names.\n'
            '- Include sender/author/location when present in the data.\n'
            '- Use bullet points for lists of items (contacts, events, files).\n'
            '- For calendar events, always include date and time.\n'
            '- For contacts, include phone number or email when available.\n'
            '- If the data doesn\'t directly answer the question, describe what you DID find (source name, date, type) and what specific information is missing.\n'
            '- End with one short follow-up question the user might ask next.';
        userMsg = '$dataLabel:\n$compactContext\n\nQuestion: $prompt';
      } else {
        systemMsg =
            '$rolePrefix'
            'Answer using ONLY the data provided. Do not make up information. '
            'If the data does not answer the question, name the source you found and say what specific detail is missing. '
            'Be specific and brief. Use bullet points for lists. '
            'Start with where you found the answer.';
        userMsg = '$dataLabel:\n$compactContext\n\nQ: $prompt';
      }
    } else {
      if (isProModel) {
        systemMsg =
            '$rolePrefix'
            'You currently have no phone data for this query. '
            'Do NOT invent contacts, events, or files. '
            'If the user asks about their phone data, tell them you don\'t '
            'have that information yet and suggest checking Settings > Your Data. '
            'For general questions, answer helpfully and concisely. '
            'If the user is searching phone info, ask one clarifying question: '
            'whether to search docs, images, contacts, calendar, or email.';
      } else {
        systemMsg =
            '$rolePrefix'
            'No phone data available. Do not make up information. Be brief. '
            'If the user is searching phone info, ask whether to search docs, images, contacts, calendar, or email.';
      }
      userMsg = prompt;
    }

    String fullPrompt =
        '<|im_start|>system\n$systemMsg<|im_end|>\n'
        '<|im_start|>user\n$userMsg<|im_end|>\n'
        '<|im_start|>assistant\n';

    final maxPromptChars = _safePromptCharBudget(
      isSmallModel: isSmallModel,
      isProModel: isProModel,
    );
    if (fullPrompt.length > maxPromptChars && compactContext.isNotEmpty) {
      final dataLabel = _intentDataLabel(intent);
      final staticOverhead = fullPrompt.length - compactContext.length;
      final targetCtx = (maxPromptChars - staticOverhead - 120).clamp(
        350,
        1800,
      );
      compactContext = _summarizeContext(compactContext, maxChars: targetCtx);
      userMsg = isProModel
          ? '$dataLabel:\n$compactContext\n\nQuestion: $prompt'
          : isSmallModel
          ? 'QUESTION: $prompt\n\nDATA:\n$compactContext\n\nANSWER:'
          : '$dataLabel:\n$compactContext\n\nQ: $prompt';
      fullPrompt =
          '<|im_start|>system\n$systemMsg<|im_end|>\n'
          '<|im_start|>user\n$userMsg<|im_end|>\n'
          '<|im_start|>assistant\n';
      debugPrint(
        '[AIManager] Stream prompt clamped: ${fullPrompt.length} chars (ctx=${compactContext.length})',
      );
    }

    final configMaxTok = ragConfig.maxResponseTokens;
    final int maxTok = isSmallModel
        ? (configMaxTok * 0.25).round().clamp(80, 220)
        : (isProModel
              ? (configMaxTok * 1.33).round().clamp(256, 512)
              : (configMaxTok * 0.6).round().clamp(140, 400));
    final double temp = isSmallModel ? 0.15 : (isProModel ? 0.35 : 0.25);
    final double repPen = isSmallModel ? 1.4 : 1.2;

    final buffer = StringBuffer();
    await for (final token in _textEngine.generateStream(
      GenerationParams(
        prompt: fullPrompt,
        maxTokens: maxTok,
        temperature: temp,
        repeatPenalty: repPen,
        stopSequences: ['<|im_end|>'],
      ),
    )) {
      buffer.write(token);
      // Yield partial text (cleaned of ChatML artifacts and think blocks).
      var partial = buffer
          .toString()
          .replaceAll('<|im_end|>', '')
          .replaceAll('<|im_start|>', '')
          .trimLeft();
      // Strip complete <think>...</think> blocks, then hide any in-progress thinking.
      partial = partial.replaceAll(RegExp(r'<think>.*?</think>', dotAll: true), '').trimLeft();
      final thinkOpen = partial.indexOf('<think>');
      if (thinkOpen >= 0) partial = partial.substring(0, thinkOpen).trimRight();
      yield partial;
    }

    // Final post-processing on complete text
    var text = buffer
        .toString()
        .replaceAll('<|im_end|>', '')
        .replaceAll('<|im_start|>', '')
        .trim();
    // Strip <think>...</think> blocks — all Qwen3 models output them (Lite, Plus, Pro).
    text = text.replaceAll(RegExp(r'<think>.*?</think>', dotAll: true), '').trim();
    final thinkIdx = text.indexOf('<think>');
    if (thinkIdx >= 0) text = text.substring(0, thinkIdx).trim();
    // Strip question-first prompt format artifacts
    text = text.replaceFirst(RegExp(r'^ANSWER:\s*', caseSensitive: false), '');
    text = text.replaceFirst(RegExp(r'^Q:\s*', caseSensitive: false), '');
    text = stripLeakedContext(text);

    final int maxChars = isSmallModel ? 320 : (isProModel ? 1200 : 560);
    if (text.length > maxChars) {
      text = _truncateToFirstParagraph(text, maxChars: maxChars);
    }
    text = _guardHallucination(text, prompt, compactContext, isSmallModel);
    text = _enforceFactGrounding(text, prompt, compactContext);
    yield text;
  }

  /// Fact-to-source guard:
  /// keeps response fluent but removes factual lines not supported by RAG context.
  String _enforceFactGrounding(String text, String query, String context) {
    if (text.trim().isEmpty || context.trim().isEmpty) return text;

    final lowerContext = context.toLowerCase();
    final sourceLines = context
        .split('\n')
        .map((l) => l.trim())
        .where((l) => l.isNotEmpty)
        .toList();
    final contentLines = sourceLines
        .where((l) => l.toLowerCase().startsWith('content:'))
        .map((l) => l.substring(8).trim().toLowerCase())
        .toList();
    final supportText = (contentLines.isNotEmpty
        ? contentLines.join(' ')
        : lowerContext);

    final parts = text
        .split(RegExp(r'(?<=[.!?])\s+|\n+'))
        .map((p) => p.trim())
        .where((p) => p.isNotEmpty)
        .toList();
    if (parts.length <= 1) return text;

    const stop = {
      'the',
      'and',
      'for',
      'with',
      'that',
      'this',
      'from',
      'have',
      'your',
      'about',
      'there',
      'which',
      'when',
      'where',
      'what',
      'into',
      'also',
      'using',
      'only',
      'data',
      'found',
      'source',
      'answer',
      'brief',
    };

    bool isFactual(String s) {
      final lower = s.toLowerCase();
      if (RegExp(r'\d').hasMatch(lower)) return true;
      if (lower.contains('@')) return true;
      if (RegExp(
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
      ).hasMatch(lower)) {
        return true;
      }
      if (lower.contains('where i found') || lower.startsWith('source')) {
        return true;
      }
      return RegExp(
        r'\b(contact|calendar|email|file|photo|document|sender|location|date|time)\b',
      ).hasMatch(lower);
    }

    bool supported(String s) {
      final words = s
          .toLowerCase()
          .replaceAll(RegExp(r'[^a-z0-9\s@._:/-]'), ' ')
          .split(RegExp(r'\s+'))
          .where((w) => w.length >= 4 && !stop.contains(w))
          .toSet();
      if (words.isEmpty) return true;
      final overlap = words.where((w) => supportText.contains(w)).length;
      return overlap >= 1 || words.length <= 2;
    }

    final kept = <String>[];
    for (final p in parts) {
      if (!isFactual(p) || supported(p)) {
        kept.add(p);
      }
    }

    if (kept.isEmpty) {
      return _fallbackResponse(query, context);
    }
    return kept.join(' ').replaceAll(RegExp(r'\s+'), ' ').trim();
  }

  /// Truncate rambling output to the first meaningful paragraph or [maxChars].
  /// Priority: paragraph break > 2 sentences > hard character limit.
  String _truncateToFirstParagraph(String text, {int maxChars = 200}) {
    // Try splitting on double newline (paragraph break)
    final paraIdx = text.indexOf('\n\n');
    if (paraIdx > 30 && paraIdx <= maxChars) {
      return text.substring(0, paraIdx).trim();
    }

    // No paragraph break — cut after second sentence-ending punctuation
    int sentenceCount = 0;
    final limit = text.length.clamp(
      0,
      maxChars + 50,
    ); // small overrun OK for sentence boundary
    for (int i = 0; i < limit; i++) {
      if (i > 0 &&
          '.!?'.contains(text[i]) &&
          (i + 1 >= text.length || text[i + 1] == ' ' || text[i + 1] == '\n')) {
        sentenceCount++;
        if (sentenceCount >= 2 && i > 40) {
          return text.substring(0, i + 1).trim();
        }
      }
    }

    // Hard character limit — find the last space before maxChars to avoid mid-word cut
    if (text.length > maxChars) {
      final lastSpace = text.lastIndexOf(' ', maxChars);
      if (lastSpace > maxChars ~/ 2) {
        return '${text.substring(0, lastSpace).trim()}…';
      }
      return '${text.substring(0, maxChars).trim()}…';
    }

    return text;
  }

  String _cleanVisionText(String text) {
    var t = text
        .replaceAll('<|im_end|>', '')
        .replaceAll('<|im_start|>', '')
        .trim();
    // Qwen3-VL-Thinking models output <think>...</think> reasoning blocks.
    // Strip them so only the final answer is shown to the user.
    t = t.replaceAll(RegExp(r'<think>.*?</think>', dotAll: true), '').trim();
    // Also strip unclosed <think> blocks (model stopped mid-reasoning).
    final thinkStart = t.indexOf('<think>');
    if (thinkStart >= 0) t = t.substring(0, thinkStart).trim();
    return t;
  }

  bool _isLowSignalVisionOutput(String text, String userPrompt) {
    final lower = text.toLowerCase().trim();
    final promptLower = userPrompt.toLowerCase().trim();
    if (lower.isEmpty) return true;
    if (lower == promptLower) return true;
    if (lower == 'describe' || lower == 'describe image') return true;
    if (lower.startsWith('describe') && lower.length < 40) return true;
    return false;
  }

  /// Detect and replace hallucinated or off-topic responses.
  ///
  /// Small on-device models often:
  /// 1. Generate text completely unrelated to the query
  /// 2. Invent facts not present in the provided context
  /// 3. Continue with random instructions or role-play
  /// 4. Repeat the same phrase in a loop
  ///
  /// This guard catches the worst cases and replaces them with a safe fallback.
  String _guardHallucination(
    String text,
    String query,
    String context,
    bool isSmallModel,
  ) {
    if (text.isEmpty) return text;

    final lower = text.toLowerCase();
    final queryLower = query.toLowerCase();

    // ── 0. Whitelist: valid "no data" responses ──
    // The model correctly says "I don't have that" when the data doesn't
    // answer the question. These responses have ZERO word overlap with the
    // query (no names, no numbers) — but that's expected, not hallucination.
    // Never replace them with a fallback.
    final isValidNullResponse =
        lower.contains("don't have") ||
        lower.contains("do not have") ||
        lower.contains("not in your data") ||
        lower.contains("not available") ||
        lower.contains("couldn't find") ||
        lower.contains("could not find") ||
        lower.contains("no information") ||
        lower.contains("not found") ||
        lower.contains("i don") || // catches "i don't..."
        lower.contains("doesn't contain") ||
        lower.contains("does not contain");
    if (isValidNullResponse) return text;

    // ── 1. Detect repetition loops ──
    // If any 8+ char substring repeats 3+ times, it's a loop.
    if (text.length > 30) {
      for (int len = 8; len <= 30; len++) {
        final chunk = text.substring(0, len).toLowerCase();
        final count = RegExp(
          RegExp.escape(chunk),
          caseSensitive: false,
        ).allMatches(lower).length;
        if (count >= 3) {
          debugPrint('[AIManager] Hallucination: repetition loop detected');
          return _fallbackResponse(query, context);
        }
      }
    }

    // ── 2. Detect role-play / instruction leaks ──
    // Model starts acting as if it's writing a prompt or giving instructions
    // to itself rather than answering the user.
    const rolePlayMarkers = [
      'as an ai',
      'as a language model',
      'i am a helpful',
      'i am an ai',
      'sure! here',
      'of course!',
      'instructions:',
      'system:',
      'user:',
      'assistant:',
      '<|im_start|>',
      '<|im_end|>',
      '<s>',
      '</s>',
      'write a',
      'create a story',
      'once upon a time',
    ];
    for (final marker in rolePlayMarkers) {
      if (lower.startsWith(marker)) {
        debugPrint(
          '[AIManager] Hallucination: role-play/leak marker "$marker"',
        );
        return _fallbackResponse(query, context);
      }
    }

    // ── 3. Detect zero relevance (no query words in response) ──
    // Extract meaningful words from the query (3+ chars, not stop words)
    const stopWords = {
      'the',
      'and',
      'for',
      'are',
      'but',
      'not',
      'you',
      'all',
      'can',
      'had',
      'her',
      'was',
      'one',
      'our',
      'out',
      'day',
      'has',
      'his',
      'how',
      'its',
      'may',
      'new',
      'now',
      'old',
      'see',
      'way',
      'who',
      'did',
      'get',
      'let',
      'say',
      'she',
      'too',
      'use',
      'what',
      'when',
      'where',
      'which',
      'will',
      'with',
      'does',
      'have',
      'this',
      'that',
      'from',
      'they',
      'been',
      'some',
      'than',
      'them',
      'then',
      'were',
      'about',
      'could',
      'would',
      'should',
      'there',
      'their',
      'these',
      'those',
      'being',
      'other',
      'after',
      'before',
      'into',
      'just',
      'like',
      'make',
      'many',
      'also',
      'most',
      'much',
      'only',
      'over',
      'such',
      'take',
      'very',
      'come',
      'tell',
      'show',
      'find',
      'give',
      'know',
      'want',
      'look',
      'need',
      'think',
    };
    final queryWords = queryLower
        .replaceAll(RegExp(r'[^a-z\s]'), '')
        .split(RegExp(r'\s+'))
        .where((w) => w.length >= 3 && !stopWords.contains(w))
        .toSet();

    if (queryWords.length >= 2) {
      final matchCount = queryWords.where((w) => lower.contains(w)).length;
      final matchRatio = matchCount / queryWords.length;

      // Also check if response references anything from the RAG context
      final contextLower = context.toLowerCase();
      final hasContextOverlap =
          contextLower.isNotEmpty &&
          contextLower
                  .split(RegExp(r'\s+'))
                  .where((w) => w.length >= 4 && lower.contains(w))
                  .length >=
              2;

      // If response shares no words with query AND no overlap with context,
      // it's very likely hallucinated.
      if (matchRatio == 0 && !hasContextOverlap && text.length > 20) {
        debugPrint(
          '[AIManager] Hallucination: zero relevance '
          '(queryWords=$queryWords, matchCount=$matchCount)',
        );
        return _fallbackResponse(query, context);
      }
    }

    // ── 4. Detect invented data when context is empty ──
    // If no RAG context was provided but the model produces specific names,
    // phone numbers, or dates, it's fabricating.
    if (context.isEmpty && isSmallModel) {
      final hasPhoneNumber = RegExp(
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
      ).hasMatch(text);
      final hasEmail = RegExp(r'\w+@\w+\.\w+').hasMatch(text);
      if (hasPhoneNumber || hasEmail) {
        debugPrint(
          '[AIManager] Hallucination: invented contact details without context',
        );
        return 'I don\'t have enough data to answer that. '
            'Try asking about your contacts, calendar, or files '
            'after indexing is complete.';
      }
    }

    return text;
  }

  /// Generate a safe fallback when hallucination is detected.
  /// If RAG context exists, attempt a minimal extractive answer.
  String _fallbackResponse(String query, String context) {
    if (context.isEmpty) {
      return 'I\'m not sure how to answer that. Could you rephrase your question?';
    }
    // Try to extract the most relevant line from context as a simple answer
    final queryWords = query
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z\s]'), '')
        .split(RegExp(r'\s+'))
        .where((w) => w.length >= 3)
        .toSet();
    final lines = context
        .split('\n')
        .where((l) => l.trim().length > 10)
        .toList();

    // Score each line by how many query words it contains
    var bestLine = '';
    var bestScore = 0;
    for (final line in lines) {
      final lineLower = line.toLowerCase();
      final score = queryWords.where((w) => lineLower.contains(w)).length;
      if (score > bestScore) {
        bestScore = score;
        bestLine = line.trim();
      }
    }

    if (bestLine.isNotEmpty && bestScore >= 1) {
      // Clean up context labels
      return bestLine
          .replaceAll(
            RegExp(r'^(CONTACT|CALENDAR EVENT|FILE|PHOTO|EMAIL): '),
            '',
          )
          .trim();
    }

    return 'I found some data but couldn\'t form a clear answer. '
        'Try rephrasing your question.';
  }

  /// Pre-summarize RAG context to fit in a small model's context window.
  ///
  /// Strategy: keep lines that contain the most information density.
  /// Prioritize lines with names, numbers, dates, and file references.
  /// Drop filler lines like "[Recent conversations]" headers.
  String _summarizeContext(String context, {int maxChars = 600}) {
    if (context.length <= maxChars) return context;

    final lines = context
        .split('\n')
        .where((l) => l.trim().isNotEmpty)
        .toList();

    // Score each line by information density so the most factual lines
    // survive context compression, not just the first-N.
    int scoreLine(String line) {
      final t = line.trim();
      if (t.startsWith('[SOURCE')) { return 100; }
      if (t.startsWith('Content:')) { return 80; }
      if (t.startsWith('Reference:') || t.startsWith('Type:')) { return 70; }
      if (t.startsWith('Date:')) { return 65; }
      if (t.startsWith('[Recent') || t.startsWith('[Knowledge') ||
          t.startsWith('[Note:')) { return 0; }
      int score = 10;
      if (RegExp(r'\d').hasMatch(t)) { score += 15; }
      if (RegExp(r'\d{1,2}[/\-:]\d{1,2}').hasMatch(t)) { score += 10; }
      if (RegExp(r'@|\+?\d[\d\s\-]{6,}').hasMatch(t)) { score += 12; }
      if (RegExp(r'[A-Z][a-z]+ [A-Z][a-z]+').hasMatch(t)) { score += 8; }
      if (t.length > 40) { score += 5; }
      return score;
    }

    // Sort by score descending, then fill to budget
    final scored = lines
        .where((l) => l.trim().length > 5)
        .map((l) => (line: l, score: scoreLine(l)))
        .where((e) => e.score > 0)
        .toList()
      ..sort((a, b) => b.score.compareTo(a.score));

    final kept = <String>[];
    int total = 0;
    for (final e in scored) {
      if (total + e.line.length + 1 > maxChars) break;
      kept.add(e.line);
      total += e.line.length + 1;
    }

    // Re-sort kept lines to restore original document order
    final lineOrder = {for (var i = 0; i < lines.length; i++) lines[i]: i};
    kept.sort((a, b) =>
        (lineOrder[a] ?? 999).compareTo(lineOrder[b] ?? 999));

    return kept.join('\n').trim();
  }

  /// Keep prompts within safe limits per tier to avoid native decode failures.
  String _compactContextForTier(
    String context, {
    required String query,
    required bool isSmallModel,
    required bool isProModel,
    required int budgetChars,
  }) {
    if (context.isEmpty) return context;

    if (isSmallModel) {
      return _summarizeContext(context, maxChars: (budgetChars * 0.68).round());
    }

    if (isProModel) {
      // Hierarchical context packing for larger Pro windows:
      // keep top sources verbatim, summarize overflow with source refs.
      final proBudget =
          (_safePromptCharBudget(isSmallModel: false, isProModel: true) * 0.7)
              .round()
              .clamp(1800, 4600);
      return _packHierarchicalContext(
        context,
        query: query,
        maxChars: proBudget,
      );
    }

    return _summarizeContext(context, maxChars: budgetChars);
  }

  int _safePromptCharBudget({
    required bool isSmallModel,
    required bool isProModel,
  }) {
    if (isSmallModel) return 2000;
    if (!isProModel) return 2600;
    if (_loadedTextContextSize >= 4096) return 5600;
    if (_loadedTextContextSize >= 3072) return 4400;
    return 3400;
  }

  int _proContextSizeForRam(int ramMB) {
    if (ramMB >= 9000) return 4096;
    if (ramMB >= 7000) return 3072;
    return 2048;
  }

  String _packHierarchicalContext(
    String context, {
    required String query,
    required int maxChars,
  }) {
    if (context.length <= maxChars) return context;

    final blocks = context
        .split(RegExp(r'\n{2,}(?=\[SOURCE |\[Linked entities\]|\[Ambiguity\])'))
        .map((b) => b.trim())
        .where((b) => b.isNotEmpty)
        .toList();
    if (blocks.isEmpty) return _summarizeContext(context, maxChars: maxChars);

    final qWords = query
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9\s]'), ' ')
        .split(RegExp(r'\s+'))
        .where((w) => w.length >= 3)
        .toSet();

    int scoreBlock(String b) {
      final lower = b.toLowerCase();
      var s = 0;
      for (final w in qWords) {
        if (lower.contains(w)) s++;
      }
      if (lower.startsWith('[source ')) s += 2;
      if (lower.contains('content:')) s += 2;
      if (lower.contains('date:')) s += 1;
      return s;
    }

    final ranked = [...blocks]
      ..sort((a, b) => scoreBlock(b).compareTo(scoreBlock(a)));

    final kept = <String>[];
    final overflow = <String>[];
    var used = 0;

    for (final b in ranked) {
      if (used + b.length + 2 <= maxChars * 0.75) {
        kept.add(b);
        used += b.length + 2;
      } else {
        overflow.add(b);
      }
    }

    if (overflow.isNotEmpty) {
      final summaryLines = <String>[];
      for (final b in overflow.take(8)) {
        final ref =
            RegExp(
              r'Reference:\s*(.+)$',
              multiLine: true,
            ).firstMatch(b)?.group(1) ??
            'source';
        final content =
            RegExp(
              r'Content:\s*(.+)$',
              multiLine: true,
            ).firstMatch(b)?.group(1) ??
            b;
        final short = content.length > 140
            ? '${content.substring(0, 140)}...'
            : content;
        summaryLines.add('- $ref: $short');
      }
      kept.add('[Source summaries]\n${summaryLines.join('\n')}');
    }

    final packed = kept.join('\n\n');
    return packed.length <= maxChars
        ? packed
        : _summarizeContext(packed, maxChars: maxChars);
  }

  /// Human-readable label for the data section based on query intent.
  /// Helps the LLM understand what type of phone data it's looking at.
  String _intentDataLabel(QueryIntent intent) {
    switch (intent) {
      case QueryIntent.contact:
        return 'Contact information from your phone';
      case QueryIntent.calendar:
        return 'Calendar events from your schedule';
      case QueryIntent.file:
        return 'Files on your device';
      case QueryIntent.photo:
        return 'Photos from your library';
      case QueryIntent.email:
        return 'Emails from your inbox';
      case QueryIntent.web:
        return 'Search results';
      case QueryIntent.chat:
        return 'Phone data';
    }
  }

  // _stripLeakedContext extracted to text_utils.dart → stripLeakedContext()

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
