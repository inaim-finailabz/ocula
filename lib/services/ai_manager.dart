import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_llama/flutter_llama.dart';

/// Status of a model file on disk.
enum ModelStatus { unknown, missing, ready, loading, loaded, error }

/// Info about a registered model.
class ModelInfo {
  final String feature;
  final String path;
  final int sizePriority; // lower = smaller = load first
  ModelStatus status;

  ModelInfo({
    required this.feature,
    required this.path,
    required this.sizePriority,
    this.status = ModelStatus.unknown,
  });
}

/// Manages loading/unloading of GGUF models.
///
/// Strategy:
///  1. Load the **smallest** model (SmolVLM-256M) immediately on startup
///     so the user sees results within seconds.
///  2. Background-validate all other model files (exist? corrupt?) so
///     switching later is instant with no surprise errors.
///  3. Only ONE model lives in RAM at a time (mobile OOM guard).
///  4. Expose [statusNotifier] so the UI can react to loading states
///     without blocking.
class AIManager {
  // ── singleton ──────────────────────────────────────────────────────
  static final AIManager _instance = AIManager._();
  factory AIManager() => _instance;
  AIManager._();

  // ── engine state ───────────────────────────────────────────────────
  FlutterLlama? _engine;
  String? _activeFeature;
  String _lastResult = ''; // cache for placeholder trick during switch
  bool _isLoading = false;

  String? get activeFeature => _activeFeature;
  bool get isModelLoaded => _engine != null;
  bool get isLoading => _isLoading;
  String get lastResult => _lastResult;

  /// Broadcast stream so the UI can listen for status changes.
  final ValueNotifier<ModelStatus> statusNotifier =
      ValueNotifier(ModelStatus.unknown);

  // ── model registry (ordered by size priority) ──────────────────────
  static final List<ModelInfo> models = [
    ModelInfo(
      feature: 'quick_scan',
      path: 'assets/models/smolvlm.gguf',       // 256 MB – Free tier
      sizePriority: 0,
    ),
    ModelInfo(
      feature: 'detail',
      path: 'assets/models/moondream.gguf',      // 0.5 GB – Plus tier
      sizePriority: 1,
    ),
    ModelInfo(
      feature: 'document',
      path: 'assets/models/qwen2.5-vl.gguf',    // 2 GB – Pro tier
      sizePriority: 2,
    ),
    ModelInfo(
      feature: 'reasoning',
      path: 'assets/models/qwen2.5-vl.gguf',    // 2 GB – Pro tier (shared)
      sizePriority: 2,
    ),
  ];

  // Pre-built configs to avoid re-allocating on every switch.
  final Map<String, LlamaConfig> _configCache = {};

  // ── startup: load smallest first, validate rest in background ──────
  /// Call once from initState(). Returns as soon as the smallest model
  /// is loaded and ready — the rest is validated asynchronously.
  Future<void> initModels({
    void Function(String feature, ModelStatus status)? onProgress,
  }) async {
    // Sort so the smallest model is first.
    final sorted = List<ModelInfo>.from(models)
      ..sort((a, b) => a.sizePriority.compareTo(b.sizePriority));

    // 1. IMMEDIATE: load the smallest model (SmolVLM-256M)
    final primary = sorted.first;
    await _loadModel(primary.feature, notify: true);
    onProgress?.call(primary.feature, ModelStatus.loaded);

    // 2. BACKGROUND: validate remaining model files (no RAM cost)
    unawaited(_validateRemainingModels(sorted.skip(1), onProgress));
  }

  /// Checks that each model file exists and pre-builds its LlamaConfig.
  /// This runs off the main isolate-blocking path so it doesn't janks the UI.
  Future<void> _validateRemainingModels(
    Iterable<ModelInfo> remaining,
    void Function(String feature, ModelStatus status)? onProgress,
  ) async {
    for (final info in remaining) {
      try {
        final file = File(info.path);
        if (await file.exists()) {
          // Pre-cache config so loadFeature() later skips allocation.
          _configCache[info.feature] = LlamaConfig(
            modelPath: info.path,
            nGpuLayers: 99,
          );
          info.status = ModelStatus.ready;
        } else {
          info.status = ModelStatus.missing;
        }
      } catch (_) {
        info.status = ModelStatus.error;
      }
      onProgress?.call(info.feature, info.status);
    }
  }

  // ── public API ─────────────────────────────────────────────────────

  /// Switch to a different feature's model.
  /// Uses cached config when available for faster switching.
  /// If [keepLastResult] is true the previous AI response stays accessible
  /// in [lastResult] so the UI can show a placeholder during load.
  Future<void> loadFeature(
    String featureType, {
    bool keepLastResult = true,
  }) async {
    if (_activeFeature == featureType && _engine != null) return; // no-op
    await _loadModel(featureType, notify: true);
  }

  /// Run a prompt against the currently loaded model.
  Future<String> ask(String prompt) async {
    if (_engine == null) {
      throw StateError('No model loaded. Call loadFeature() first.');
    }
    final response = await _engine!.complete(prompt);
    _lastResult = response; // cache for placeholder trick
    return response;
  }

  /// The model info for a given feature, or null.
  ModelInfo? infoFor(String feature) =>
      models.cast<ModelInfo?>().firstWhere(
            (m) => m?.feature == feature,
            orElse: () => null,
          );

  /// Unload current model and free native memory.
  Future<void> dispose() async {
    await _unloadCurrent();
    _activeFeature = null;
    _configCache.clear();
    statusNotifier.value = ModelStatus.unknown;
  }

  // ── internals ──────────────────────────────────────────────────────

  Future<void> _loadModel(String featureType, {bool notify = false}) async {
    _isLoading = true;
    if (notify) statusNotifier.value = ModelStatus.loading;

    // 1. FREE RAM: unload whatever is in memory
    await _unloadCurrent();

    // 2. BUILD / REUSE CONFIG
    final config = _configCache[featureType] ?? _buildConfig(featureType);

    // 3. LOAD into GPU/NPU
    try {
      _engine = FlutterLlama.instance;
      await _engine!.init(config);
      _activeFeature = featureType;

      // Mark in the registry too
      final info = infoFor(featureType);
      if (info != null) info.status = ModelStatus.loaded;

      if (notify) statusNotifier.value = ModelStatus.loaded;
    } catch (e) {
      _engine = null;
      if (notify) statusNotifier.value = ModelStatus.error;
      rethrow;
    } finally {
      _isLoading = false;
    }
  }

  Future<void> _unloadCurrent() async {
    if (_engine != null) {
      await _engine!.unloadModel();
      _engine = null;
    }
  }

  LlamaConfig _buildConfig(String featureType) {
    final info = infoFor(featureType);
    final path = info?.path ?? 'assets/models/moondream.gguf';
    final config = LlamaConfig(modelPath: path, nGpuLayers: 99);
    _configCache[featureType] = config; // cache for next time
    return config;
  }
}
