import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'ai_manager.dart';
import 'env_config.dart';

/// On-demand model download and lifecycle manager.
///
/// Strategy:
/// 1. App ships with ZERO model files (~30MB install)
/// 2. On first launch, download the free-tier model in background
/// 3. Plus/Pro models download only when user upgrades or needs them
/// 4. Models stored in app documents dir (persists across updates)
/// 5. Downloads are resumable — user can close app mid-download
/// 6. Disk space checked before each download
///
/// Per-OS storage:
/// - iOS:    Documents/models/  (backed up to iCloud)
/// - Android: getExternalFilesDir/models/  (no user permission needed)
/// - macOS:  Application Support/models/
/// - OHOS:   files/models/

class ModelInfo {
  final String fileName;
  final String displayName;
  final String downloadUrl;
  final int sizeBytes;
  final AITier tier;
  final bool isVisionProjector;
  final bool isEmbeddingModel;

  const ModelInfo({
    required this.fileName,
    required this.displayName,
    required this.downloadUrl,
    required this.sizeBytes,
    required this.tier,
    this.isVisionProjector = false,
    this.isEmbeddingModel = false,
  });

  double get sizeMB => sizeBytes / (1024 * 1024);
  String get sizeLabel {
    if (sizeMB >= 1024) return '${(sizeMB / 1024).toStringAsFixed(1)} GB';
    return '${sizeMB.round()} MB';
  }
}

/// Download progress callback.
typedef DownloadProgress = void Function(double progress, String status);

enum ModelStatus { notDownloaded, downloading, ready, error }

class OculaModelManager {
  static final OculaModelManager _instance = OculaModelManager._();
  factory OculaModelManager() => _instance;
  OculaModelManager._();

  /// Default model server base URL.
  /// In dev: reads from .env.dev (e.g. http://192.168.3.14:8080).
  /// In prod: reads from .env.prod (e.g. https://backend-ocula.finailabz.com).
  static const defaultModelServerUrl = EnvConfig.modelServerUrl;

  /// SharedPreferences key for the model server URL.
  static const _prefKeyModelServer = 'model_server_url';

  /// Cached model server URL (loaded from prefs on first use).
  String? _cachedModelServerUrl;

  // ── Model Registry ── 2026 Ocula Intelligence Stack
  //
  // Ocula Lite (free)  – Qwen3-1.7B: text RAG — contacts, email, calendar, docs
  //                      Delivered at install via iOS ODR / Android PAD fast-follow.
  // Ocula Plus (plus)  – Qwen3-VL-2B-Thinking Q4: vision + multimodal queries
  //                      Downloaded on-demand when user first needs image analysis.
  // Ocula Pro  (pro)   – Qwen3-VL-4B-Thinking Q4: higher-quality VL — iPad / ≥6 GB RAM
  //                      Same architecture as Plus, larger param count, fits mid-range devices.
  //                      Downloaded from backend on demand.
  // Embed              – Qwen3-Embedding-0.6B: sentence similarity for RAG (always on)
  static const models = [
    // ── OCULA LITE: Always-on text RAG ────────────────────────────────────
    // iOS: ODR initial-install tags "ocula-lite-p1/p2/p3" (3×<400MB packs)
    // Android: PAD fast-follow (monolithic file in models_pack asset pack)
    // fileName is the first GGUF split part; llama.cpp auto-loads the rest.
    ModelInfo(
      fileName: 'qwen3-1.7b-q4_k_m-00001-of-00003.gguf',
      displayName: 'Ocula Lite',
      downloadUrl:
          'https://backend-ocula.finailabz.com/models/qwen3-1.7b-q4_k_m-00001-of-00003.gguf',
      sizeBytes: 1282 * 1024 * 1024, // ~1.28 GB total (actual: 1,282,439,328 bytes, 3×~450MB parts)
      tier: AITier.free,
    ),
    // ── OCULA EMBED: Multilingual sentence similarity for RAG ─────────────
    // Qwen3-Embedding-0.6B: 1024-dim, multilingual — far better than MiniLM
    // for non-English content. Q8_0 used since embedding quality degrades with Q4.
    ModelInfo(
      fileName: 'Qwen3-Embedding-0.6B-Q8_0.gguf',
      displayName: 'Ocula Embed',
      downloadUrl:
          'https://backend-ocula.finailabz.com/models/Qwen3-Embedding-0.6B-Q8_0.gguf',
      sizeBytes: 639 * 1024 * 1024, // ~639 MB
      tier: AITier.free,
      isEmbeddingModel: true,
    ),
    // ── OCULA PLUS: Vision + multimodal queries ────────────────────────────
    // Downloaded on-demand when user sends a photo or image query.
    ModelInfo(
      fileName: 'Qwen3VL-2B-Thinking-Q4_K_M.gguf',
      displayName: 'Ocula Plus',
      downloadUrl:
          'https://backend-ocula.finailabz.com/models/Qwen3VL-2B-Thinking-Q4_K_M.gguf',
      sizeBytes: 1110 * 1024 * 1024, // ~1.1 GB
      tier: AITier.plus,
    ),
    ModelInfo(
      fileName: 'mmproj-Qwen3VL-2B-Thinking-F16.gguf',
      displayName: 'Ocula Plus Vision',
      downloadUrl:
          'https://backend-ocula.finailabz.com/models/mmproj-Qwen3VL-2B-Thinking-F16.gguf',
      sizeBytes: 819 * 1024 * 1024, // ~819 MB
      tier: AITier.plus,
      isVisionProjector: true,
    ),
    // ── OCULA PRO: Higher-quality VL — iPad / mid-high RAM devices (≥6 GB) ──
    // Qwen3-VL-4B Q4: ~2.5 GB weights + ~836 MB mmproj ≈ 4–5 GB working set.
    // Replaces Qwen2.5-VL-7B (was ~8 GB working set, iPad Pro only).
    // Downloaded from backend on demand.
    ModelInfo(
      fileName: 'Qwen3VL-4B-Thinking-Q4_K_M.gguf',
      displayName: 'Ocula Pro',
      downloadUrl:
          'https://backend-ocula.finailabz.com/models/Qwen3VL-4B-Thinking-Q4_K_M.gguf',
      sizeBytes: 2500 * 1024 * 1024, // ~2.5 GB (HF: 2,500,000,000 bytes)
      tier: AITier.pro,
    ),
    ModelInfo(
      fileName: 'mmproj-Qwen3VL-4B-Thinking-F16.gguf',
      displayName: 'Ocula Pro Vision',
      downloadUrl:
          'https://backend-ocula.finailabz.com/models/mmproj-Qwen3VL-4B-Thinking-F16.gguf',
      sizeBytes: 836 * 1024 * 1024, // ~836 MB (HF: 836,000,000 bytes)
      tier: AITier.pro,
      isVisionProjector: true,
    ),
  ];

  // Bump this whenever free model shards change — invalidates old manifest
  // and forces a clean re-copy on next launch.
  static const int _freeModelDirVersion = 32;

  String? _modelsDir;
  HttpClient? _httpClient;
  List<ModelInfo> _enterpriseModels = [];
  final Map<String, double> _downloadProgress = {};

  /// Filenames for which the user has requested cancellation.
  final Set<String> _cancelledDownloads = {};
  final StreamController<Map<String, double>>
  _downloadProgressStreamController = StreamController.broadcast();

  /// Emits user-friendly feature names when a tier's models are fully downloaded.
  /// Values: 'Quick Scan', 'Detail & Counting', 'Vision & Deep Analysis'
  final StreamController<String> _featureReadyController =
      StreamController.broadcast();

  /// Emits the AITier enum whenever that tier's models are fully downloaded.
  /// AIManager subscribes to this to set a pending-upgrade flag.
  final StreamController<AITier> _tierReadyController =
      StreamController.broadcast();

  /// Emits true when the free model is ready, false when install failed.
  /// The home screen subscribes to this to auto-load the model and show retry.
  final StreamController<bool> _freeModelStatusController =
      StreamController.broadcast();

  Stream<Map<String, double>> get downloadProgressStream =>
      _downloadProgressStreamController.stream;

  /// Cancel an in-progress download for [fileName].
  ///
  /// Sets a cancellation flag that the download loop checks after each chunk.
  /// The partial file is deleted and the progress entry is cleared.
  void cancelDownload(String fileName) {
    _cancelledDownloads.add(fileName);
    debugPrint('[ModelManager] cancelDownload requested for $fileName');
  }

  /// Subscribe to this for user-facing "feature X is ready" notifications.
  Stream<String> get featureReadyStream => _featureReadyController.stream;

  /// Subscribe to this for programmatic "tier X is ready" signals.
  /// AIManager listens here to set the pending-upgrade flag.
  Stream<AITier> get tierReadyStream => _tierReadyController.stream;

  /// Emits true when free model install succeeds, false when it fails.
  /// Home screen subscribes to auto-load the model and offer retry.
  Stream<bool> get freeModelStatusStream => _freeModelStatusController.stream;

  /// Human-friendly feature label for each tier (never expose model names).
  static String featureLabel(AITier tier) {
    switch (tier) {
      case AITier.free:
        return 'Ocula Lite';
      case AITier.plus:
        return 'Ocula Plus';
      case AITier.pro:
        return 'Ocula Pro';
      case AITier.enterprise:
        return 'Enterprise';
    }
  }

  /// Delete legacy model files no longer in the active registry.
  /// SmolVLM2 was replaced by Qwen3-VL in build 19.
  /// Moondream2 (old Plus tier) replaced by Qwen3-VL-2B (Plus) in build 20.
  /// all-MiniLM-L6-v2 replaced by Qwen3-Embedding-0.6B (build 21).
  /// Qwen3-VL files are NOT deleted — they are now the Plus/Vision tier.
  Future<void> _cleanupLegacyModels() async {
    final dir = await modelsDir;
    const legacy = [
      'SmolVLM2-500M-Video-Instruct-finetuned-Q8_0.gguf',
      'mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf',
      // Moondream2 was Plus tier in builds 17-19; replaced by Qwen3-VL in build 20
      'moondream2-text-model-Q4_K_M.gguf',
      'moondream2-mmproj-f16-20250414.gguf',
      // all-MiniLM replaced by Qwen3-Embedding-0.6B in build 21 (multilingual)
      'all-MiniLM-L6-v2.Q8_0.gguf',
      // Monolithic qwen2.5-1.5b replaced by 3-part split GGUF in build 26
      // (Apple ODR limit: 512 MB per asset pack; split allows all parts as initial-install tags)
      'qwen2.5-1.5b-instruct-q4_k_m.gguf',
      // qwen2.5-1.5b split parts replaced by Qwen3-1.7B in build 34
      'qwen2.5-1.5b-instruct-q4_k_m-00001-of-00003.gguf',
      'qwen2.5-1.5b-instruct-q4_k_m-00002-of-00003.gguf',
      'qwen2.5-1.5b-instruct-q4_k_m-00003-of-00003.gguf',
      // Qwen2.5-VL-7B replaced by Qwen3-VL-4B as Pro tier (build 32)
      // 7B was ~8 GB working set (iPad Pro only); 4B is ~4 GB (iPad / ≥6 GB RAM)
      'Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf',
      'mmproj-Qwen2.5-VL-7B-Instruct-F16.gguf',
    ];
    for (final name in legacy) {
      final f = File('$dir/$name');
      if (await f.exists()) {
        await f.delete();
        debugPrint('[ModelManager] Deleted legacy model: $name');
      }
    }
  }

  /// Install the free-tier main model and signal readiness.
  ///
  /// Install priority:
  ///   1. iOS ODR / Android PAD — zero-cost, already on device after install
  ///   2. Backend download — monolithic GGUF from the model server (fallback)
  ///
  /// This method can be called without await (fire-and-forget from the splash
  /// or from a retry button in the home screen). Progress is always emitted on
  /// [downloadProgressStream] so the home-screen banner reflects it even when
  /// no caller-supplied [onProgress] is provided.
  ///
  /// On success emits true on [freeModelStatusStream] and fires [tierReadyStream]
  /// so the home screen can auto-load the model.
  /// On failure emits false on [freeModelStatusStream] so the home screen can
  /// show a retry button.
  Future<bool> ensureFreeModelReady({DownloadProgress? onProgress}) async {
    _cleanupLegacyModels().catchError((_) {});
    _purgeOldFreeShardManifests().catchError((_) {});

    final mainModel = models
        .where(
          (m) =>
              m.tier == AITier.free &&
              !m.isVisionProjector &&
              !m.isEmbeddingModel,
        )
        .first;

    // Wrap the caller's callback to also publish progress to the download stream
    // so the home-screen banner shows free-model install progress even when we
    // are not blocking at the splash screen.
    DownloadProgress combined = (progress, status) {
      if (progress > 0) {
        _downloadProgress[mainModel.fileName] = progress;
        _downloadProgressStreamController.add(Map.from(_downloadProgress));
      }
      onProgress?.call(progress, status);
    };

    // Step 1: iOS ODR / Android PAD (instant on App Store builds).
    combined(0.05, 'Checking for bundled AI engine...');
    if (await _ensureBundledModelCopied(mainModel.fileName, onProgress: combined)) {
      combined(1.0, 'Bundled AI engine ready!');
    } else if (await isDownloaded(mainModel.fileName)) {
      // Step 2: Already downloaded from a previous run.
      combined(1.0, 'AI engine ready!');
    } else {
      // Step 3: Fallback — download the monolithic GGUF from the backend.
      // This is the priority fallback when ODR/PAD is unavailable (TestFlight,
      // direct installs, developer builds).
      combined(0.0, 'Downloading AI engine...');
      final ok = await download(mainModel, onProgress: combined);
      if (!ok) {
        _downloadProgress.remove(mainModel.fileName);
        _downloadProgressStreamController.add(Map.from(_downloadProgress));
        _freeModelStatusController.add(false);
        return false;
      }
    }

    // Clear the progress entry — banner disappears.
    _downloadProgress.remove(mainModel.fileName);
    _downloadProgressStreamController.add(Map.from(_downloadProgress));

    // Signal that the free model is ready.  Home screen picks this up to
    // auto-load the model and dismiss any "not ready" state.
    _freeModelStatusController.add(true);
    _featureReadyController.add(featureLabel(AITier.free));
    _tierReadyController.add(AITier.free);

    // Projector and embedding — fire-and-forget (not required for text RAG).
    final projector = models
        .where((m) => m.tier == AITier.free && m.isVisionProjector)
        .firstOrNull;
    if (projector != null && !await isDownloaded(projector.fileName)) {
      _ensureBundledModelCopied(projector.fileName)
          .then((copied) {
            if (!copied) download(projector).catchError((_) => false);
          })
          .catchError((_) {});
    }

    final embedModel = models.where((m) => m.isEmbeddingModel).firstOrNull;
    if (embedModel != null && !await isDownloaded(embedModel.fileName)) {
      _ensureBundledModelCopied(embedModel.fileName)
          .then((copied) {
            if (!copied) {
              download(embedModel)
                  .then((ok) {
                    if (ok) debugPrint('[ModelManager] Embedding model downloaded');
                  })
                  .catchError((e) {
                    debugPrint('[ModelManager] Embedding model download failed: $e');
                  });
            } else {
              debugPrint('[ModelManager] Embedding model copied from bundle');
            }
          })
          .catchError((_) {});
    }

    return true;
  }

  /// Download remaining models in the background.
  /// Includes: free-tier projector (not required for splash) + all plus/pro models.
  /// Emits feature-ready notifications as each tier completes.
  /// Each download has a 10-minute timeout so one failure doesn't block the rest.
  Future<void> downloadRemainingInBackground() async {
    debugPrint(
      '[ModelManager] Starting background download of remaining models...',
    );

    // First, ensure the free-tier projector is copied/downloaded
    final freeProjector = models
        .where((m) => m.tier == AITier.free && m.isVisionProjector)
        .firstOrNull;
    if (freeProjector != null && !await isDownloaded(freeProjector.fileName)) {
      // Try bundled copy first, then download
      final copied = await _ensureBundledModelCopied(freeProjector.fileName);
      if (!copied) {
        try {
          await download(
            freeProjector,
            onProgress: (progress, status) {
              _downloadProgress[freeProjector.fileName] = progress;
              _downloadProgressStreamController.add(_downloadProgress);
            },
          ).timeout(const Duration(minutes: 10));
          debugPrint('[ModelManager] ✓ Free projector downloaded');
        } catch (e) {
          debugPrint('[ModelManager] ✗ Free projector download failed: $e');
        }
      }
    }

    // Then copy/download plus/pro tier models
    for (final tier in [AITier.plus, AITier.pro]) {
      final tierModels = modelsForTier(tier);
      bool allReady = true;

      for (final model in tierModels) {
        if (await isDownloaded(model.fileName)) {
          debugPrint('[ModelManager] ${model.fileName} already downloaded');
          continue;
        }
        // Try bundled copy first (instant), then download as fallback
        final copied = await _ensureBundledModelCopied(model.fileName);
        if (copied) {
          debugPrint('[ModelManager] ✓ ${model.fileName} copied from bundle');
          continue;
        }
        try {
          debugPrint(
            '[ModelManager] Downloading ${model.fileName} (${model.sizeLabel})...',
          );
          final ok = await download(
            model,
            onProgress: (progress, status) {
              _downloadProgress[model.fileName] = progress;
              _downloadProgressStreamController.add(_downloadProgress);
            },
          ).timeout(const Duration(minutes: 10));
          if (!ok) {
            allReady = false;
            debugPrint(
              '[ModelManager] ✗ ${model.fileName} download returned false',
            );
          } else {
            debugPrint('[ModelManager] ✓ ${model.fileName} downloaded');
          }
        } catch (e) {
          allReady = false;
          debugPrint('[ModelManager] ✗ ${model.fileName} download failed: $e');
        }
      }

      if (allReady) {
        final label = featureLabel(tier);
        debugPrint('[ModelManager] ★ Tier ${tier.name} ready: $label');
        _featureReadyController.add(label);
        _tierReadyController.add(tier);
      }
    }
    // Guarantee the embedding model is available across all tiers.
    // It may have been skipped during splash (e.g. network error or bundled
    // copy not present). Re-check and download if missing.
    final embedModel = models.where((m) => m.isEmbeddingModel).firstOrNull;
    if (embedModel != null && !await isDownloaded(embedModel.fileName)) {
      final copied = await _ensureBundledModelCopied(embedModel.fileName);
      if (!copied) {
        try {
          debugPrint('[ModelManager] Downloading embedding model (missed during splash)...');
          await download(
            embedModel,
            onProgress: (progress, status) {
              _downloadProgress[embedModel.fileName] = progress;
              _downloadProgressStreamController.add(_downloadProgress);
            },
          ).timeout(const Duration(minutes: 5));
          debugPrint('[ModelManager] ✓ Embedding model downloaded');
        } catch (e) {
          debugPrint('[ModelManager] ✗ Embedding model download failed: $e');
        }
      }
    }

    debugPrint('[ModelManager] Background download pass complete');
  }

  Future<void> downloadAllModelsInBackground() async {
    final tiers = [AITier.free, AITier.plus, AITier.pro];
    for (final tier in tiers) {
      final tierModels = modelsForTier(tier);
      for (final model in tierModels) {
        if (!await isDownloaded(model.fileName)) {
          download(
            model,
            onProgress: (progress, status) {
              _downloadProgress[model.fileName] = progress;
              _downloadProgressStreamController.add(_downloadProgress);
            },
          );
        }
      }
    }
  }

  /// Get the models directory, creating it if needed.
  Future<String> get modelsDir async {
    if (_modelsDir != null) return _modelsDir!;
    final appDir = await getApplicationSupportDirectory();
    final dir = Directory('${appDir.path}/models');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    _modelsDir = dir.path;
    return _modelsDir!;
  }

  /// Full path for a model file on this device.
  Future<String> modelPath(String fileName) async {
    final dir = await modelsDir;
    return '$dir/$fileName';
  }

  /// Check if a model file exists in local storage and is valid.
  /// Does NOT check the app bundle — models must be copied to local
  /// storage before use because llama.cpp needs mmap-able files and
  /// the iOS app bundle is code-signed / read-only.
  ///
  /// For split GGUF models, ALL parts must be present and ≥1 MB.
  /// llama.cpp auto-loads sibling parts from the same directory —
  /// if any part is missing it will fail to load, so we treat
  /// a partial set as "not downloaded".
  Future<bool> isDownloaded(String fileName) async {
    final parts = _allSplitParts(fileName);
    for (final part in parts) {
      final path = await modelPath(part);
      final file = File(path);
      if (!file.existsSync()) return false;
      final size = await file.length();
      if (size < 1024 * 1024) return false;
    }
    return parts.isNotEmpty;
  }

  // ── iOS On-Demand Resources (ODR) ──────────────────────────────
  static const _odrChannel         = MethodChannel('com.finailabz.ocula/odr');
  static const _odrProgressChannel = EventChannel('com.finailabz.ocula/odr_progress');

  /// Map a model filename to its iOS ODR tag name.
  /// Split GGUF parts each get a numbered tag (ocula-lite-p1, p2, p3…).
  static String _odrTagForFile(String fileName) {
    if (fileName.contains('qwen3-1.7b') || fileName.contains('qwen2.5-1.5b')) {
      final m = RegExp(r'-0*(\d+)-of-\d+').firstMatch(fileName);
      if (m != null) return 'ocula-lite-p${m.group(1)}';
      return 'ocula-lite-p1';
    }
    if (fileName.startsWith('all-MiniLM')) return 'ocula-embed';
    if (fileName.startsWith('Qwen3VL') || fileName.startsWith('mmproj-Qwen3VL')) {
      return 'ocula-vision';
    }
    return 'ocula-model';
  }

  /// For a split-GGUF first-part filename, return all part filenames in order.
  /// Returns [fileName] unchanged for non-split models.
  /// Example: "model-00001-of-00003.gguf" → ["model-00001-of-00003.gguf",
  ///           "model-00002-of-00003.gguf", "model-00003-of-00003.gguf"]
  static List<String> _allSplitParts(String fileName) {
    final match =
        RegExp(r'^(.+-)\d{5}-of-(\d{5})\.gguf$').firstMatch(fileName);
    if (match == null) return [fileName];
    final base = match.group(1)!;
    final total = int.parse(match.group(2)!);
    return List.generate(
      total,
      (i) =>
          '$base${(i + 1).toString().padLeft(5, '0')}-of-${total.toString().padLeft(5, '0')}.gguf',
    );
  }

  /// On iOS with App Store / TestFlight install, request the ODR tag and return
  /// the file path. Returns null if ODR is not configured — caller falls through
  /// to network download transparently.
  ///
  /// For split-GGUF models the single-tag path is still used here (individual
  /// part request). The multi-part batch path is in [_ensureSplitModelCopied].
  Future<String?> _findODRPath(String fileName, {DownloadProgress? onProgress}) async {
    if (!Platform.isIOS) return null;
    try {
      final tag = _odrTagForFile(fileName);

      // Subscribe to KVO progress events BEFORE triggering beginAccessingResources.
      StreamSubscription<dynamic>? progressSub;
      if (onProgress != null) {
        progressSub = _odrProgressChannel.receiveBroadcastStream().listen((event) {
          if (event is Map && event['tag'] == tag) {
            final pct = ((event['progress'] as num?)?.toDouble() ?? 0.0).clamp(0.0, 1.0);
            // Keep progress in the 0.05–0.95 band so the caller's surrounding
            // bookkeeping values (0.1 before, 1.0 after) are never overwritten.
            final mapped = 0.05 + pct * 0.90;
            onProgress(mapped, 'Downloading via iOS delivery... ${(pct * 100).toStringAsFixed(0)}%');
          }
        });
      }

      try {
        final path = await _odrChannel.invokeMethod<String>(
          'requestODRTag',
          {'tag': tag, 'fileName': fileName},
        );
        if (path != null) {
          debugPrint('[ModelManager] ODR path for $fileName: $path');
        }
        return path;
      } finally {
        await progressSub?.cancel();
      }
    } catch (e) {
      debugPrint('[ModelManager] ODR not available for $fileName: $e');
      return null;
    }
  }

  /// Request all split-GGUF ODR tags in one coordinated download session.
  /// Returns a map of {fileName → bundlePath} for each resolved part,
  /// or null if ODR is unavailable (dev / TestFlight without App Store processing).
  Future<Map<String, String>?> _findODRPathBatch(
    List<String> parts, {
    DownloadProgress? onProgress,
  }) async {
    if (!Platform.isIOS) return null;
    try {
      final tags = parts.map(_odrTagForFile).toList();
      final firstTag = tags.first;

      StreamSubscription<dynamic>? progressSub;
      if (onProgress != null) {
        progressSub = _odrProgressChannel.receiveBroadcastStream().listen((event) {
          if (event is Map && event['tag'] == firstTag) {
            final pct = ((event['progress'] as num?)?.toDouble() ?? 0.0).clamp(0.0, 1.0);
            final mapped = 0.05 + pct * 0.90;
            onProgress(
              mapped,
              'Downloading AI model via iOS delivery... ${(pct * 100).toStringAsFixed(0)}%',
            );
          }
        });
      }

      try {
        final raw = await _odrChannel.invokeMethod<Map<Object?, Object?>>(
          'requestODRTagsBatch',
          {'tags': tags, 'fileNames': parts},
        );
        if (raw == null) return null;
        final result = <String, String>{};
        for (final entry in raw.entries) {
          if (entry.key is String && entry.value is String) {
            result[entry.key as String] = entry.value as String;
          }
        }
        debugPrint('[ModelManager] ODR batch resolved: ${result.keys.join(', ')}');
        return result.isEmpty ? null : result;
      } finally {
        await progressSub?.cancel();
      }
    } catch (e) {
      debugPrint('[ModelManager] ODR batch not available: $e');
      return null;
    }
  }

  // ── Shard manifest helpers ─────────────────────────────────────────────────
  // The manifest records the exact byte count for each successfully copied shard.
  // Bumping _freeModelDirVersion changes the manifest filename, which forces a
  // clean re-copy — old corrupt files in the same path cannot interfere.

  Future<File> get _freeShardManifestFile async {
    final dir = await modelsDir;
    return File('$dir/free_shard_manifest_v$_freeModelDirVersion.json');
  }

  Future<Map<String, dynamic>?> _readFreeShardManifest() async {
    try {
      final f = await _freeShardManifestFile;
      if (!f.existsSync()) return null;
      return jsonDecode(await f.readAsString()) as Map<String, dynamic>;
    } catch (_) {
      return null;
    }
  }

  Future<void> _writeFreeShardManifest(
      List<Map<String, dynamic>> shards) async {
    final f = await _freeShardManifestFile;
    await f.writeAsString(
        jsonEncode({'version': _freeModelDirVersion, 'shards': shards}));
    debugPrint(
        '[ModelManager] Wrote shard manifest v$_freeModelDirVersion (${shards.length} shards)');
  }

  /// Delete manifests from previous versions so stale files don't linger
  /// on disk between app launches.
  Future<void> _purgeOldFreeShardManifests() async {
    try {
      final dir = await modelsDir;
      await for (final entity in Directory(dir).list()) {
        if (entity is File) {
          final name = p.basename(entity.path);
          if (name.startsWith('free_shard_manifest_v') &&
              name !=
                  'free_shard_manifest_v$_freeModelDirVersion.json') {
            await entity.delete().catchError((_) {});
            debugPrint('[ModelManager] Purged old shard manifest: $name');
          }
        }
      }
    } catch (e) {
      debugPrint('[ModelManager] _purgeOldFreeShardManifests error: $e');
    }
  }

  // ── Low-level file utilities ───────────────────────────────────────────────

  /// Synchronous GGUF magic-header check. Reads only 4 bytes — zero RAM cost.
  bool _hasGGUFHeaderSync(String path) {
    try {
      final raf = File(path).openSync(mode: FileMode.read);
      try {
        final bytes = raf.readSync(4);
        return bytes.length == 4 && String.fromCharCodes(bytes) == 'GGUF';
      } finally {
        raf.closeSync();
      }
    } catch (_) {
      return false;
    }
  }

  /// Extended GGUF structural check for split part 1.
  ///
  /// Reads the first 24 bytes and validates:
  ///   bytes  0- 3: "GGUF" magic
  ///   bytes  4- 7: version == 2 or 3 (uint32 LE)
  ///   bytes  8-15: tensor_count > 0  (uint64 LE)
  ///   bytes 16-23: metadata_kv_count > 0 (uint64 LE)
  ///
  /// A truncated or all-zero file after the first 4 bytes would fail on
  /// the version check. Returns false on any I/O error.
  bool _isGGUFPart1Valid(String path) {
    try {
      final raf = File(path).openSync(mode: FileMode.read);
      try {
        final b = raf.readSync(24);
        if (b.length < 24) return false;
        // Magic
        if (String.fromCharCodes(b.sublist(0, 4)) != 'GGUF') return false;
        // Version (uint32 LE) — must be 2 or 3
        final version = b[4] | (b[5] << 8) | (b[6] << 16) | (b[7] << 24);
        if (version < 2 || version > 3) return false;
        // tensor_count (uint64 LE) — must be > 0
        int tensorCount = 0;
        for (int i = 0; i < 8; i++) tensorCount |= b[8 + i] << (i * 8);
        if (tensorCount <= 0) return false;
        // metadata_kv_count (uint64 LE) — must be > 0
        int metaCount = 0;
        for (int i = 0; i < 8; i++) metaCount |= b[16 + i] << (i * 8);
        if (metaCount <= 0) return false;
        return true;
      } finally {
        raf.closeSync();
      }
    } catch (_) {
      return false;
    }
  }

  /// Atomic file copy: src → dest.tmp → dest (rename).
  ///
  /// Verifies that the written byte count equals the source size before
  /// renaming. If anything goes wrong the .tmp is deleted — the final
  /// [destPath] is never left in a partial state.
  /// Delete all shard dest files and their .tmp counterparts so the next
  /// retry starts with a completely clean slate.
  Future<void> _wipeSplitShards(List<String> parts) async {
    for (final part in parts) {
      final dest = await modelPath(part);
      for (final path in [dest, '$dest.tmp']) {
        final f = File(path);
        if (f.existsSync()) {
          await f.delete().catchError((_) {});
          debugPrint('[ModelManager] wipe_shard: $path');
        }
      }
    }
  }

  Future<void> _atomicFileCopy(
    File src,
    String destPath, {
    String label = '',
    void Function(double progress)? onProgress,
  }) async {
    final tmpPath = '$destPath.tmp';
    final tmpFile = File(tmpPath);

    // Remove any stale .tmp from a previous interrupted copy
    if (tmpFile.existsSync()) {
      debugPrint('[ModelManager] Removing stale tmp: $tmpPath');
      await tmpFile.delete();
    }

    try {
      final srcSize = await src.length();
      debugPrint(
        '[ModelManager] copy_start: $label'
        ' src=${src.path} expected=$srcSize bytes',
      );

      // For large files (>10 MB) use a chunked stream so the caller can
      // report progress during the copy instead of freezing at one percent.
      if (onProgress != null && srcSize > 10 * 1024 * 1024) {
        final sink = tmpFile.openWrite();
        int written = 0;
        try {
          await for (final chunk in src.openRead()) {
            sink.add(chunk);
            written += chunk.length;
            onProgress((written / srcSize).clamp(0.0, 1.0));
          }
          await sink.flush();
        } finally {
          await sink.close();
        }
      } else {
        await src.copy(tmpPath);
      }

      final tmpSize = await tmpFile.length();
      if (tmpSize != srcSize) {
        throw Exception(
            'copy_truncated: expected $srcSize bytes, wrote $tmpSize bytes');
      }
      // POSIX rename is atomic when src and dst are on the same filesystem
      await tmpFile.rename(destPath);
      debugPrint(
        '[ModelManager] copy_done: $label rename_ok $tmpSize bytes',
      );
    } catch (e) {
      // Clean up tmp — never leave a partial final file
      if (tmpFile.existsSync()) {
        await tmpFile.delete().catchError((_) {});
      }
      rethrow;
    }
  }

  /// Copy all parts of a split GGUF model to local storage.
  /// On iOS: batch-requests all ODR tags (one download session, shared progress).
  /// On Android: reads each part from Play Asset Delivery.
  /// Returns true only when every part is present in local storage.
  Future<bool> _ensureSplitModelCopied(
    List<String> parts, {
    DownloadProgress? onProgress,
    int? expectedTotalBytes,
  }) async {
    // ── Step 1: Check manifest first ────────────────────────────────────────
    // The manifest records the exact byte count for every shard that was
    // atomically copied and verified. If it exists and all sizes match we can
    // skip the copy entirely — no heuristics, no relative-size guessing.
    final manifest = await _readFreeShardManifest();
    if (manifest != null) {
      try {
        final manifestShards =
            (manifest['shards'] as List).cast<Map<String, dynamic>>();
        bool allValid = manifestShards.length == parts.length;
        for (int si = 0; si < manifestShards.length; si++) {
          final shard = manifestShards[si];
          final name = shard['name'] as String;
          final expectedBytes = shard['bytes'] as int;
          final path = await modelPath(name);
          final f = File(path);
          final exists = f.existsSync();
          final actualSize = exists ? await f.length() : 0;
          final headerOk = exists && _hasGGUFHeaderSync(path);
          // For part 1 (first shard), run the extended structural check so that
          // corrupt ODR assets written by older builds are caught here too.
          final structuralOk = (si == 0 && exists)
              ? _isGGUFPart1Valid(path)
              : headerOk;
          debugPrint(
            '[ModelManager] manifest_check: $name '
            'exists=$exists size=$actualSize expected=$expectedBytes '
            'header=$headerOk structural=$structuralOk',
          );
          if (!exists || actualSize != expectedBytes || !headerOk || !structuralOk) {
            allValid = false;
            break;
          }
        }
        if (allValid) {
          debugPrint(
              '[ModelManager] All shards verified via manifest — skipping copy');
          return true;
        }
        // Manifest present but validation failed — delete it and force re-copy
        debugPrint('[ModelManager] Manifest validation failed — forcing re-copy');
        final mf = await _freeShardManifestFile;
        await mf.delete().catchError((_) {});
      } catch (e) {
        debugPrint('[ModelManager] Manifest parse error: $e — forcing re-copy');
      }
    }

    // ── Step 2: Copy shards (iOS ODR) ───────────────────────────────────────
    if (Platform.isIOS) {
      final pathMap = await _findODRPathBatch(parts, onProgress: onProgress);
      if (pathMap != null) {
        final copiedShards = <Map<String, dynamic>>[];
        for (int i = 0; i < parts.length; i++) {
          final part = parts[i];
          final src = pathMap[part];
          if (src == null) {
            debugPrint('[ModelManager] copy_failed: ODR pathMap missing $part');
            await _wipeSplitShards(parts);
            return false;
          }
          final srcFile = File(src);
          final srcSize = await srcFile.length();
          final dest = await modelPath(part);
          await Directory(File(dest).parent.path).create(recursive: true);

          // Force overwrite if dest exists but size or header is wrong
          final destFile = File(dest);
          if (destFile.existsSync()) {
            final destSize = await destFile.length();
            final headerOk = _hasGGUFHeaderSync(dest);
            if (destSize == srcSize && headerOk) {
              debugPrint(
                  '[ModelManager] Shard $part already correct ($destSize bytes), skipping');
              copiedShards.add({'name': part, 'bytes': destSize});
              onProgress?.call(
                0.95 + ((i + 1) / parts.length) * 0.05,
                'Verified AI model part ${i + 1} of ${parts.length}...',
              );
              continue;
            }
            debugPrint(
              '[ModelManager] Shard $part mismatch '
              '(size=$destSize expected=$srcSize header=$headerOk) — deleting',
            );
            await destFile.delete().catchError((_) {});
          }

          // Up to 3 attempts per shard before failing the whole copy
          bool shardOk = false;
          for (int attempt = 0; attempt < 3 && !shardOk; attempt++) {
            if (attempt > 0) {
              debugPrint(
                  '[ModelManager] copy_retry: $part attempt ${attempt + 1}');
              await File('$dest.tmp').delete().catchError((_) {});
              await Future<void>.delayed(const Duration(milliseconds: 500));
            }
            onProgress?.call(
              0.95 + (i / parts.length) * 0.05,
              'Installing AI model part ${i + 1} of ${parts.length}...',
            );
            try {
              await _atomicFileCopy(
                srcFile,
                dest,
                label: 'part ${i + 1}/${parts.length}: $part',
                onProgress: (p) => onProgress?.call(
                  0.95 + ((i + p) / parts.length) * 0.05,
                  'Installing AI model part ${i + 1} of ${parts.length}...',
                ),
              );
              final finalSize = await File(dest).length();
              if (!_hasGGUFHeaderSync(dest)) {
                debugPrint(
                    '[ModelManager] copy_failed: $part invalid GGUF header after copy');
                await File(dest).delete().catchError((_) {});
                continue;
              }
              copiedShards.add({'name': part, 'bytes': finalSize});
              debugPrint(
                  '[ModelManager] ✓ Shard installed: $part ($finalSize bytes)');
              shardOk = true;
            } catch (e) {
              debugPrint(
                  '[ModelManager] copy_error attempt ${attempt + 1}: $part → $e');
            }
          }

          if (!shardOk) {
            debugPrint('[ModelManager] copy_failed after 3 attempts: $part — wiping all shards');
            await _wipeSplitShards(parts);
            return false;
          }
        }
        // ── Deep validation before committing the manifest ──────────────────
        // A 4-byte GGUF magic check is not enough — corrupt ODR assets can
        // pass the header check but have garbage tensor data.  Validate:
        //   1. Part 1 structural integrity (version + tensor_count + meta_count)
        //   2. Total shard size ≥ 90% of the declared model size
        final part1DestPath = await modelPath(parts.first);
        if (!_isGGUFPart1Valid(part1DestPath)) {
          debugPrint(
            '[ModelManager] ODR part-1 failed structural GGUF check — '
            'corrupt ODR asset, falling back to backend download',
          );
          await _wipeSplitShards(parts);
          return false;
        }
        if (expectedTotalBytes != null && expectedTotalBytes > 0) {
          final totalCopied =
              copiedShards.fold<int>(0, (s, e) => s + (e['bytes'] as int));
          final minExpected = (expectedTotalBytes * 0.90).toInt();
          if (totalCopied < minExpected) {
            debugPrint(
              '[ModelManager] ODR total shard size $totalCopied < 90% of '
              'expected $expectedTotalBytes — falling back to backend download',
            );
            await _wipeSplitShards(parts);
            return false;
          }
        }
        // All shards verified — commit the manifest
        await _writeFreeShardManifest(copiedShards);
        return true;
      }
      return false;
    }

    // ── Step 2: Copy shards (Android PAD) ───────────────────────────────────
    if (Platform.isAndroid) {
      final copiedShards = <Map<String, dynamic>>[];
      for (int i = 0; i < parts.length; i++) {
        final part = parts[i];
        final src = await _findAssetPackPath(part);
        if (src == null) {
          debugPrint('[ModelManager] copy_failed: PAD path missing for $part');
          await _wipeSplitShards(parts);
          return false;
        }
        final srcFile = File(src);
        final srcSize = await srcFile.length();
        final dest = await modelPath(part);
        await Directory(File(dest).parent.path).create(recursive: true);

        // Force overwrite if dest exists but size or header is wrong
        final destFile = File(dest);
        if (destFile.existsSync()) {
          final destSize = await destFile.length();
          final headerOk = _hasGGUFHeaderSync(dest);
          if (destSize == srcSize && headerOk) {
            debugPrint(
                '[ModelManager] Shard $part already correct ($destSize bytes), skipping');
            copiedShards.add({'name': part, 'bytes': destSize});
            onProgress?.call(
              0.95 + ((i + 1) / parts.length) * 0.05,
              'Verified AI model part ${i + 1} of ${parts.length}...',
            );
            continue;
          }
          debugPrint(
            '[ModelManager] Shard $part mismatch '
            '(size=$destSize expected=$srcSize header=$headerOk) — deleting',
          );
          await destFile.delete().catchError((_) {});
        }

        // Up to 3 attempts per shard before failing the whole copy
        bool shardOk = false;
        for (int attempt = 0; attempt < 3 && !shardOk; attempt++) {
          if (attempt > 0) {
            debugPrint(
                '[ModelManager] copy_retry: $part attempt ${attempt + 1}');
            await File('$dest.tmp').delete().catchError((_) {});
            await Future<void>.delayed(const Duration(milliseconds: 500));
          }
          onProgress?.call(
            0.95 + (i / parts.length) * 0.05,
            'Installing AI model part ${i + 1} of ${parts.length}...',
          );
          try {
            await _atomicFileCopy(
              srcFile,
              dest,
              label: 'part ${i + 1}/${parts.length}: $part',
              onProgress: (p) => onProgress?.call(
                0.95 + ((i + p) / parts.length) * 0.05,
                'Installing AI model part ${i + 1} of ${parts.length}...',
              ),
            );
            final finalSize = await File(dest).length();
            if (!_hasGGUFHeaderSync(dest)) {
              debugPrint(
                  '[ModelManager] copy_failed: $part invalid GGUF header after copy');
              await File(dest).delete().catchError((_) {});
              continue;
            }
            copiedShards.add({'name': part, 'bytes': finalSize});
            debugPrint(
                '[ModelManager] ✓ Shard installed: $part ($finalSize bytes)');
            shardOk = true;
          } catch (e) {
            debugPrint(
                '[ModelManager] copy_error attempt ${attempt + 1}: $part → $e');
          }
        }

        if (!shardOk) {
          debugPrint('[ModelManager] copy_failed after 3 attempts: $part — wiping all shards');
          await _wipeSplitShards(parts);
          return false;
        }
      }
      // Deep validation — same checks as the iOS ODR path above.
      final part1DestPath = await modelPath(parts.first);
      if (!_isGGUFPart1Valid(part1DestPath)) {
        debugPrint(
          '[ModelManager] PAD part-1 failed structural GGUF check — wiping',
        );
        await _wipeSplitShards(parts);
        return false;
      }
      if (expectedTotalBytes != null && expectedTotalBytes > 0) {
        final totalCopied =
            copiedShards.fold<int>(0, (s, e) => s + (e['bytes'] as int));
        final minExpected = (expectedTotalBytes * 0.90).toInt();
        if (totalCopied < minExpected) {
          debugPrint(
            '[ModelManager] PAD total shard size $totalCopied < 90% of '
            'expected $expectedTotalBytes — wiping',
          );
          await _wipeSplitShards(parts);
          return false;
        }
      }
      await _writeFreeShardManifest(copiedShards);
      return true;
    }

    return false;
  }

  // ── Play Asset Delivery (Android) ──────────────────────────────
  static const _assetPackChannel = MethodChannel('com.finailabz.ocula/asset_pack');

  /// On Android with Play Store install, check if the model file is available
  /// in the Play Asset Delivery "models_pack" asset pack.
  /// Returns the file path if found, null otherwise.
  Future<String?> _findAssetPackPath(String fileName) async {
    if (!Platform.isAndroid) return null;
    try {
      final path = await _assetPackChannel.invokeMethod<String>(
        'getAssetPackPath',
        {'packName': 'models_pack', 'fileName': fileName},
      );
      if (path != null) {
        debugPrint('[ModelManager] Found model in asset pack: $path');
      }
      return path;
    } catch (e) {
      // Asset pack not available (sideloaded APK, not from Play Store)
      debugPrint('[ModelManager] Asset pack not available: $e');
      return null;
    }
  }

  /// Resolve the physical file path of a model asset on disk.
  /// Avoids rootBundle.load() which loads the entire file into RAM.
  /// Returns null if the asset can't be found — caller falls through to network download.
  ///
  /// Android: checks Play Asset Delivery (fast-follow pack)
  /// iOS:     checks On-Demand Resources (ODR) — silently returns null in dev/TestFlight
  /// macOS:   checks physical Flutter asset path inside the app bundle
  Future<String?> _findBundledAssetPath(String assetKey, {DownloadProgress? onProgress}) async {
    // Android: try Play Asset Delivery pack first (Play Store installs)
    if (Platform.isAndroid) {
      final fileName = assetKey.split('/').last;
      return await _findAssetPackPath(fileName);
    }

    // iOS: try On-Demand Resources first (App Store installs with ODR configured)
    if (Platform.isIOS) {
      final fileName = assetKey.split('/').last;
      return await _findODRPath(fileName, onProgress: onProgress);
    }

    try {
      // macOS: Flutter assets are physically on disk inside the app bundle.
      // macOS: <bundle>/Contents/Frameworks/App.framework/Resources/flutter_assets/<assetKey>
      final exe = Platform.resolvedExecutable;
      final bundleDir = File(exe).parent.path;

      final candidates = [
        // macOS
        '$bundleDir/Contents/Frameworks/App.framework/Resources/flutter_assets/$assetKey',
        // macOS alternate
        '$bundleDir/../Frameworks/App.framework/Resources/flutter_assets/$assetKey',
      ];

      for (final path in candidates) {
        final file = File(path);
        if (await file.exists()) {
          debugPrint('[ModelManager] Found bundled asset at: $path');
          return path;
        }
      }

      debugPrint(
        '[ModelManager] Bundled asset not found on disk for: $assetKey',
      );
      debugPrint('[ModelManager] Searched: $candidates');
      return null;
    } catch (e) {
      debugPrint('[ModelManager] Error finding bundled asset path: $e');
      return null;
    }
  }

  /// Check if a valid model is bundled as an asset with the app.
  /// Uses physical file path to avoid loading entire model into RAM.
  /// On Android (no physical path), we check if the asset exists at all
  /// by loading only the first few KB via rootBundle.
  Future<bool> _isValidBundledModel(String fileName) async {
    try {
      // Strategy 1: zero-copy physical path check (iOS/macOS)
      final physicalPath = await _findBundledAssetPath(
        'assets/models/$fileName',
      );
      if (physicalPath != null) {
        final file = File(physicalPath);
        final size = await file.length();
        if (size < 1024 * 1024) return false;

        // Read just the first 4 bytes to check GGUF header
        final raf = await file.open(mode: FileMode.read);
        try {
          final header = await raf.read(4);
          final magic = String.fromCharCodes(header);
          return magic == 'GGUF';
        } finally {
          await raf.close();
        }
      }

      // Strategy 2 (Android / other): rootBundle.load loads everything into RAM.
      // Instead, just try to load the asset — if it throws, it doesn't exist.
      // We accept the OOM risk here only for the *check* call; the actual
      // copy (_ensureBundledModelCopied) uses chunked writing.
      // On Android the asset exists inside the APK; rootBundle is the only way.
      debugPrint('[ModelManager] Falling back to rootBundle for $fileName');
      final bundledData = await rootBundle.load('assets/models/$fileName');
      final bytes = bundledData.buffer.asUint8List();
      if (bytes.length < 1024 * 1024) return false;
      if (bytes.length > 4) {
        final header = String.fromCharCodes(bytes.take(4));
        return header == 'GGUF';
      }
      return false;
    } catch (e) {
      debugPrint(
        '[ModelManager] _isValidBundledModel failed for $fileName: $e',
      );
      return false;
    }
  }

  /// Copy a bundled model to local storage if it doesn't exist.
  /// Uses streamed file copy from the physical asset path to avoid OOM.
  /// Falls back to rootBundle.load() only on platforms where physical path
  /// is unavailable (should not happen on iOS/macOS/Android).
  ///
  /// [onProgress] is forwarded to _findBundledAssetPath → _findODRPath so the
  /// splash screen shows live download progress during ODR delivery.
  Future<bool> _ensureBundledModelCopied(String fileName, {DownloadProgress? onProgress}) async {
    // Handle split GGUF models (multiple ODR/PAD asset packs on iOS/Android).
    final allParts = _allSplitParts(fileName);
    if (allParts.length > 1) {
      // Look up declared model size so _ensureSplitModelCopied can validate
      // that total shard bytes are within 90% of what we expect.
      final modelInfo = models.where((m) => m.fileName == fileName).firstOrNull;
      return _ensureSplitModelCopied(
        allParts,
        onProgress: onProgress,
        expectedTotalBytes: modelInfo?.sizeBytes,
      );
    }

    final localPath = await modelPath(fileName);
    final localFile = File(localPath);

    // Already exists locally and is valid
    if (localFile.existsSync() && await localFile.length() > 1024 * 1024) {
      debugPrint(
        '[ModelManager] $fileName already in local storage, skipping copy',
      );
      return true;
    }

    try {
      // ── Strategy 1: Streamed file copy from physical asset path ──
      // This avoids loading the entire model (100MB+) into RAM.
      final physicalPath = await _findBundledAssetPath(
        'assets/models/$fileName',
        onProgress: onProgress,
      );

      if (physicalPath != null) {
        final sourceFile = File(physicalPath);
        final sourceSize = await sourceFile.length();

        if (sourceSize < 1024 * 1024) {
          debugPrint(
            '[ModelManager] Bundled $fileName too small (${sourceSize} bytes)',
          );
          return false;
        }

        // Validate GGUF header (read only 4 bytes)
        final raf = await sourceFile.open(mode: FileMode.read);
        final header = await raf.read(4);
        await raf.close();
        if (String.fromCharCodes(header) != 'GGUF') {
          debugPrint(
            '[ModelManager] Bundled $fileName has invalid GGUF header',
          );
          return false;
        }

        // Streamed copy — constant memory usage regardless of file size
        debugPrint(
          '[ModelManager] Copying bundled $fileName (${(sourceSize / (1024 * 1024)).toStringAsFixed(1)} MB) via streamed file copy...',
        );
        await localFile.create(recursive: true);
        await sourceFile.copy(localPath);
        debugPrint('[ModelManager] ✓ Copied $fileName to local storage');
        return true;
      }

      // ── Strategy 2: Android / other — chunked write via rootBundle ──
      // On Android, assets are inside the APK and can only be read via rootBundle.
      // We load into an ImmutableBuffer (native heap, not Dart heap) when possible,
      // then write in chunks to avoid Dart GC pressure.
      debugPrint(
        '[ModelManager] Physical path not found, falling back to rootBundle for $fileName',
      );

      final bundledData = await rootBundle.load('assets/models/$fileName');
      final bytes = bundledData.buffer.asUint8List();

      if (bytes.length < 1024 * 1024) {
        debugPrint(
          '[ModelManager] Bundled $fileName too small (${bytes.length} bytes)',
        );
        return false;
      }
      if (bytes.length > 4) {
        final header = String.fromCharCodes(bytes.take(4));
        if (header != 'GGUF') {
          debugPrint(
            '[ModelManager] Bundled $fileName has invalid GGUF header',
          );
          return false;
        }
      }

      // Write in 4MB chunks to reduce peak Dart heap pressure
      debugPrint(
        '[ModelManager] Writing $fileName in chunks (${(bytes.length / (1024 * 1024)).toStringAsFixed(1)} MB)...',
      );
      await localFile.create(recursive: true);
      final sink = localFile.openWrite();
      const chunkSize = 4 * 1024 * 1024; // 4 MB
      for (int offset = 0; offset < bytes.length; offset += chunkSize) {
        final end = (offset + chunkSize).clamp(0, bytes.length);
        sink.add(bytes.sublist(offset, end));
        // Yield to event loop between chunks
        await sink.flush();
      }
      await sink.close();
      debugPrint(
        '[ModelManager] ✓ Copied $fileName via rootBundle (${(bytes.length / (1024 * 1024)).toStringAsFixed(1)} MB)',
      );
      return true;
    } catch (e, stack) {
      debugPrint('[ModelManager] Failed to copy bundled model $fileName: $e');
      debugPrint('[ModelManager] Stack: $stack');
      // Clean up partial file
      if (await localFile.exists()) {
        await localFile.delete();
      }
      return false;
    }
  }

  /// Get status of a specific model.
  Future<ModelStatus> getStatus(String fileName) async {
    if (await isDownloaded(fileName)) return ModelStatus.ready;
    final prefs = await SharedPreferences.getInstance();
    final downloading = prefs.getBool('downloading_$fileName') ?? false;
    if (downloading) return ModelStatus.downloading;
    return ModelStatus.notDownloaded;
  }

  /// Check if the free model is ready (needed for first launch).
  ///
  /// Primary path: validate every shard against the manifest written after the
  /// last successful atomic copy — exact byte counts, no relative heuristics.
  ///
  /// Fallback: monolithic network-download case (full model stored under the
  /// part-1 filename, ≥ 80% of declared size + valid GGUF header).
  /// Split shards WITHOUT a manifest are NOT trusted — they may be partial.
  Future<bool> get isFreeModelReady async {
    // ── Primary: manifest-based validation ──────────────────────────────────
    final manifest = await _readFreeShardManifest();
    if (manifest != null) {
      try {
        final shards =
            (manifest['shards'] as List).cast<Map<String, dynamic>>();
        for (final shard in shards) {
          final name = shard['name'] as String;
          final expectedBytes = shard['bytes'] as int;
          final path = await modelPath(name);
          final f = File(path);
          if (!f.existsSync()) {
            debugPrint('[ModelManager] isFreeModelReady: missing shard $name');
            return false;
          }
          final actualSize = await f.length();
          if (actualSize != expectedBytes) {
            debugPrint(
              '[ModelManager] isFreeModelReady: shard $name size mismatch '
              '(expected=$expectedBytes actual=$actualSize)',
            );
            return false;
          }
          if (!_hasGGUFHeaderSync(path)) {
            debugPrint(
                '[ModelManager] isFreeModelReady: shard $name invalid GGUF header');
            return false;
          }
        }
        debugPrint(
            '[ModelManager] isFreeModelReady: all ${shards.length} shards verified ✓');
        return true;
      } catch (e) {
        debugPrint('[ModelManager] isFreeModelReady: manifest parse error — $e');
        // Fall through to monolithic check
      }
    }

    // ── Fallback A: split network-download (no manifest) ────────────────────
    // The direct HTTP path downloads all split parts but does not write
    // the ODR/PAD manifest. Accept this path if all parts exist, pass
    // GGUF header checks, and each part is plausibly sized.
    final freeModel = models.firstWhere(
      (m) =>
          m.tier == AITier.free &&
          !m.isVisionProjector &&
          !m.isEmbeddingModel,
    );
    final splitParts = _allSplitParts(freeModel.fileName);
    if (splitParts.length > 1) {
      final perPartExpected = freeModel.sizeBytes ~/ splitParts.length;
      var allGood = true;
      for (final part in splitParts) {
        final partPath = await modelPath(part);
        final ok = await isValidLocalModel(
          partPath,
          expectedSizeBytes: perPartExpected,
        );
        if (!ok) {
          allGood = false;
          break;
        }
      }
      if (allGood) {
        debugPrint(
          '[ModelManager] isFreeModelReady: split fallback verified ${splitParts.length} parts ✓',
        );
        return true;
      }
    }

    // ── Fallback B: monolithic network-download ──────────────────────────────
    // The network download path saves the full model under the part-1 filename.
    // Accept it only if it is ≥ 80% of declared size AND has a valid GGUF header.
    // Split shards without a manifest are explicitly rejected — they could be
    // partial copies from a previous interrupted ODR/PAD session.
    final part1Path = await modelPath(freeModel.fileName);
    final part1 = File(part1Path);
    if (!part1.existsSync()) return false;
    final part1Size = await part1.length();

    if (part1Size >= (freeModel.sizeBytes * 0.8).toInt()) {
      final ok = _hasGGUFHeaderSync(part1Path);
      debugPrint(
          '[ModelManager] isFreeModelReady: monolithic fallback header=$ok size=$part1Size');
      return ok;
    }

    // Not manifest-verified, split-verified, or monolithic-valid.
    debugPrint(
      '[ModelManager] isFreeModelReady: no manifest, part-1 size=$part1Size '
      '(< 80% of ${freeModel.sizeBytes}) — not ready',
    );
    return false;
  }

  /// Get all models needed for a tier.
  List<ModelInfo> modelsForTier(AITier tier) {
    return tier == AITier.enterprise
        ? _enterpriseModels
        : models.where((m) => m.tier == tier).toList();
  }

  /// Validate that a local file is a plausible GGUF model.
  ///
  /// Checks:
  /// 1. File exists
  /// 2. No `.partial` sibling (download still in progress)
  /// 3. Size ≥ [minBytes] (default 1 MB — rejects stubs / error pages)
  /// 4. First 4 bytes == 'GGUF' magic header
  ///
  /// If [expectedSizeBytes] is provided, also checks that the file is at
  /// least 95% of the expected size (allows minor metadata differences
  /// between quantisation runs but catches truncated downloads).
  Future<bool> isValidLocalModel(
    String path, {
    int minBytes = 1024 * 1024,
    int? expectedSizeBytes,
  }) async {
    try {
      final file = File(path);
      if (!file.existsSync()) return false;

      // Reject if a .partial sibling exists — download is still in flight
      if (File('$path.partial').existsSync()) return false;

      final size = file.lengthSync();
      if (size < minBytes) return false;

      // Expected-size gate: reject files that are < 95% of the expected size.
      // This catches downloads that completed HTTP-wise but were truncated
      // (e.g. CDN timeout, partial Range response treated as full).
      if (expectedSizeBytes != null &&
          size < (expectedSizeBytes * 0.95).round()) {
        return false;
      }

      // GGUF magic header check (4 bytes, no RAM cost)
      final raf = file.openSync(mode: FileMode.read);
      try {
        final header = raf.readSync(4);
        if (header.length < 4) return false;
        return String.fromCharCodes(header) == 'GGUF';
      } finally {
        raf.closeSync();
      }
    } catch (_) {
      return false;
    }
  }

  /// Get the main model path for a tier (not the vision projector).
  /// Only returns paths in local storage (writable, mmap-able).
  /// The app bundle is NOT usable because llama.cpp needs mmap and
  /// the iOS bundle is code-signed / read-only → SIGSEGV.
  Future<String?> mainModelPath(AITier tier) async {
    if (tier == AITier.enterprise) {
      return await enterpriseModelPath;
    }

    final model = models
        .where(
          (m) => m.tier == tier && !m.isVisionProjector && !m.isEmbeddingModel,
        )
        .firstOrNull;
    if (model == null) return null;

    final path = await modelPath(model.fileName);
    if (File(path).existsSync()) return path;
    return await _findCompatibleTierModelPath(tier, isVisionProjector: false);
  }

  /// Get the embedding model path (tier-independent — shared across all tiers).
  /// Returns null if not downloaded yet.
  Future<String?> embeddingModelPath() async {
    final model = models.where((m) => m.isEmbeddingModel).firstOrNull;
    if (model == null) return null;
    final path = await modelPath(model.fileName);
    if (File(path).existsSync()) return path;
    return null;
  }

  /// Get vision projector path for a tier (if any).
  /// Only returns paths in local storage (writable, mmap-able).
  Future<String?> visionProjectorPath(AITier tier) async {
    final model = models
        .where((m) => m.tier == tier && m.isVisionProjector)
        .firstOrNull;
    if (model == null) return null;

    final path = await modelPath(model.fileName);
    if (File(path).existsSync()) return path;
    return await _findCompatibleTierModelPath(tier, isVisionProjector: true);
  }

  /// Fallback resolver for tier models when filenames differ from registry
  /// (e.g. finetuned exports such as Qwen3-VL-2B-finetuned-Q4_K_M.gguf).
  Future<String?> _findCompatibleTierModelPath(
    AITier tier, {
    required bool isVisionProjector,
  }) async {
    try {
      final dir = Directory(await modelsDir);
      if (!dir.existsSync()) return null;

      final files = dir
          .listSync()
          .whereType<File>()
          .where((f) => f.path.toLowerCase().endsWith('.gguf'))
          .toList();

      bool matches(File f) {
        final n = p.basename(f.path).toLowerCase();
        if (isVisionProjector) {
          if (tier == AITier.free)
            return n.contains('mmproj') &&
                (n.contains('smolvlm') || n.contains('qwen'));
          if (tier == AITier.plus)
            return n.contains('mmproj') &&
                (n.contains('moondream') || (n.contains('qwen') && n.contains('2b')));
          if (tier == AITier.pro)
            return n.contains('mmproj') && n.contains('qwen') && (n.contains('4b') || n.contains('7b'));
          return false;
        }
        if (tier == AITier.free)
          return (n.contains('smolvlm') ||
                  n.contains('qwen3-1.7b') ||
                  n.contains('qwen2.5-1.5b') ||
                  (n.contains('qwen') && n.contains('1.7b'))) &&
              !n.contains('mmproj');
        if (tier == AITier.plus)
          return (n.contains('moondream') ||
                  (n.contains('qwen') && n.contains('2b')) ||
                  n.contains('qwen3vl-2b')) &&
              !n.contains('mmproj');
        if (tier == AITier.pro) {
          return n.contains('qwen') &&
              !n.contains('mmproj') &&
              (n.contains('4b') || n.contains('2b') || n.contains('vl'));
        }
        return false;
      }

      final candidates = files.where(matches).toList()
        ..sort((a, b) => b.lastModifiedSync().compareTo(a.lastModifiedSync()));

      if (candidates.isEmpty) return null;
      final chosen = candidates.first.path;
      debugPrint(
        '[ModelManager] Using compatible tier model for ${tier.name}: $chosen',
      );
      return chosen;
    } catch (e) {
      debugPrint('[ModelManager] _findCompatibleTierModelPath failed: $e');
      return null;
    }
  }

  /// Check if any model for a given tier is downloaded.
  Future<bool> isAnyModelDownloadedForTier(AITier tier) async {
    final tierModels = modelsForTier(tier);
    if (tierModels.isEmpty) return false;

    for (final model in tierModels) {
      if (await isDownloaded(model.fileName)) {
        return true;
      }
    }
    return false;
  }

  /// Available disk space in bytes.
  Future<int> get availableDiskSpace async {
    // Platform-specific; rough estimate via temp file
    try {
      final dir = await modelsDir;
      final stat = await FileStat.stat(dir);
      // FileStat doesn't give free space directly.
      // Use a conservative estimate — allow download if > 500MB free.
      // In production, use platform channels for actual free space.
      return stat.size >= 0 ? 10 * 1024 * 1024 * 1024 : 0; // assume 10GB
    } catch (_) {
      return 0;
    }
  }

  /// Resolve download URL for the current platform.
  /// Android emulator uses 10.0.2.2 to reach the host machine's localhost.
  /// Also rebases hardcoded URLs to the configured model server.
  String _resolveUrl(String url) {
    // Rebase from hardcoded default to the configured server
    if (_cachedModelServerUrl != null && _cachedModelServerUrl!.isNotEmpty) {
      final base = _cachedModelServerUrl!;
      // Replace the hardcoded base URL with the configured one
      url = url.replaceFirst(
        'https://backend-ocula.finailabz.com',
        base.endsWith('/') ? base.substring(0, base.length - 1) : base,
      );
    }
    if (Platform.isAndroid) {
      // Android emulator can't reach the host's LAN IP or localhost directly;
      // 10.0.2.2 is the emulator's alias for the host machine.
      final uri = Uri.parse(url);
      if (uri.host == 'localhost' ||
          uri.host == '127.0.0.1' ||
          _isPrivateIp(uri.host)) {
        return url.replaceFirst('://${uri.host}:', '://10.0.2.2:');
      }
    }
    return url;
  }

  /// Check if an IP is a private/LAN address (192.168.x.x, 10.x.x.x, 172.16-31.x.x).
  static bool _isPrivateIp(String host) {
    final parts = host.split('.');
    if (parts.length != 4) return false;
    final a = int.tryParse(parts[0]) ?? -1;
    final b = int.tryParse(parts[1]) ?? -1;
    return a == 10 ||
        (a == 172 && b >= 16 && b <= 31) ||
        (a == 192 && b == 168);
  }

  /// Get the current model server URL.
  Future<String> getModelServerUrl() async {
    if (_cachedModelServerUrl != null) return _cachedModelServerUrl!;
    final prefs = await SharedPreferences.getInstance();
    _cachedModelServerUrl =
        prefs.getString(_prefKeyModelServer) ?? defaultModelServerUrl;
    return _cachedModelServerUrl!;
  }

  /// Set the model server URL.
  Future<void> setModelServerUrl(String url) async {
    final uri = Uri.tryParse(url);
    if (uri == null || uri.scheme.toLowerCase() != 'https') {
      throw ArgumentError('Model server URL must use https://');
    }
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefKeyModelServer, url);
    _cachedModelServerUrl = url;
  }

  /// Download a model with progress tracking.
  /// Supports resume via Range header if partial file exists.
  Future<bool> download(ModelInfo model, {DownloadProgress? onProgress}) async {
    // Ensure model server URL is loaded before downloading
    await getModelServerUrl();

    final path = await modelPath(model.fileName);
    final file = File(path);
    final partialFile = File('$path.partial');

    // Already downloaded — but validate it's actually a good GGUF file.
    // A previous interrupted download or CDN error could have left a bad file.
    // For split GGUF models, model.sizeBytes is the TOTAL across all parts;
    // each individual part is ~1/N of that, so use per-part expected size.
    final _splitPartsCount = () {
      final m = RegExp(r'-\d{5}-of-(\d{5})\.gguf$').firstMatch(model.fileName);
      return m != null ? int.parse(m.group(1)!) : 1;
    }();
    final _perPartExpected = model.sizeBytes ~/ _splitPartsCount;
    if (await file.exists()) {
      if (await isValidLocalModel(path, expectedSizeBytes: _perPartExpected)) {
        onProgress?.call(1.0, 'Ready');
        return true;
      }
      // Bad file at final path — delete and re-download
      debugPrint(
        '[ModelManager] ${model.fileName} exists but failed validation — re-downloading',
      );
      try { await file.delete(); } catch (_) {}
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('downloading_${model.fileName}', true);

    try {
      _httpClient ??= HttpClient()
        ..connectionTimeout = const Duration(seconds: 15);
      final resolvedUrl = _resolveUrl(model.downloadUrl);
      final resolvedUri = Uri.parse(resolvedUrl);
      if (resolvedUri.scheme.toLowerCase() != 'https') {
        throw Exception('Blocked insecure model download URL: $resolvedUrl');
      }
      debugPrint('[ModelManager] Downloading from: $resolvedUrl');
      final request = await _httpClient!.getUrl(resolvedUri);

      // Resume support — if partial file exists, request remaining bytes
      int existingBytes = 0;
      if (await partialFile.exists()) {
        existingBytes = await partialFile.length();
        request.headers.set('Range', 'bytes=$existingBytes-');
        onProgress?.call(existingBytes / model.sizeBytes, 'Resuming...');
      } else {
        onProgress?.call(0.0, 'Starting download...');
      }

      final response = await request.close();

      // If server doesn't support range, start over
      if (response.statusCode == 200) {
        existingBytes = 0;
        if (await partialFile.exists()) await partialFile.delete();
      } else if (response.statusCode != 206) {
        throw Exception('HTTP ${response.statusCode}');
      }

      final sink = partialFile.openWrite(
        mode: existingBytes > 0 ? FileMode.append : FileMode.write,
      );

      int received = existingBytes;
      final total = model.sizeBytes;
      bool cancelled = false;

      await for (final chunk in response) {
        if (_cancelledDownloads.contains(model.fileName)) {
          cancelled = true;
          break;
        }
        sink.add(chunk);
        received += chunk.length;
        final progress = (received / total).clamp(0.0, 1.0);
        final mb = (received / (1024 * 1024)).toStringAsFixed(1);
        final totalMb = (total / (1024 * 1024)).toStringAsFixed(0);

        onProgress?.call(progress, '$mb / $totalMb MB');
      }

      await sink.flush();
      await sink.close();

      if (cancelled) {
        _cancelledDownloads.remove(model.fileName);
        _downloadProgress.remove(model.fileName);
        _downloadProgressStreamController.add(Map.from(_downloadProgress));
        try { await partialFile.delete(); } catch (_) {}
        await prefs.setBool('downloading_${model.fileName}', false);
        debugPrint('[ModelManager] Download cancelled: ${model.fileName}');
        return false;
      }

      // Move partial → final
      await partialFile.rename(path);

      // Post-download GGUF validation — catch corrupt / truncated files
      // before signalling "ready" to the rest of the system.
      // _perPartExpected already accounts for split models (see above).
      final valid = await isValidLocalModel(
        path,
        expectedSizeBytes: _perPartExpected,
      );
      if (!valid) {
        debugPrint(
          '[ModelManager] ✗ ${model.fileName} failed GGUF validation after download — deleting',
        );
        try { await File(path).delete(); } catch (_) {}
        await prefs.setBool('downloading_${model.fileName}', false);
        onProgress?.call(
          0.0,
          'Error: downloaded file is not a valid GGUF model',
        );
        return false;
      }

      // For part 1 of a split-GGUF: auto-download remaining parts.
      // Sibling URLs are derived by substituting the part number in downloadUrl.
      // The guard (group(1) == 1) prevents infinite recursion when parts 2/3
      // call download() — they skip this block entirely.
      final splitMatch =
          RegExp(r'-(\d{5})-of-(\d{5})\.gguf$').firstMatch(model.fileName);
      if (splitMatch != null &&
          int.parse(splitMatch.group(1)!) == 1 &&
          int.parse(splitMatch.group(2)!) > 1) {
        final allParts = _allSplitParts(model.fileName);
        final partSizeBytes = model.sizeBytes ~/ allParts.length;
        final urlPartPattern = RegExp(r'-\d{5}-of-\d{5}\.gguf$');
        for (int i = 1; i < allParts.length; i++) {
          final partNum = (i + 1).toString().padLeft(5, '0');
          final totalNum = allParts.length.toString().padLeft(5, '0');
          final partUrl = model.downloadUrl
              .replaceFirst(urlPartPattern, '-$partNum-of-$totalNum.gguf');
          final partModel = ModelInfo(
            fileName: allParts[i],
            displayName: model.displayName,
            downloadUrl: partUrl,
            sizeBytes: partSizeBytes,
            tier: model.tier,
          );
          debugPrint(
            '[ModelManager] Downloading split part ${i + 1}/${allParts.length}: ${allParts[i]}',
          );
          // Offset progress so sibling parts continue from where part 1 left off
          // instead of resetting to 0. Part 1 already consumed 0→(1/N), so
          // part i starts at i/N and spans 1/N of the total bar.
          final partOffset = i / allParts.length;
          final partScale = 1.0 / allParts.length;
          final partOk = await download(
            partModel,
            onProgress: onProgress == null
                ? null
                : (p, s) => onProgress((partOffset + p * partScale).clamp(0.0, 1.0), s),
          );
          if (!partOk) {
            await prefs.setBool('downloading_${model.fileName}', false);
            return false;
          }
        }
      }

      await prefs.setBool('downloading_${model.fileName}', false);
      onProgress?.call(1.0, 'Ready');
      return true;
    } catch (e) {
      await prefs.setBool('downloading_${model.fileName}', false);
      onProgress?.call(0.0, 'Error: $e');
      debugPrint('Model download error: $e');
      return false;
    }
  }

  /// Download all models for a tier.
  Future<bool> downloadTier(AITier tier, {DownloadProgress? onProgress}) async {
    final tierModels = modelsForTier(tier);
    for (int i = 0; i < tierModels.length; i++) {
      final model = tierModels[i];
      final prefix = tierModels.length > 1
          ? '(${i + 1}/${tierModels.length}) '
          : '';
      final success = await download(
        model,
        onProgress: (p, s) =>
            onProgress?.call((i + p) / tierModels.length, '$prefix$s'),
      );
      if (!success) return false;
    }
    return true;
  }

  /// Delete a downloaded model to free disk space.
  Future<void> deleteModel(String fileName) async {
    final path = await modelPath(fileName);
    final file = File(path);
    if (await file.exists()) {
      await file.delete();
    }
    // Also clean up partial files
    final partial = File('$path.partial');
    if (await partial.exists()) {
      await partial.delete();
    }
  }

  /// Delete all models for a tier.
  Future<void> deleteTier(AITier tier) async {
    for (final model in modelsForTier(tier)) {
      await deleteModel(model.fileName);
    }
  }

  /// Total downloaded size in bytes.
  Future<int> get totalDownloadedSize async {
    int total = 0;
    for (final model in models) {
      final path = await modelPath(model.fileName);
      final file = File(path);
      if (await file.exists()) {
        total += await file.length();
      }
    }
    return total;
  }

  /// Human-readable total size string.
  Future<String> get totalSizeLabel async {
    final bytes = await totalDownloadedSize;
    final mb = bytes / (1024 * 1024);
    if (mb >= 1024) return '${(mb / 1024).toStringAsFixed(1)} GB';
    return '${mb.round()} MB';
  }

  /// Clear downloaded model files from storage (advanced feature).
  /// This will delete all downloaded model files to free up disk space.
  Future<bool> clearModelFiles() async {
    try {
      final dir = await modelsDir;
      final modelsDirectory = Directory(dir);
      if (await modelsDirectory.exists()) {
        await modelsDirectory.delete(recursive: true);
        // Reset cached directory path
        _modelsDir = null;
        return true;
      }
      return true;
    } catch (e) {
      debugPrint('Failed to clear model files: $e');
      return false;
    }
  }

  /// Get total size of downloaded models in bytes.
  Future<int> getTotalModelSize() async {
    try {
      final dir = await modelsDir;
      final modelsDirectory = Directory(dir);
      if (!await modelsDirectory.exists()) return 0;

      int totalSize = 0;
      await for (final entity in modelsDirectory.list()) {
        if (entity is File) {
          final stat = await entity.stat();
          totalSize += stat.size;
        }
      }
      return totalSize;
    } catch (e) {
      debugPrint('Failed to calculate model size: $e');
      return 0;
    }
  }

  /// Refresh enterprise models based on current settings.
  Future<void> refreshEnterpriseModels() async {
    final prefs = await SharedPreferences.getInstance();
    final isEnabled = prefs.getBool('enterprise_enabled') ?? false;

    if (!isEnabled) {
      _enterpriseModels.clear();
      return;
    }

    final useLocal = prefs.getBool('enterprise_use_local') ?? true;
    final modelPath = prefs.getString('enterprise_model_path') ?? '';
    final modelUrl = prefs.getString('enterprise_model_url') ?? '';

    _enterpriseModels.clear();

    if (useLocal && modelPath.isNotEmpty) {
      // Create a virtual ModelInfo for the local enterprise model
      final fileName = modelPath.split('/').last;
      final file = File(modelPath);
      final sizeBytes = await file.exists() ? await file.length() : 0;

      _enterpriseModels.add(
        ModelInfo(
          fileName: fileName,
          displayName: 'Enterprise Model',
          downloadUrl: '', // Local file, no download URL
          sizeBytes: sizeBytes,
          tier: AITier.enterprise,
        ),
      );
    } else if (!useLocal && modelUrl.isNotEmpty) {
      // Create a virtual ModelInfo for the remote API
      _enterpriseModels.add(
        ModelInfo(
          fileName: 'enterprise-api',
          displayName: 'Enterprise API',
          downloadUrl: modelUrl,
          sizeBytes: 0, // API, no local size
          tier: AITier.enterprise,
        ),
      );
    }
  }

  /// Get enterprise model path if configured locally.
  Future<String?> get enterpriseModelPath async {
    final prefs = await SharedPreferences.getInstance();
    final isEnabled = prefs.getBool('enterprise_enabled') ?? false;
    final useLocal = prefs.getBool('enterprise_use_local') ?? true;

    if (!isEnabled || !useLocal) return null;

    final modelPath = prefs.getString('enterprise_model_path') ?? '';
    if (modelPath.isEmpty) return null;

    final file = File(modelPath);
    return await file.exists() ? modelPath : null;
  }
}
