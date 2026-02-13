import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
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
  // Served by serve_models.sh (python3 http.server on port 8080)
  //
  // Tier free  (Sensor)     – SmolVLM2-500M: fast response, video support, better instruction following
  // Tier plus  (Specialist) – Moondream 2:   pointing (x,y) & counting
  // Tier pro   (Thinker)    – Qwen3-VL-2B:   chain-of-thought reasoning
  // Embedding                – all-MiniLM-L6-v2: sentence similarity for RAG
  static const models = [
    // ── SENSOR: Always-on, fast & capable ──
    ModelInfo(
      fileName: 'SmolVLM2-500M-Video-Instruct-Q8_0.gguf',
      displayName: 'Sensor Engine',
      downloadUrl: 'https://backend-ocula.finailabz.com/models/SmolVLM2-500M-Video-Instruct-Q8_0.gguf',
      sizeBytes: 437 * 1024 * 1024, // ~437 MB
      tier: AITier.free,
    ),
    ModelInfo(
      fileName: 'mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf',
      displayName: 'Sensor Vision',
      downloadUrl: 'https://backend-ocula.finailabz.com/models/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf',
      sizeBytes: 109 * 1024 * 1024, // ~109 MB
      tier: AITier.free,
      isVisionProjector: true,
    ),
    // ── EMBEDDING: Sentence similarity for RAG search ──
    ModelInfo(
      fileName: 'all-MiniLM-L6-v2.Q8_0.gguf',
      displayName: 'Search Engine',
      downloadUrl: 'https://backend-ocula.finailabz.com/models/all-MiniLM-L6-v2.Q8_0.gguf',
      sizeBytes: 25 * 1024 * 1024, // ~25 MB
      tier: AITier.free,
      isEmbeddingModel: true,
    ),
    // ── SPECIALIST: Spatial tasks, pointing & counting ──
    ModelInfo(
      fileName: 'moondream2-text-model-Q4_K_M.gguf',
      displayName: 'Specialist Engine',
      downloadUrl: 'https://backend-ocula.finailabz.com/models/moondream2-text-model-Q4_K_M.gguf',
      sizeBytes: 900 * 1024 * 1024, // ~900 MB (Q4_K_M quantized)
      tier: AITier.plus,
    ),
    ModelInfo(
      fileName: 'moondream2-mmproj-f16-20250414.gguf',
      displayName: 'Specialist Vision',
      downloadUrl: 'https://backend-ocula.finailabz.com/models/moondream2-mmproj-f16-20250414.gguf',
      sizeBytes: 910 * 1024 * 1024, // ~910 MB
      tier: AITier.plus,
      isVisionProjector: true,
    ),
    // ── THINKER: Reasoning with chain-of-thought ──
    ModelInfo(
      fileName: 'Qwen3VL-2B-Thinking-Q4_K_M.gguf',
      displayName: 'Thinker Engine',
      downloadUrl: 'https://backend-ocula.finailabz.com/models/Qwen3VL-2B-Thinking-Q4_K_M.gguf',
      sizeBytes: 1110 * 1024 * 1024, // ~1.11 GB
      tier: AITier.pro,
    ),
    ModelInfo(
      fileName: 'mmproj-Qwen3VL-2B-Thinking-F16.gguf',
      displayName: 'Thinker Vision',
      downloadUrl: 'https://backend-ocula.finailabz.com/models/mmproj-Qwen3VL-2B-Thinking-F16.gguf',
      sizeBytes: 819 * 1024 * 1024, // ~819 MB
      tier: AITier.pro,
      isVisionProjector: true,
    ),
  ];

  String? _modelsDir;
  HttpClient? _httpClient;
  List<ModelInfo> _enterpriseModels = [];
  final Map<String, double> _downloadProgress = {};
  final StreamController<Map<String, double>> _downloadProgressStreamController = StreamController.broadcast();

  /// Emits user-friendly feature names when a tier's models are fully downloaded.
  /// Values: 'Quick Scan', 'Detail & Counting', 'Vision & Deep Analysis'
  final StreamController<String> _featureReadyController = StreamController.broadcast();

  /// Emits the AITier enum whenever that tier's models are fully downloaded.
  /// AIManager subscribes to this to set a pending-upgrade flag.
  final StreamController<AITier> _tierReadyController = StreamController.broadcast();

  Stream<Map<String, double>> get downloadProgressStream => _downloadProgressStreamController.stream;

  /// Subscribe to this for user-facing "feature X is ready" notifications.
  Stream<String> get featureReadyStream => _featureReadyController.stream;

  /// Subscribe to this for programmatic "tier X is ready" signals.
  /// AIManager listens here to set the pending-upgrade flag.
  Stream<AITier> get tierReadyStream => _tierReadyController.stream;

  /// Human-friendly feature label for each tier (never expose model names).
  static String featureLabel(AITier tier) {
    switch (tier) {
      case AITier.free:
        return 'Quick Scan';
      case AITier.plus:
        return 'Spatial & Counting';
      case AITier.pro:
        return 'Deep Thinking';
      case AITier.enterprise:
        return 'Enterprise Backend';
    }
  }

  /// Download the free-tier main model only. Returns true when it's ready.
  /// Intended to run during the splash screen so the user can interact ASAP.
  /// First tries to copy bundled model, then downloads if needed.
  /// Vision projector and embedding model are fetched in the background.
  Future<bool> ensureFreeModelReady({
    DownloadProgress? onProgress,
  }) async {
    // Only the main model (not the projector or embedding) is required to start
    final mainModel = models
        .where((m) => m.tier == AITier.free && !m.isVisionProjector && !m.isEmbeddingModel)
        .first;

    // Step 1: Try to copy bundled model first (instant)
    onProgress?.call(0.1, 'Checking for bundled AI engine...');
    if (await _ensureBundledModelCopied(mainModel.fileName)) {
      onProgress?.call(1.0, 'Bundled AI engine ready!');
    } else if (await isDownloaded(mainModel.fileName)) {
      // Step 2: Already downloaded
      onProgress?.call(1.0, 'AI engine ready!');
    } else {
      // Step 3: Download as fallback
      onProgress?.call(0.0, 'Downloading AI engine...');
      final ok = await download(mainModel, onProgress: onProgress);
      if (!ok) return false;
    }

    // Try to copy the vision projector too — AWAIT it so vision is ready
    // immediately. Without this, the first image query after launch fails
    // because the projector copy is still in progress.
    final projector = models
        .where((m) => m.tier == AITier.free && m.isVisionProjector)
        .firstOrNull;
    if (projector != null) {
      onProgress?.call(0.9, 'Preparing vision engine...');
      final projCopied = await _ensureBundledModelCopied(projector.fileName);
      if (!projCopied && !await isDownloaded(projector.fileName)) {
        // Last resort: download projector — don't block splash if it fails
        download(projector).catchError((_) => false);
      }
    }

    // Ensure embedding model is available (bundled at ~25MB, copies instantly)
    final embedModel = models.where((m) => m.isEmbeddingModel).firstOrNull;
    if (embedModel != null && !await isDownloaded(embedModel.fileName)) {
      // Try bundled copy first (instant), then download as fallback
      final copied = await _ensureBundledModelCopied(embedModel.fileName);
      if (!copied) {
        // Fire-and-forget download fallback — don't block splash
        download(embedModel).then((ok) {
          if (ok) debugPrint('[ModelManager] Embedding model downloaded');
        }).catchError((e) {
          debugPrint('[ModelManager] Embedding model download failed: $e');
        });
      } else {
        debugPrint('[ModelManager] Embedding model copied from bundle');
      }
    }

    return true;
  }

  /// Download remaining models in the background.
  /// Includes: free-tier projector (not required for splash) + all plus/pro models.
  /// Emits feature-ready notifications as each tier completes.
  /// Each download has a 10-minute timeout so one failure doesn't block the rest.
  Future<void> downloadRemainingInBackground() async {
    debugPrint('[ModelManager] Starting background download of remaining models...');

    // First, ensure the free-tier projector is copied/downloaded
    final freeProjector = models
        .where((m) => m.tier == AITier.free && m.isVisionProjector)
        .firstOrNull;
    if (freeProjector != null && !await isDownloaded(freeProjector.fileName)) {
      // Try bundled copy first, then download
      final copied = await _ensureBundledModelCopied(freeProjector.fileName);
      if (!copied) {
        try {
          await download(freeProjector, onProgress: (progress, status) {
            _downloadProgress[freeProjector.fileName] = progress;
            _downloadProgressStreamController.add(_downloadProgress);
          }).timeout(const Duration(minutes: 10));
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
          debugPrint('[ModelManager] Downloading ${model.fileName} (${model.sizeLabel})...');
          final ok = await download(model, onProgress: (progress, status) {
            _downloadProgress[model.fileName] = progress;
            _downloadProgressStreamController.add(_downloadProgress);
          }).timeout(const Duration(minutes: 10));
          if (!ok) {
            allReady = false;
            debugPrint('[ModelManager] ✗ ${model.fileName} download returned false');
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
    debugPrint('[ModelManager] Background download pass complete');
  }

  Future<void> downloadAllModelsInBackground() async {
    final tiers = [AITier.free, AITier.plus, AITier.pro];
    for (final tier in tiers) {
      final tierModels = modelsForTier(tier);
      for (final model in tierModels) {
        if (!await isDownloaded(model.fileName)) {
          download(model, onProgress: (progress, status) {
            _downloadProgress[model.fileName] = progress;
            _downloadProgressStreamController.add(_downloadProgress);
          });
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
  Future<bool> isDownloaded(String fileName) async {
    final path = await modelPath(fileName);
    final file = File(path);
    if (file.existsSync()) {
      // Validate that it's a proper model file, not corrupted
      final size = await file.length();
      if (size > 1024 * 1024) { // At least 1MB for valid model
        return true;
      }
    }
    return false;
  }
  
  /// Resolve the physical file path of a Flutter asset on disk.
  /// Avoids rootBundle.load() which loads the entire file into RAM.
  /// Returns null if the asset can't be found on disk.
  ///
  /// iOS/macOS: assets live as real files inside the .app bundle
  /// Android:   assets are inside the APK zip — no physical path available
  Future<String?> _findBundledAssetPath(String assetKey) async {
    // Android assets are packed inside the APK; no filesystem path exists.
    if (Platform.isAndroid) return null;

    try {
      // On iOS & macOS, Flutter assets are physically on disk inside the app bundle.
      // iOS:   <bundle>/Frameworks/App.framework/flutter_assets/<assetKey>
      // macOS: <bundle>/Contents/Frameworks/App.framework/Resources/flutter_assets/<assetKey>
      final exe = Platform.resolvedExecutable;
      final bundleDir = File(exe).parent.path;

      final candidates = [
        // iOS
        '$bundleDir/Frameworks/App.framework/flutter_assets/$assetKey',
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

      debugPrint('[ModelManager] Bundled asset not found on disk for: $assetKey');
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
      final physicalPath = await _findBundledAssetPath('assets/models/$fileName');
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
      debugPrint('[ModelManager] _isValidBundledModel failed for $fileName: $e');
      return false;
    }
  }
  
  /// Copy a bundled model to local storage if it doesn't exist.
  /// Uses streamed file copy from the physical asset path to avoid OOM.
  /// Falls back to rootBundle.load() only on platforms where physical path
  /// is unavailable (should not happen on iOS/macOS/Android).
  Future<bool> _ensureBundledModelCopied(String fileName) async {
    final localPath = await modelPath(fileName);
    final localFile = File(localPath);

    // Already exists locally and is valid
    if (localFile.existsSync() && await localFile.length() > 1024 * 1024) {
      debugPrint('[ModelManager] $fileName already in local storage, skipping copy');
      return true;
    }

    try {
      // ── Strategy 1: Streamed file copy from physical asset path ──
      // This avoids loading the entire model (100MB+) into RAM.
      final physicalPath = await _findBundledAssetPath('assets/models/$fileName');

      if (physicalPath != null) {
        final sourceFile = File(physicalPath);
        final sourceSize = await sourceFile.length();

        if (sourceSize < 1024 * 1024) {
          debugPrint('[ModelManager] Bundled $fileName too small (${sourceSize} bytes)');
          return false;
        }

        // Validate GGUF header (read only 4 bytes)
        final raf = await sourceFile.open(mode: FileMode.read);
        final header = await raf.read(4);
        await raf.close();
        if (String.fromCharCodes(header) != 'GGUF') {
          debugPrint('[ModelManager] Bundled $fileName has invalid GGUF header');
          return false;
        }

        // Streamed copy — constant memory usage regardless of file size
        debugPrint('[ModelManager] Copying bundled $fileName (${(sourceSize / (1024 * 1024)).toStringAsFixed(1)} MB) via streamed file copy...');
        await localFile.create(recursive: true);
        await sourceFile.copy(localPath);
        debugPrint('[ModelManager] ✓ Copied $fileName to local storage');
        return true;
      }

      // ── Strategy 2: Android / other — chunked write via rootBundle ──
      // On Android, assets are inside the APK and can only be read via rootBundle.
      // We load into an ImmutableBuffer (native heap, not Dart heap) when possible,
      // then write in chunks to avoid Dart GC pressure.
      debugPrint('[ModelManager] Physical path not found, falling back to rootBundle for $fileName');

      final bundledData = await rootBundle.load('assets/models/$fileName');
      final bytes = bundledData.buffer.asUint8List();

      if (bytes.length < 1024 * 1024) {
        debugPrint('[ModelManager] Bundled $fileName too small (${bytes.length} bytes)');
        return false;
      }
      if (bytes.length > 4) {
        final header = String.fromCharCodes(bytes.take(4));
        if (header != 'GGUF') {
          debugPrint('[ModelManager] Bundled $fileName has invalid GGUF header');
          return false;
        }
      }

      // Write in 4MB chunks to reduce peak Dart heap pressure
      debugPrint('[ModelManager] Writing $fileName in chunks (${(bytes.length / (1024 * 1024)).toStringAsFixed(1)} MB)...');
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
      debugPrint('[ModelManager] ✓ Copied $fileName via rootBundle (${(bytes.length / (1024 * 1024)).toStringAsFixed(1)} MB)');
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
  Future<bool> get isFreeModelReady async {
    final freeModel = models.firstWhere((m) => m.tier == AITier.free);
    return isDownloaded(freeModel.fileName);
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
      if (expectedSizeBytes != null && size < (expectedSizeBytes * 0.95).round()) {
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
    
    final model = models.where((m) => m.tier == tier && !m.isVisionProjector && !m.isEmbeddingModel).firstOrNull;
    if (model == null) return null;

    final path = await modelPath(model.fileName);
    if (File(path).existsSync()) return path;
    return null;
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
    final model = models.where((m) => m.tier == tier && m.isVisionProjector).firstOrNull;
    if (model == null) return null;

    final path = await modelPath(model.fileName);
    if (File(path).existsSync()) return path;
    return null;
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
      return url.replaceFirst('://localhost:', '://10.0.2.2:');
    }
    return url;
  }

  /// Get the current model server URL.
  Future<String> getModelServerUrl() async {
    if (_cachedModelServerUrl != null) return _cachedModelServerUrl!;
    final prefs = await SharedPreferences.getInstance();
    _cachedModelServerUrl = prefs.getString(_prefKeyModelServer) ?? defaultModelServerUrl;
    return _cachedModelServerUrl!;
  }

  /// Set the model server URL.
  Future<void> setModelServerUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefKeyModelServer, url);
    _cachedModelServerUrl = url;
  }

  /// Download a model with progress tracking.
  /// Supports resume via Range header if partial file exists.
  Future<bool> download(
    ModelInfo model, {
    DownloadProgress? onProgress,
  }) async {
    // Ensure model server URL is loaded before downloading
    await getModelServerUrl();

    final path = await modelPath(model.fileName);
    final file = File(path);
    final partialFile = File('$path.partial');

    // Already downloaded — but validate it's actually a good GGUF file.
    // A previous interrupted download or CDN error could have left a bad file.
    if (await file.exists()) {
      if (await isValidLocalModel(path, expectedSizeBytes: model.sizeBytes)) {
        onProgress?.call(1.0, 'Ready');
        return true;
      }
      // Bad file at final path — delete and re-download
      debugPrint('[ModelManager] ${model.fileName} exists but failed validation — re-downloading');
      await file.delete().catchError((_) {});
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('downloading_${model.fileName}', true);

    try {
      _httpClient ??= HttpClient()
        ..connectionTimeout = const Duration(seconds: 15);
      final resolvedUrl = _resolveUrl(model.downloadUrl);
      debugPrint('[ModelManager] Downloading from: $resolvedUrl');
      final request = await _httpClient!.getUrl(Uri.parse(resolvedUrl));

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

      await for (final chunk in response) {
        sink.add(chunk);
        received += chunk.length;
        final progress = (received / total).clamp(0.0, 1.0);
        final mb = (received / (1024 * 1024)).toStringAsFixed(1);
        final totalMb = (total / (1024 * 1024)).toStringAsFixed(0);
        
        onProgress?.call(progress, '$mb / $totalMb MB');
      }

      await sink.flush();
      await sink.close();

      // Move partial → final
      await partialFile.rename(path);

      // Post-download GGUF validation — catch corrupt / truncated files
      // before signalling "ready" to the rest of the system.
      final valid = await isValidLocalModel(
        path,
        expectedSizeBytes: model.sizeBytes,
      );
      if (!valid) {
        debugPrint('[ModelManager] ✗ ${model.fileName} failed GGUF validation after download — deleting');
        await File(path).delete().catchError((_) {});
        await prefs.setBool('downloading_${model.fileName}', false);
        onProgress?.call(0.0, 'Error: downloaded file is not a valid GGUF model');
        return false;
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
  Future<bool> downloadTier(
    AITier tier, {
    DownloadProgress? onProgress,
  }) async {
    final tierModels = modelsForTier(tier);
    for (int i = 0; i < tierModels.length; i++) {
      final model = tierModels[i];
      final prefix = tierModels.length > 1
          ? '(${i + 1}/${tierModels.length}) '
          : '';
      final success = await download(
        model,
        onProgress: (p, s) => onProgress?.call(
          (i + p) / tierModels.length,
          '$prefix$s',
        ),
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
      
      _enterpriseModels.add(ModelInfo(
        fileName: fileName,
        displayName: 'Enterprise Model',
        downloadUrl: '', // Local file, no download URL
        sizeBytes: sizeBytes,
        tier: AITier.enterprise,
      ));
    } else if (!useLocal && modelUrl.isNotEmpty) {
      // Create a virtual ModelInfo for the remote API
      _enterpriseModels.add(ModelInfo(
        fileName: 'enterprise-api',
        displayName: 'Enterprise API',
        downloadUrl: modelUrl,
        sizeBytes: 0, // API, no local size
        tier: AITier.enterprise,
      ));
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
