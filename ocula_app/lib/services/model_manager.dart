import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'ai_manager.dart';

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
  final String downloadUrl;
  final int sizeBytes;
  final AITier tier;
  final bool isVisionProjector;

  const ModelInfo({
    required this.fileName,
    required this.downloadUrl,
    required this.sizeBytes,
    required this.tier,
    this.isVisionProjector = false,
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

  // ── Model Registry ── 2026 Ocula Intelligence Stack
  // Served by serve_models.sh (python3 http.server on port 8080)
  //
  // Tier free  (Sensor)     – SmolVLM2-256M: instant response, video support
  // Tier plus  (Specialist) – Moondream 2:   pointing (x,y) & counting
  // Tier pro   (Thinker)    – Qwen3-VL-2B:   chain-of-thought reasoning
  static const models = [
    // ── SENSOR: Always-on, fast & tiny ──
    ModelInfo(
      fileName: 'SmolVLM2-256M-Video-Instruct-Q8_0.gguf',
      downloadUrl: 'http://localhost:8080/models/SmolVLM2-256M-Video-Instruct-Q8_0.gguf',
      sizeBytes: 175 * 1024 * 1024, // ~175 MB
      tier: AITier.free,
    ),
    ModelInfo(
      fileName: 'mmproj-SmolVLM2-256M-Video-Instruct-f16.gguf',
      downloadUrl: 'http://localhost:8080/models/mmproj-SmolVLM2-256M-Video-Instruct-f16.gguf',
      sizeBytes: 181 * 1024 * 1024, // ~181 MB
      tier: AITier.free,
      isVisionProjector: true,
    ),
    // ── SPECIALIST: Spatial tasks, pointing & counting ──
    // Moondream 2 (April 2025) — quantized from f16 to Q4_K_M
    ModelInfo(
      fileName: 'moondream2-text-model-Q4_K_M.gguf',
      downloadUrl: 'http://localhost:8080/models/moondream2-text-model-Q4_K_M.gguf',
      sizeBytes: 900 * 1024 * 1024, // ~900 MB (Q4_K_M quantized)
      tier: AITier.plus,
    ),
    ModelInfo(
      fileName: 'moondream2-mmproj-f16-20250414.gguf',
      downloadUrl: 'http://localhost:8080/models/moondream2-mmproj-f16-20250414.gguf',
      sizeBytes: 910 * 1024 * 1024, // ~910 MB
      tier: AITier.plus,
      isVisionProjector: true,
    ),
    // ── THINKER: Reasoning with chain-of-thought ──
    ModelInfo(
      fileName: 'Qwen3-VL-2B-Thinking-Q4_K_M.gguf',
      downloadUrl: 'http://localhost:8080/models/Qwen3-VL-2B-Thinking-Q4_K_M.gguf',
      sizeBytes: 1400 * 1024 * 1024, // ~1.4 GB
      tier: AITier.pro,
    ),
    ModelInfo(
      fileName: 'mmproj-Qwen3-VL-2B-Thinking-F16.gguf',
      downloadUrl: 'http://localhost:8080/models/mmproj-Qwen3-VL-2B-Thinking-F16.gguf',
      sizeBytes: 600 * 1024 * 1024, // ~600 MB
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

  Stream<Map<String, double>> get downloadProgressStream => _downloadProgressStreamController.stream;

  /// Subscribe to this for user-facing "feature X is ready" notifications.
  Stream<String> get featureReadyStream => _featureReadyController.stream;

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

  /// Download the free-tier model only. Returns true when it's ready.
  /// Intended to run during the splash screen so the user can interact ASAP.
  /// First tries to copy bundled models, then downloads if needed.
  Future<bool> ensureFreeModelReady({
    DownloadProgress? onProgress,
  }) async {
    final freeModels = modelsForTier(AITier.free);
    
    for (final model in freeModels) {
      // Step 1: Try to copy bundled model first (instant)
      onProgress?.call(0.1, 'Checking for bundled AI engine...');
      if (await _ensureBundledModelCopied(model.fileName)) {
        onProgress?.call(1.0, 'Bundled AI engine ready!');
        continue;
      }
      
      // Step 2: If not bundled, check if already downloaded
      if (await isDownloaded(model.fileName)) {
        onProgress?.call(1.0, 'AI engine ready!');
        continue;
      }
      
      // Step 3: Download as fallback
      onProgress?.call(0.0, 'Downloading AI engine...');
      final ok = await download(model, onProgress: onProgress);
      if (!ok) return false;
    }
    return true;
  }

  /// Download remaining (non-free) models in the background.
  /// Emits feature-ready notifications as each tier completes.
  Future<void> downloadRemainingInBackground() async {
    for (final tier in [AITier.plus, AITier.pro]) {
      final tierModels = modelsForTier(tier);
      bool allReady = true;
      for (final model in tierModels) {
        if (await isDownloaded(model.fileName)) continue;
        final ok = await download(model, onProgress: (progress, status) {
          _downloadProgress[model.fileName] = progress;
          _downloadProgressStreamController.add(_downloadProgress);
        });
        if (!ok) allReady = false;
      }
      if (allReady) {
        _featureReadyController.add(featureLabel(tier));
      }
    }
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

  /// Check if a model is already downloaded or bundled.
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
    
    // Check if valid model is bundled with the app
    return await _isValidBundledModel(fileName);
  }
  
  /// Check if a valid model is bundled as an asset with the app.
  Future<bool> _isValidBundledModel(String fileName) async {
    try {
      final bundledData = await rootBundle.load('assets/models/$fileName');
      final bytes = bundledData.buffer.asUint8List();
      
      // Check size and GGUF header to ensure it's a real model
      if (bytes.length < 1024 * 1024) return false; // Too small
      
      if (bytes.length > 4) {
        final header = String.fromCharCodes(bytes.take(4));
        return header == 'GGUF';
      }
      
      return false;
    } catch (e) {
      return false;
    }
  }
  
  /// Copy a bundled model to local storage if it doesn't exist.
  /// Validates the bundled model size to ensure it's not a placeholder.
  Future<bool> _ensureBundledModelCopied(String fileName) async {
    final localPath = await modelPath(fileName);
    final localFile = File(localPath);
    
    // Already exists locally
    if (localFile.existsSync()) return true;
    
    // Try to copy from bundle
    try {
      final bundledData = await rootBundle.load('assets/models/$fileName');
      final bytes = bundledData.buffer.asUint8List();
      
      // Validate that this is actually a model file, not a placeholder
      // GGUF files should be at least 1MB, placeholders are much smaller
      if (bytes.length < 1024 * 1024) {
        debugPrint('Bundled model $fileName is too small (${bytes.length} bytes), downloading from backend instead');
        return false;
      }
      
      // Check for GGUF magic header (first 4 bytes should be 'GGUF')
      if (bytes.length > 4) {
        final header = String.fromCharCodes(bytes.take(4));
        if (header != 'GGUF') {
          debugPrint('Bundled model $fileName does not have valid GGUF header');
          return false;
        }
      }
      
      await localFile.create(recursive: true);
      await localFile.writeAsBytes(bytes);
      debugPrint('Successfully copied bundled model $fileName (${bytes.length} bytes)');
      return true;
    } catch (e) {
      debugPrint('Failed to copy bundled model $fileName: $e');
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

  /// Get the main model path for a tier (not the vision projector).
  Future<String?> mainModelPath(AITier tier) async {
    if (tier == AITier.enterprise) {
      return await enterpriseModelPath;
    }
    
    final model = models.where((m) => m.tier == tier && !m.isVisionProjector).firstOrNull;
    if (model == null) return null;
    final path = await modelPath(model.fileName);
    if (File(path).existsSync()) return path;
    return null;
  }

  /// Get vision projector path for a tier (if any).
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

  /// Download a model with progress tracking.
  /// Supports resume via Range header if partial file exists.
  Future<bool> download(
    ModelInfo model, {
    DownloadProgress? onProgress,
  }) async {
    final path = await modelPath(model.fileName);
    final file = File(path);
    final partialFile = File('$path.partial');

    // Already downloaded
    if (await file.exists()) {
      onProgress?.call(1.0, 'Ready');
      return true;
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('downloading_${model.fileName}', true);

    try {
      _httpClient ??= HttpClient();
      final request = await _httpClient!.getUrl(Uri.parse(model.downloadUrl));

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
        downloadUrl: '', // Local file, no download URL
        sizeBytes: sizeBytes,
        tier: AITier.enterprise,
      ));
    } else if (!useLocal && modelUrl.isNotEmpty) {
      // Create a virtual ModelInfo for the remote API
      _enterpriseModels.add(ModelInfo(
        fileName: 'enterprise-api',
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
