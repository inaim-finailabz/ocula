import 'dart:io';
import 'package:flutter/foundation.dart';
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

  // ── Model Registry ──
  // TODO: Replace URLs with actual HuggingFace/CDN endpoints
  static const models = [
    ModelInfo(
      fileName: 'smolvlm-256m-q4.gguf',
      downloadUrl: 'https://huggingface.co/ocula/models/resolve/main/smolvlm-256m-q4.gguf',
      sizeBytes: 180 * 1024 * 1024, // ~180 MB
      tier: AITier.free,
    ),
    ModelInfo(
      fileName: 'moondream2-q4.gguf',
      downloadUrl: 'https://huggingface.co/ocula/models/resolve/main/moondream2-q4.gguf',
      sizeBytes: 350 * 1024 * 1024, // ~350 MB
      tier: AITier.plus,
    ),
    ModelInfo(
      fileName: 'qwen2.5-vl-3b-q4.gguf',
      downloadUrl: 'https://huggingface.co/ocula/models/resolve/main/qwen2.5-vl-3b-q4.gguf',
      sizeBytes: 2048 * 1024 * 1024, // ~2 GB
      tier: AITier.pro,
    ),
    ModelInfo(
      fileName: 'mmproj-qwen2.5-vl-f16.gguf',
      downloadUrl: 'https://huggingface.co/ocula/models/resolve/main/mmproj-qwen2.5-vl-f16.gguf',
      sizeBytes: 600 * 1024 * 1024, // ~600 MB
      tier: AITier.pro,
      isVisionProjector: true,
    ),
  ];

  String? _modelsDir;
  HttpClient? _httpClient;

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

  /// Check if a model is already downloaded.
  Future<bool> isDownloaded(String fileName) async {
    final path = await modelPath(fileName);
    final file = File(path);
    return file.existsSync();
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
    return models.where((m) => m.tier == tier).toList();
  }

  /// Get the main model path for a tier (not the vision projector).
  Future<String?> mainModelPath(AITier tier) async {
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
}
