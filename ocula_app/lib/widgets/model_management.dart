import 'dart:async';
import 'package:flutter/material.dart';
import '../services/ai_manager.dart';
import '../services/model_manager.dart';

class ModelManagement extends StatefulWidget {
  const ModelManagement({super.key});

  @override
  State<ModelManagement> createState() => _ModelManagementState();
}

class _ModelManagementState extends State<ModelManagement> {
  final OculaModelManager _modelManager = OculaModelManager();
  Map<String, ModelStatus> _modelStatuses = {};
  final Map<String, double> _downloadProgress = {};
  final Map<String, String> _downloadStatusText = {};
  final Map<String, DateTime> _downloadStartTime = {};
  final Map<String, double> _downloadSpeedMBps = {};

  /// Verified models: true = GGUF header + size OK, false = failed, null = not checked
  Map<String, bool?> _verified = {};

  /// Best tier the device can run (null while loading)
  String? _recommendedTierLabel;
  AITier? _recommendedTier;

  /// Currently active AI tier (from AIManager stream)
  AITier? _activeTier;
  StreamSubscription<AITier>? _activeTierSub;

  /// Tier being activated right now (for loading state)
  AITier? _activatingTier;

  @override
  void initState() {
    super.initState();
    _activeTier = AIManager().activeTier;
    _activeTierSub = AIManager().activeTierStream.listen((tier) {
      if (mounted) setState(() => _activeTier = tier);
    });
    _loadModelStatuses();
    _loadRecommendedTier();
  }

  @override
  void dispose() {
    _activeTierSub?.cancel();
    super.dispose();
  }

  Future<void> _loadRecommendedTier() async {
    final ai = AIManager();
    AITier best = AITier.free;
    for (final tier in [AITier.pro, AITier.plus, AITier.free]) {
      if (await ai.canDeviceRunTier(tier)) {
        best = tier;
        break;
      }
    }
    if (mounted) {
      setState(() {
        _recommendedTier = best;
        _recommendedTierLabel = OculaModelManager.featureLabel(best);
      });
    }
  }

  Future<void> _activateTier(AITier tier) async {
    if (_activatingTier != null) return;
    setState(() => _activatingTier = tier);
    try {
      await AIManager().switchEngine(tier);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              '${OculaModelManager.featureLabel(tier)} is now active',
            ),
            duration: const Duration(seconds: 2),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Could not activate ${OculaModelManager.featureLabel(tier)}: $e'),
            duration: const Duration(seconds: 3),
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _activatingTier = null);
    }
  }

  /// Returns true if all main (non-projector, non-embed) models for a tier
  /// are downloaded and verified.
  bool _isTierFullyReady(AITier tier) {
    final tierModels = _modelManager
        .modelsForTier(tier)
        .where((m) => !m.isVisionProjector && !m.isEmbeddingModel)
        .toList();
    if (tierModels.isEmpty) return false;
    return tierModels.every(
      (m) =>
          _modelStatuses[m.fileName] == ModelStatus.ready &&
          _verified[m.fileName] == true,
    );
  }

  Future<void> _loadModelStatuses() async {
    final statuses = <String, ModelStatus>{};
    final verified = <String, bool?>{};
    for (final model in OculaModelManager.models) {
      statuses[model.fileName] = await _modelManager.getStatus(model.fileName);
      if (statuses[model.fileName] == ModelStatus.ready) {
        final path = await _modelManager.modelPath(model.fileName);
        verified[model.fileName] = await _modelManager.isValidLocalModel(
          path,
          expectedSizeBytes: model.sizeBytes,
        );
      }
    }
    if (mounted) {
      setState(() {
        _modelStatuses = statuses;
        _verified = verified;
      });
    }
  }

  void _downloadModel(ModelInfo model) {
    final startTime = DateTime.now();
    final startProgress = _downloadProgress[model.fileName] ?? 0.0;
    setState(() {
      _modelStatuses[model.fileName] = ModelStatus.downloading;
      _downloadStartTime[model.fileName] = startTime;
    });

    _modelManager.download(
      model,
      onProgress: (progress, status) {
        if (mounted) {
          setState(() {
            _downloadProgress[model.fileName] = progress;
            _downloadStatusText[model.fileName] = status;

            // Calculate download speed for ETA
            final elapsed =
                DateTime.now().difference(startTime).inSeconds.toDouble();
            if (elapsed > 2) {
              final progressDone = progress - startProgress;
              if (progressDone > 0) {
                final speedMBps = (progressDone * model.sizeMB) / elapsed;
                _downloadSpeedMBps[model.fileName] = speedMBps;
              }
            }

            if (progress >= 1.0) {
              _downloadStartTime.remove(model.fileName);
              _downloadSpeedMBps.remove(model.fileName);
              _downloadStatusText.remove(model.fileName);
              _loadModelStatuses();
            }
          });
        }
      },
    );
  }

  Future<void> _forceRedownload(ModelInfo model) async {
    await _modelManager.deleteModel(model.fileName);
    if (mounted) {
      setState(() {
        _modelStatuses[model.fileName] = ModelStatus.notDownloaded;
        _verified.remove(model.fileName);
        _downloadProgress.remove(model.fileName);
        _downloadStatusText.remove(model.fileName);
        _downloadSpeedMBps.remove(model.fileName);
      });
      _downloadModel(model);
    }
  }

  void _deleteModel(ModelInfo model) {
    _modelManager.deleteModel(model.fileName).then((_) {
      _loadModelStatuses();
    });
  }

  /// Format "X.X MB / Y MB · ~Z left" for the download subtitle.
  String _progressSubtitle(ModelInfo model, double progress) {
    final receivedMB = progress * model.sizeMB;
    final received = receivedMB >= 1024
        ? '${(receivedMB / 1024).toStringAsFixed(2)} GB'
        : '${receivedMB.toStringAsFixed(1)} MB';
    final total = model.sizeLabel;
    final base = '$received / $total';

    final speedMBps = _downloadSpeedMBps[model.fileName];
    if (speedMBps != null && speedMBps > 0 && progress < 1.0) {
      final remainingMB = (1.0 - progress) * model.sizeMB;
      final remainingSec = remainingMB / speedMBps;
      final eta = remainingSec < 60
          ? '~${remainingSec.round()}s left'
          : remainingSec < 3600
          ? '~${(remainingSec / 60).floor()}m ${(remainingSec % 60).round()}s left'
          : '~${(remainingSec / 3600).toStringAsFixed(1)}h left';
      return '$base · $eta';
    }
    return base;
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'AI Models',
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
        const SizedBox(height: 10),

        // Recommended tier for this device
        if (_recommendedTierLabel != null)
          Container(
            margin: const EdgeInsets.only(bottom: 14),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
            decoration: BoxDecoration(
              color: colors.primaryContainer.withAlpha(70),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: colors.primary.withAlpha(50)),
            ),
            child: Row(
              children: [
                Icon(Icons.memory_outlined, size: 16, color: colors.primary),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    _recommendedTier != null &&
                            _isTierFullyReady(_recommendedTier!) &&
                            _activeTier != _recommendedTier
                        ? '$_recommendedTierLabel is ready — tap Activate to switch'
                        : 'Recommended for your device: $_recommendedTierLabel',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                      color: colors.primary,
                    ),
                  ),
                ),
              ],
            ),
          ),

        for (final tier in AITier.values)
          _buildTierSection(tier, colors),
      ],
    );
  }

  Widget _buildTierSection(AITier tier, ColorScheme colors) {
    final tierModels = _modelManager.modelsForTier(tier);
    if (tierModels.isEmpty) return const SizedBox.shrink();

    final isActive = _activeTier == tier;
    final isFree = tier == AITier.free;
    final fullyReady = _isTierFullyReady(tier);
    final isActivating = _activatingTier == tier;

    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                OculaModelManager.featureLabel(tier),
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                  color: colors.primary,
                ),
              ),
              const SizedBox(width: 8),
              // Active / Default chip
              if (isActive || isFree)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: isActive
                        ? Colors.green.withAlpha(40)
                        : colors.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                      color: isActive
                          ? Colors.green
                          : colors.outline.withAlpha(80),
                    ),
                  ),
                  child: Text(
                    isFree && !isActive ? 'Default' : 'Active',
                    style: TextStyle(
                      fontSize: 11,
                      color: isActive ? Colors.green : colors.onSurface.withAlpha(160),
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 8),
          for (final model in tierModels)
            _buildModelTile(model, colors),
          // Activate button (non-free tiers that are fully ready but not active)
          if (!isFree && fullyReady && !isActive) ...[
            const SizedBox(height: 4),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: isActivating ? null : () => _activateTier(tier),
                child: isActivating
                    ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('Activate'),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildModelTile(ModelInfo model, ColorScheme colors) {
    final status = _modelStatuses[model.fileName] ?? ModelStatus.notDownloaded;
    final progress = _downloadProgress[model.fileName];
    final isVerified = _verified[model.fileName];
    final isDownloading = status == ModelStatus.downloading;

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 4),
      color: colors.surfaceContainerHighest,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 10, 8, 10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                // Model name + status icon
                Expanded(
                  child: Row(
                    children: [
                      Flexible(
                        child: Text(
                          model.displayName,
                          style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                      if (status == ModelStatus.ready && isVerified == true)
                        Padding(
                          padding: const EdgeInsets.only(left: 6),
                          child: Icon(
                            Icons.verified,
                            size: 15,
                            color: Colors.greenAccent,
                          ),
                        ),
                      if (status == ModelStatus.ready && isVerified == false)
                        Padding(
                          padding: const EdgeInsets.only(left: 6),
                          child: Icon(
                            Icons.error_outline,
                            size: 15,
                            color: colors.error,
                          ),
                        ),
                    ],
                  ),
                ),

                // Action buttons / spinner
                _buildActionWidget(model, status, progress, isVerified, colors),
              ],
            ),

            // Subtitle row
            Padding(
              padding: const EdgeInsets.only(top: 2),
              child: Row(
                children: [
                  Text(
                    model.sizeLabel,
                    style: TextStyle(
                      fontSize: 12,
                      color: colors.onSurface.withAlpha(140),
                    ),
                  ),
                  if (status == ModelStatus.ready && isVerified == true)
                    Text(
                      '  ·  verified',
                      style: TextStyle(
                        fontSize: 11,
                        color: Colors.greenAccent,
                      ),
                    ),
                  if (status == ModelStatus.ready && isVerified == false)
                    Text(
                      '  ·  corrupt',
                      style: TextStyle(fontSize: 11, color: colors.error),
                    ),
                  if (isDownloading && progress != null)
                    Expanded(
                      child: Text(
                        '  ·  ${_progressSubtitle(model, progress)}',
                        style: TextStyle(
                          fontSize: 11,
                          color: colors.primary,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                ],
              ),
            ),

            // LinearProgressIndicator during download
            if (isDownloading) ...[
              const SizedBox(height: 8),
              ClipRRect(
                borderRadius: BorderRadius.circular(2),
                child: LinearProgressIndicator(
                  value: progress,
                  minHeight: 3,
                  backgroundColor: colors.primary.withAlpha(25),
                  valueColor: AlwaysStoppedAnimation<Color>(colors.primary),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildActionWidget(
    ModelInfo model,
    ModelStatus status,
    double? progress,
    bool? isVerified,
    ColorScheme colors,
  ) {
    switch (status) {
      case ModelStatus.ready:
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            IconButton(
              icon: const Icon(Icons.refresh, size: 20),
              tooltip: 'Force re-download',
              onPressed: () => _showRedownloadDialog(model),
            ),
            IconButton(
              icon: const Icon(Icons.delete, size: 20),
              tooltip: 'Delete',
              onPressed: () => _deleteModel(model),
            ),
          ],
        );
      case ModelStatus.downloading:
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          child: SizedBox(
            width: 20,
            height: 20,
            child: CircularProgressIndicator(
              value: progress,
              strokeWidth: 2,
              color: colors.primary,
            ),
          ),
        );
      case ModelStatus.notDownloaded:
      case ModelStatus.error:
        return IconButton(
          icon: const Icon(Icons.download),
          onPressed: () => _downloadModel(model),
        );
    }
  }

  void _showRedownloadDialog(ModelInfo model) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Force Re-download'),
        content: Text(
          'Delete and re-download ${model.displayName} (${model.sizeLabel})?\n\n'
          'This will verify the file with GGUF header and size check after download.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              Navigator.pop(ctx);
              _forceRedownload(model);
            },
            child: const Text('Re-download'),
          ),
        ],
      ),
    );
  }
}
