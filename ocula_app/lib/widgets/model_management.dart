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
  Map<String, double> _downloadProgress = {};
  /// Verified models: true = GGUF header + size OK, false = failed, null = not checked
  Map<String, bool?> _verified = {};

  @override
  void initState() {
    super.initState();
    _loadModelStatuses();
  }

  Future<void> _loadModelStatuses() async {
    final statuses = <String, ModelStatus>{};
    final verified = <String, bool?>{};
    for (final model in OculaModelManager.models) {
      statuses[model.fileName] = await _modelManager.getStatus(model.fileName);
      // Verify downloaded models (GGUF header + size)
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
    setState(() {
      _modelStatuses[model.fileName] = ModelStatus.downloading;
    });
    _modelManager.download(
      model,
      onProgress: (progress, status) {
        if (mounted) {
          setState(() {
            _downloadProgress[model.fileName] = progress;
            if (progress == 1.0) {
              _loadModelStatuses();
            }
          });
        }
      },
    );
  }

  Future<void> _forceRedownload(ModelInfo model) async {
    // Delete existing file first, then re-download
    await _modelManager.deleteModel(model.fileName);
    if (mounted) {
      setState(() {
        _modelStatuses[model.fileName] = ModelStatus.notDownloaded;
        _verified.remove(model.fileName);
        _downloadProgress.remove(model.fileName);
      });
      _downloadModel(model);
    }
  }

  void _deleteModel(ModelInfo model) {
    _modelManager.deleteModel(model.fileName).then((_) {
      _loadModelStatuses();
    });
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
        const SizedBox(height: 8),
        for (final tier in AITier.values)
          _buildTierSection(tier, colors),
      ],
    );
  }

  Widget _buildTierSection(AITier tier, ColorScheme colors) {
    final tierModels = _modelManager.modelsForTier(tier);
    if (tierModels.isEmpty) return const SizedBox.shrink();
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            OculaModelManager.featureLabel(tier),
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: colors.primary,
            ),
          ),
          const SizedBox(height: 8),
          for (final model in tierModels)
            _buildModelTile(model, colors),
        ],
      ),
    );
  }

  Widget _buildModelTile(ModelInfo model, ColorScheme colors) {
    final status = _modelStatuses[model.fileName] ?? ModelStatus.notDownloaded;
    final progress = _downloadProgress[model.fileName];
    final isVerified = _verified[model.fileName];

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 4),
      color: colors.surfaceContainerHighest,
      child: ListTile(
        title: Row(
          children: [
            Expanded(child: Text(model.displayName)),
            if (status == ModelStatus.ready && isVerified == true)
              Icon(Icons.verified, size: 16, color: Colors.greenAccent),
            if (status == ModelStatus.ready && isVerified == false)
              Icon(Icons.error_outline, size: 16, color: colors.error),
          ],
        ),
        subtitle: Row(
          children: [
            Text(model.sizeLabel),
            if (status == ModelStatus.ready && isVerified == true)
              Text('  ·  verified',
                style: TextStyle(fontSize: 11, color: Colors.greenAccent)),
            if (status == ModelStatus.ready && isVerified == false)
              Text('  ·  corrupt',
                style: TextStyle(fontSize: 11, color: colors.error)),
            if (status == ModelStatus.downloading && progress != null)
              Text('  ·  ${(progress * 100).toStringAsFixed(0)}%',
                style: TextStyle(fontSize: 11, color: colors.primary)),
          ],
        ),
        trailing: _buildActionWidget(model, status, progress, isVerified),
      ),
    );
  }

  Widget _buildActionWidget(
    ModelInfo model,
    ModelStatus status,
    double? progress,
    bool? isVerified,
  ) {
    switch (status) {
      case ModelStatus.ready:
        return Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Force re-download (refresh) button
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
        return SizedBox(
          width: 24,
          height: 24,
          child: CircularProgressIndicator(
            value: progress,
            strokeWidth: 2.5,
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
