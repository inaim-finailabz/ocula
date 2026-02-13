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

  @override
  void initState() {
    super.initState();
    _loadModelStatuses();
  }

  Future<void> _loadModelStatuses() async {
    final statuses = <String, ModelStatus>{};
    for (final model in OculaModelManager.models) {
      statuses[model.fileName] = await _modelManager.getStatus(model.fileName);
    }
    if (mounted) {
      setState(() {
        _modelStatuses = statuses;
      });
    }
  }

  void _downloadModel(ModelInfo model) {
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
    _loadModelStatuses();
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

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 4),
      color: colors.surfaceContainerHighest,
      child: ListTile(
        title: Text(model.displayName),
        subtitle: Text(model.sizeLabel),
        trailing: _buildActionWidget(model, status, progress),
      ),
    );
  }

  Widget _buildActionWidget(ModelInfo model, ModelStatus status, double? progress) {
    switch (status) {
      case ModelStatus.ready:
        return IconButton(
          icon: const Icon(Icons.delete),
          onPressed: () => _deleteModel(model),
        );
      case ModelStatus.downloading:
        return CircularProgressIndicator(value: progress);
      case ModelStatus.notDownloaded:
      case ModelStatus.error:
        return IconButton(
          icon: const Icon(Icons.download),
          onPressed: () => _downloadModel(model),
        );
    }
  }
}
