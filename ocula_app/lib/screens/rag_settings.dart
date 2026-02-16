import 'dart:async';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import '../services/ai_manager.dart';
import '../services/model_manager.dart';
import '../services/rag_config.dart';

/// Settings card for tuning RAG search and generation parameters.
class RagSettings extends StatefulWidget {
  const RagSettings({super.key});

  @override
  State<RagSettings> createState() => _RagSettingsState();
}

class _RagSettingsState extends State<RagSettings> {
  final RagConfig _config = RagConfig();
  final OculaModelManager _models = OculaModelManager();
  bool _loading = true;

  late double _vectorWeight;
  late int _topK;
  late double _minScore;
  late int _contextBudget;
  late int _maxTokens;
  late int _chunkSize;
  late String _modelOverride;

  /// Download status per tier key ('free', 'plus', 'pro').
  final Map<String, bool> _downloaded = {};

  @override
  void initState() {
    super.initState();
    _load();
  }

  /// Validate a tier the same way switchEngine does.
  /// For multimodal tiers, optionally require the vision projector too.
  Future<bool> _isTierReady(AITier tier, {bool requireVision = false}) async {
    final path = await _models.mainModelPath(tier);
    if (path == null) return false;

    // Look up expected file size from the model registry
    final modelInfo = OculaModelManager.models
        .where(
          (m) => m.tier == tier && !m.isVisionProjector && !m.isEmbeddingModel,
        )
        .firstOrNull;
    final registryName = modelInfo?.fileName;
    final actualName = p.basename(path);
    final expectedSize = (registryName != null && registryName == actualName)
        ? modelInfo?.sizeBytes
        : null;

    // Full GGUF validation with expected size check (catches truncated downloads)
    final valid = await _models.isValidLocalModel(
      path,
      expectedSizeBytes: expectedSize,
    );
    if (!valid) return false;

    if (requireVision) {
      final projPath = await _models.visionProjectorPath(tier);
      if (projPath == null) return false;
      final projValid = await _models.isValidLocalModel(projPath);
      if (!projValid) return false;
    }

    // RAM gate — same check switchEngine uses before loading.
    return await AIManager().canDeviceRunTier(tier);
  }

  Future<void> _load() async {
    await _config.load();

    // Check if each tier is fully ready (downloaded + valid size + enough RAM)
    final freeOk = await _isTierReady(AITier.free);
    final plusOk = await _isTierReady(AITier.plus, requireVision: true);
    final proOk = await _isTierReady(AITier.pro, requireVision: true);

    if (!mounted) return;
    setState(() {
      _vectorWeight = _config.vectorWeight;
      _topK = _config.topK;
      _minScore = _config.minScore;
      _contextBudget = _config.contextBudgetChars;
      _maxTokens = _config.maxResponseTokens;
      _chunkSize = _config.chunkSize;
      _modelOverride = _config.modelOverride;
      _downloaded['free'] = freeOk;
      _downloaded['plus'] = plusOk;
      _downloaded['pro'] = proOk;
      _loading = false;
    });
  }

  Future<void> _resetDefaults() async {
    await _config.resetDefaults();
    await _load();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('RAG settings reset to defaults'),
          backgroundColor: Colors.green,
        ),
      );
    }
  }

  Future<void> _selectModel(String value) async {
    setState(() => _modelOverride = value);
    _config.setModelOverride(value);

    // 'auto' keeps intent-based routing.
    if (value == 'auto') return;

    final tier = switch (value) {
      'free' => AITier.free,
      'plus' => AITier.plus,
      'pro' => AITier.pro,
      _ => null,
    };
    if (tier == null) return;

    final ai = AIManager();
    final canRun = await ai.canDeviceRunTier(tier);
    if (!canRun) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            '${_tierDisplayName(value)} needs more device memory. '
            'Keeping current model.',
          ),
          backgroundColor: Colors.orange,
        ),
      );
      final fallbackKey = _tierKey(ai.activeTier ?? AITier.free);
      if (fallbackKey != null) {
        setState(() => _modelOverride = fallbackKey);
        _config.setModelOverride(fallbackKey);
      }
      return;
    }

    // If already ready, switch now. Otherwise download in background and
    // auto-switch when ready.
    final requireVision = tier == AITier.plus || tier == AITier.pro;
    if (await _isTierReady(tier, requireVision: requireVision)) {
      await _switchToTierWithFeedback(tier, value);
      return;
    }

    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          '${_tierDisplayName(value)} is downloading in background. '
          'Ocula will switch automatically when ready.',
        ),
        backgroundColor: Colors.orange,
      ),
    );
    unawaited(_downloadTierAndAutoSwitch(tier, value));
  }

  Future<void> _downloadTierAndAutoSwitch(AITier tier, String tierKey) async {
    final ok = await _models.downloadTier(tier);
    if (!mounted) return;

    if (!ok) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Download failed for ${_tierDisplayName(tierKey)}.'),
          backgroundColor: Colors.red,
        ),
      );
      await _load();
      return;
    }

    final requireVision = tier == AITier.plus || tier == AITier.pro;
    final ready = await _isTierReady(tier, requireVision: requireVision);
    if (!mounted) return;
    setState(() => _downloaded[tierKey] = ready);
    if (!ready) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            '${_tierDisplayName(tierKey)} downloaded, but device cannot run it right now.',
          ),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    // User may have changed selection while downloading.
    if (_modelOverride != tierKey) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('${_tierDisplayName(tierKey)} is ready to use.'),
          backgroundColor: Colors.green,
        ),
      );
      return;
    }

    await _switchToTierWithFeedback(tier, tierKey, downloadedNow: true);
  }

  Future<void> _switchToTierWithFeedback(
    AITier tier,
    String tierKey, {
    bool downloadedNow = false,
  }) async {
    try {
      final ai = AIManager();
      await ai.switchEngine(tier);
      final active = ai.activeTier;
      if (!mounted) return;
      if (active == tier) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              downloadedNow
                  ? '${_tierDisplayName(tierKey)} is ready and now active.'
                  : 'Switched to ${_tierDisplayName(tierKey)}',
            ),
            backgroundColor: Colors.green,
          ),
        );
      } else {
        final fallbackKey = _tierKey(active);
        if (fallbackKey != null) {
          setState(() => _modelOverride = fallbackKey);
          _config.setModelOverride(fallbackKey);
        }
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Could not switch to ${_tierDisplayName(tierKey)}. '
              'Still using ${_tierDisplayName(fallbackKey ?? "free")}.',
            ),
            backgroundColor: Colors.orange,
          ),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to switch model: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  String _tierDisplayName(String key) {
    switch (key) {
      case 'free':
        return 'Sensor';
      case 'plus':
        return 'Specialist';
      case 'pro':
        return 'Thinker';
      default:
        return 'Auto';
    }
  }

  String? _tierKey(AITier? tier) {
    if (tier == null) return null;
    switch (tier) {
      case AITier.free:
        return 'free';
      case AITier.plus:
        return 'plus';
      case AITier.pro:
        return 'pro';
      case AITier.enterprise:
        return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) return const SizedBox.shrink();

    final colors = Theme.of(context).colorScheme;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.tune, color: colors.primary),
                const SizedBox(width: 10),
                Text(
                  'Search & RAG Tuning',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              'Fine-tune how Ocula searches your data and generates responses.',
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(color: Colors.grey[600]),
            ),
            const SizedBox(height: 16),

            // ── Model Override ──
            Text(
              'AI Model',
              style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
            ),
            const SizedBox(height: 6),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _modelChip('auto', 'Auto', colors, alwaysAvailable: true),
                _modelChip('free', 'Sensor', colors),
                _modelChip('plus', 'Specialist', colors),
                _modelChip('pro', 'Thinker', colors),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              _modelOverride == 'auto'
                  ? 'Ocula picks the best model based on your query.'
                  : 'All queries will use ${_tierDisplayName(_modelOverride)}.',
              style: TextStyle(
                fontSize: 11,
                color: colors.onSurface.withAlpha(100),
              ),
            ),
            const SizedBox(height: 16),

            // ── Vector Weight ──
            _sliderTile(
              label: 'Semantic vs Keyword',
              value: _vectorWeight,
              min: 0.0,
              max: 1.0,
              divisions: 20,
              valueLabel: '${(_vectorWeight * 100).round()}% semantic',
              hint:
                  'Higher = more meaning-based search. Lower = more keyword matching.',
              onChanged: (v) {
                setState(() => _vectorWeight = v);
                _config.setVectorWeight(v);
              },
            ),

            // ── Top-K ──
            _sliderTile(
              label: 'Results Count',
              value: _topK.toDouble(),
              min: 1,
              max: 20,
              divisions: 19,
              valueLabel: '$_topK results',
              hint: 'How many search results are retrieved per query.',
              onChanged: (v) {
                setState(() => _topK = v.round());
                _config.setTopK(v.round());
              },
            ),

            // ── Min Score ──
            _sliderTile(
              label: 'Quality Threshold',
              value: _minScore,
              min: 0.0,
              max: 0.5,
              divisions: 10,
              valueLabel: _minScore.toStringAsFixed(2),
              hint:
                  'Minimum relevance score. Higher = stricter, fewer results.',
              onChanged: (v) {
                setState(() => _minScore = v);
                _config.setMinScore(v);
              },
            ),

            // ── Context Budget ──
            _sliderTile(
              label: 'Context Budget',
              value: _contextBudget.toDouble(),
              min: 200,
              max: 4000,
              divisions: 19,
              valueLabel: '$_contextBudget chars',
              hint: 'How much retrieved text is sent to the model.',
              onChanged: (v) {
                setState(() => _contextBudget = v.round());
                _config.setContextBudgetChars(v.round());
              },
            ),

            // ── Max Tokens ──
            _sliderTile(
              label: 'Max Response Length',
              value: _maxTokens.toDouble(),
              min: 50,
              max: 1024,
              divisions: 19,
              valueLabel: '$_maxTokens tokens',
              hint:
                  'Maximum generation length. Larger models use the full budget.',
              onChanged: (v) {
                setState(() => _maxTokens = v.round());
                _config.setMaxResponseTokens(v.round());
              },
            ),

            // ── Chunk Size ──
            _sliderTile(
              label: 'Chunk Size',
              value: _chunkSize.toDouble(),
              min: 200,
              max: 2000,
              divisions: 18,
              valueLabel: '$_chunkSize chars',
              hint: 'Text chunk size for indexing. Affects new indexes only.',
              onChanged: (v) {
                setState(() => _chunkSize = v.round());
                _config.setChunkSize(v.round());
              },
            ),

            const SizedBox(height: 8),
            Center(
              child: TextButton.icon(
                onPressed: _resetDefaults,
                icon: const Icon(Icons.restore, size: 18),
                label: const Text('Reset to Defaults'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _modelChip(
    String value,
    String label,
    ColorScheme colors, {
    bool alwaysAvailable = false,
  }) {
    final isSelected = _modelOverride == value;
    final isAvailable = alwaysAvailable || (_downloaded[value] == true);

    return ChoiceChip(
      label: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(label),
          if (!alwaysAvailable) ...[
            const SizedBox(width: 4),
            Icon(
              isAvailable ? Icons.check_circle : Icons.cloud_download_outlined,
              size: 14,
              color: isSelected
                  ? colors.onSecondaryContainer
                  : isAvailable
                  ? Colors.greenAccent
                  : colors.onSurface.withAlpha(80),
            ),
          ],
        ],
      ),
      selected: isSelected,
      onSelected: (_) => _selectModel(value),
    );
  }

  Widget _sliderTile({
    required String label,
    required double value,
    required double min,
    required double max,
    required int divisions,
    required String valueLabel,
    required String hint,
    required ValueChanged<double> onChanged,
  }) {
    final colors = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                label,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const Spacer(),
              Text(
                valueLabel,
                style: TextStyle(
                  fontSize: 13,
                  color: colors.primary,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
          Slider(
            value: value,
            min: min,
            max: max,
            divisions: divisions,
            onChanged: onChanged,
          ),
          Text(
            hint,
            style: TextStyle(
              fontSize: 11,
              color: colors.onSurface.withAlpha(100),
            ),
          ),
        ],
      ),
    );
  }
}
