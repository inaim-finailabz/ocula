import 'package:flutter/material.dart';
import '../services/rag_config.dart';

/// Settings card for tuning RAG search and generation parameters.
class RagSettings extends StatefulWidget {
  const RagSettings({super.key});

  @override
  State<RagSettings> createState() => _RagSettingsState();
}

class _RagSettingsState extends State<RagSettings> {
  final RagConfig _config = RagConfig();
  bool _loading = true;

  late double _vectorWeight;
  late int _topK;
  late double _minScore;
  late int _contextBudget;
  late int _maxTokens;
  late int _chunkSize;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    await _config.load();
    if (!mounted) return;
    setState(() {
      _vectorWeight = _config.vectorWeight;
      _topK = _config.topK;
      _minScore = _config.minScore;
      _contextBudget = _config.contextBudgetChars;
      _maxTokens = _config.maxResponseTokens;
      _chunkSize = _config.chunkSize;
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

            _sliderTile(
              label: 'Semantic vs Keyword',
              value: _vectorWeight,
              min: 0.0,
              max: 1.0,
              divisions: 20,
              valueLabel: '${(_vectorWeight * 100).round()}% semantic',
              hint: 'Higher = more meaning-based search. Lower = more keyword matching.',
              onChanged: (v) {
                setState(() => _vectorWeight = v);
                _config.setVectorWeight(v);
              },
            ),

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

            _sliderTile(
              label: 'Quality Threshold',
              value: _minScore,
              min: 0.0,
              max: 0.5,
              divisions: 10,
              valueLabel: _minScore.toStringAsFixed(2),
              hint: 'Minimum relevance score. Higher = stricter, fewer results.',
              onChanged: (v) {
                setState(() => _minScore = v);
                _config.setMinScore(v);
              },
            ),

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

            _sliderTile(
              label: 'Max Response Length',
              value: _maxTokens.toDouble(),
              min: 50,
              max: 1024,
              divisions: 19,
              valueLabel: '$_maxTokens tokens',
              hint: 'Maximum generation length. Larger models use the full budget.',
              onChanged: (v) {
                setState(() => _maxTokens = v.round());
                _config.setMaxResponseTokens(v.round());
              },
            ),

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
                style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
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
            style: TextStyle(fontSize: 11, color: colors.onSurface.withAlpha(100)),
          ),
        ],
      ),
    );
  }
}
