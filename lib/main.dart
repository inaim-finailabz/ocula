import 'package:flutter/material.dart';
import 'services/ai_manager.dart';
import 'services/speech_service.dart';
import 'services/export_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const OculaApp());
}

class OculaApp extends StatelessWidget {
  const OculaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ocula',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _ai = AIManager();
  final _export = ExportService();
  late final SpeechService _speech;

  String _result = '';
  bool _isProcessing = false;
  String _activeMode = 'quick_scan';
  bool _primaryModelReady = false;

  /// Track which background models have been validated.
  final Map<String, ModelStatus> _modelStatuses = {};

  @override
  void initState() {
    super.initState();
    _speech = SpeechService(aiManager: _ai);
    _speech.init();
    _bootModels();
  }

  /// Optimised startup:
  ///  1. Load SmolVLM-256M **immediately** (smallest, ~256 MB).
  ///  2. Return control to the UI — the user can already interact.
  ///  3. Validate remaining model files in the background so later
  ///     switches are instant (no surprise "file missing" errors).
  Future<void> _bootModels() async {
    await _ai.initModels(
      onProgress: (feature, status) {
        if (!mounted) return;
        setState(() {
          _modelStatuses[feature] = status;
          if (feature == 'quick_scan' && status == ModelStatus.loaded) {
            _primaryModelReady = true;
          }
        });
      },
    );
  }

  /// Switch mode with the "placeholder trick":
  ///  • Keep the last AI result visible during the switch.
  ///  • Show a shimmer / spinner overlay (not a blank screen).
  ///  • Skip reload if we're already on the same model.
  Future<void> _switchMode(String mode) async {
    if (_activeMode == mode) return;

    setState(() {
      _isProcessing = true;
      _activeMode = mode;
    });

    // loadFeature() is a no-op if already on this model.
    await _ai.loadFeature(mode, keepLastResult: true);

    if (mounted) setState(() => _isProcessing = false);
  }

  @override
  void dispose() {
    _ai.dispose();
    super.dispose();
  }

  // ── UI helpers ──────────────────────────────────────────────────────
  Widget _buildStatusBar() {
    if (!_primaryModelReady) {
      return const Padding(
        padding: EdgeInsets.symmetric(vertical: 12),
        child: Column(
          children: [
            LinearProgressIndicator(),
            SizedBox(height: 4),
            Text('Loading AI engine…', style: TextStyle(fontSize: 12)),
          ],
        ),
      );
    }

    // Show readiness dots for each model tier
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      child: Row(
        children: AIManager.models
            .map((m) => Padding(
                  padding: const EdgeInsets.only(right: 6),
                  child: Tooltip(
                    message: '${m.feature}: ${m.status.name}',
                    child: Icon(
                      m.status == ModelStatus.loaded || m.status == ModelStatus.ready
                          ? Icons.check_circle
                          : m.status == ModelStatus.missing
                              ? Icons.cloud_download
                              : Icons.circle_outlined,
                      size: 12,
                      color: m.status == ModelStatus.loaded
                          ? Colors.greenAccent
                          : m.status == ModelStatus.ready
                              ? Colors.white54
                              : m.status == ModelStatus.missing
                                  ? Colors.orangeAccent
                                  : Colors.grey,
                    ),
                  ),
                ))
            .toList(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Determine what to show in the result area.
    // If a model switch is in progress, keep the last result visible
    // (the "placeholder trick" from MODEL_STRATEGY.md).
    final displayResult = _result.isNotEmpty
        ? _result
        : _ai.lastResult.isNotEmpty
            ? _ai.lastResult
            : 'Point camera and tap capture';

    return Scaffold(
      appBar: AppBar(
        title: const Text('Ocula'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(20),
          child: _buildStatusBar(),
        ),
      ),
      body: Column(
        children: [
          // Mode selector
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: SegmentedButton<String>(
              segments: const [
                ButtonSegment(value: 'quick_scan', label: Text('Scan')),
                ButtonSegment(value: 'detail', label: Text('Detail')),
                ButtonSegment(value: 'document', label: Text('Document')),
                ButtonSegment(value: 'reasoning', label: Text('Reason')),
              ],
              selected: {_activeMode},
              onSelectionChanged: _primaryModelReady
                  ? (sel) => _switchMode(sel.first)
                  : null, // disable until primary model is ready
            ),
          ),

          // Status — shimmer overlay during model switch
          if (_isProcessing)
            const Padding(
              padding: EdgeInsets.all(16.0),
              child: Column(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 8),
                  Text('Deep Analysis in Progress…'),
                ],
              ),
            ),

          // Result display (shows last result as placeholder during switch)
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: AnimatedOpacity(
                opacity: _isProcessing ? 0.4 : 1.0,
                duration: const Duration(milliseconds: 200),
                child: Text(displayResult),
              ),
            ),
          ),
        ],
      ),
      bottomNavigationBar: BottomAppBar(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            // Voice input
            IconButton(
              icon: Icon(_speech.isListening ? Icons.mic : Icons.mic_none),
              onPressed: !_primaryModelReady
                  ? null
                  : () {
                      if (_speech.isListening) {
                        _speech.stopListening();
                      } else {
                        _speech.startListening(
                          onResult: (text) =>
                              setState(() => _result = 'You said: $text'),
                          onAIResponse: (response) =>
                              setState(() => _result = response),
                        );
                      }
                    },
            ),
            // Capture (placeholder — would trigger camera)
            IconButton(
              icon: const Icon(Icons.camera_alt, size: 36),
              onPressed: !_primaryModelReady
                  ? null
                  : () {
                      // TODO: Capture image from camera and run inference
                    },
            ),
            // Export to PDF
            IconButton(
              icon: const Icon(Icons.picture_as_pdf),
              onPressed: _result.isEmpty
                  ? null
                  : () => _export.exportAndShare(_result),
            ),
          ],
        ),
      ),
    );
  }
}
