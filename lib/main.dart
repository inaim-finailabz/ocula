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

  @override
  void initState() {
    super.initState();
    _speech = SpeechService(aiManager: _ai);
    _speech.init();
    _initDefaultModel();
  }

  Future<void> _initDefaultModel() async {
    // Load the free-tier model on app start (SmolVLM-256M)
    await _ai.loadFeature('quick_scan');
  }

  Future<void> _switchMode(String mode) async {
    if (_activeMode == mode) return;

    setState(() {
      _isProcessing = true;
      _activeMode = mode;
    });

    await _ai.loadFeature(mode);

    setState(() => _isProcessing = false);
  }

  @override
  void dispose() {
    _ai.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Ocula')),
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
              onSelectionChanged: (sel) => _switchMode(sel.first),
            ),
          ),

          // Status
          if (_isProcessing)
            const Padding(
              padding: EdgeInsets.all(16.0),
              child: Column(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 8),
                  Text('Deep Analysis in Progress...'),
                ],
              ),
            ),

          // Result display
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Text(_result.isEmpty ? 'Point camera and tap capture' : _result),
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
              onPressed: () {
                if (_speech.isListening) {
                  _speech.stopListening();
                } else {
                  _speech.startListening(
                    onResult: (text) => setState(() => _result = 'You said: $text'),
                    onAIResponse: (response) => setState(() => _result = response),
                  );
                }
              },
            ),
            // Capture (placeholder — would trigger camera)
            IconButton(
              icon: const Icon(Icons.camera_alt, size: 36),
              onPressed: () {
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
