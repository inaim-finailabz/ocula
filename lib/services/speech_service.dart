import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart';
import 'ai_manager.dart';

/// Handles voice input (STT) and voice output (TTS) for the AI assistant.
class SpeechService {
  final stt.SpeechToText _speech = stt.SpeechToText();
  final FlutterTts _tts = FlutterTts();
  final AIManager _aiManager;

  bool _isListening = false;

  bool get isListening => _isListening;

  SpeechService({AIManager? aiManager})
      : _aiManager = aiManager ?? AIManager();

  /// Initialize TTS with natural voice settings.
  Future<void> init() async {
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.5);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);
  }

  /// Start listening to the user's voice and pass recognized text to the AI.
  Future<void> startListening({
    required Function(String) onResult,
    Function(String)? onAIResponse,
  }) async {
    final available = await _speech.initialize();
    if (!available) return;

    _isListening = true;
    _speech.listen(onResult: (val) async {
      final text = val.recognizedWords;
      onResult(text);

      // When speech is finalized, send to the AI model
      if (val.finalResult && text.isNotEmpty) {
        _isListening = false;
        if (onAIResponse != null) {
          final response = await _aiManager.ask(text);
          onAIResponse(response);
          await speak(response);
        }
      }
    });
  }

  /// Stop listening.
  Future<void> stopListening() async {
    _isListening = false;
    await _speech.stop();
  }

  /// Speak the given text aloud.
  Future<void> speak(String text) async {
    await _tts.speak(text);
  }

  /// Stop any ongoing speech.
  Future<void> stopSpeaking() async {
    await _tts.stop();
  }
}
