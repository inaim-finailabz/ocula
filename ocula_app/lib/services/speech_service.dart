import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:path_provider/path_provider.dart';
import 'ai_manager.dart';

/// Handles voice input (STT) and voice output (TTS) for the AI assistant.
class SpeechService {
  final stt.SpeechToText _speech = stt.SpeechToText();
  final FlutterTts _tts = FlutterTts();
  final AIManager _aiManager;

  bool _isListening = false;
  bool _sttInitialized = false;
  bool _isSpeaking = false;

  bool get isListening => _isListening;
  FlutterTts get tts => _tts;

  // Current settings
  double _rate = 0.5;
  double _pitch = 1.0;
  double _volume = 1.0;
  String _language = 'en-US';
  Map<String, String>? _voice; // {"name": ..., "locale": ...}

  /// Path to user-uploaded custom voice sample file.
  /// Stored locally for future on-device voice cloning support.
  String? _customVoicePath;

  double get rate => _rate;
  double get pitch => _pitch;
  double get volume => _volume;
  String get language => _language;
  Map<String, String>? get voice => _voice;
  String? get customVoicePath => _customVoicePath;
  bool get hasCustomVoice =>
      _customVoicePath != null && _customVoicePath!.isNotEmpty;

  bool _voiceEnabled = true;
  bool get voiceEnabled => _voiceEnabled;

  Future<void> setVoiceEnabled(bool v) async {
    _voiceEnabled = v;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('voice_enabled', v);
  }

  SpeechService({AIManager? aiManager}) : _aiManager = aiManager ?? AIManager();

  /// Initialize TTS with saved or default voice settings.
  Future<void> init() async {
    await _loadSettings();
    await _applySettings();

    // Make speak() Future wait until TTS actually finishes (or is stopped).
    // Without this, speak() resolves immediately after the platform receives
    // the command, causing _isSpeaking to flip false while audio plays.
    await _tts.awaitSpeakCompletion(true);
    _tts.setStartHandler(() => _isSpeaking = true);
    _tts.setCompletionHandler(() => _isSpeaking = false);
    _tts.setCancelHandler(() => _isSpeaking = false);
    _tts.setErrorHandler((_) => _isSpeaking = false);
  }

  /// Get available TTS voices.
  Future<List<Map<String, String>>> getVoices() async {
    final voices = await _tts.getVoices;
    final list = <Map<String, String>>[];
    for (final v in voices) {
      final map = Map<String, String>.from(v as Map);
      list.add(map);
    }
    // Sort by name for easier browsing
    list.sort((a, b) => (a['name'] ?? '').compareTo(b['name'] ?? ''));
    return list;
  }

  /// Get available TTS languages.
  Future<List<String>> getLanguages() async {
    final langs = await _tts.getLanguages;
    final list = langs.cast<String>().toList();
    list.sort();
    return list;
  }

  /// Update speech rate (0.0 to 1.0).
  Future<void> setRate(double value) async {
    _rate = value;
    await _tts.setSpeechRate(value);
    await _saveSettings();
  }

  /// Update pitch (0.5 to 2.0).
  Future<void> setPitch(double value) async {
    _pitch = value;
    await _tts.setPitch(value);
    await _saveSettings();
  }

  /// Update volume (0.0 to 1.0).
  Future<void> setVolume(double value) async {
    _volume = value;
    await _tts.setVolume(value);
    await _saveSettings();
  }

  /// Update language.
  Future<void> setLanguage(String lang) async {
    _language = lang;
    await _tts.setLanguage(lang);
    await _saveSettings();
  }

  /// Update voice.
  Future<void> setVoice(Map<String, String> v) async {
    _voice = v;
    await _tts.setVoice(v);
    await _saveSettings();
  }

  /// Preview current TTS settings.
  Future<void> preview() async {
    await speak('Hello, I am Ocula. How can I help you today?');
  }

  /// Start listening to the user's voice and pass recognized text to the AI.
  Future<void> startListening({
    required Function(String) onResult,
    Function(String)? onAIResponse,
    Function(String)? onFinalText,
    Function? onError,
  }) async {
    // Initialize STT only once
    if (!_sttInitialized) {
      final available = await _speech.initialize(
        onError: (error) {
          debugPrint('STT error: $error');
          // Only invoke onError for permanent failures (listen session ended)
          if (error.permanent && _isListening) {
            _isListening = false;
            if (onError != null) {
              onError();
            }
          }
        },
      );
      debugPrint('STT available: $available');

      if (!available) {
        if (onError != null) {
          onError();
        }
        return;
      }
      _sttInitialized = true;
    }

    _isListening = true;
    bool handled = false;
    _speech.listen(
      onResult: (val) async {
        final text = val.recognizedWords;
        onResult(text);

        // When speech is finalized (silence detected), send to the AI model
        if (val.finalResult && text.isNotEmpty && !handled) {
          handled = true;
          _isListening = false;
          if (onFinalText != null) {
            onFinalText(text);
          }
          if (onAIResponse != null) {
            final response = await _aiManager.ask(text);
            onAIResponse(response);
            await speak(response);
          }
        }
      },
      // Auto-stop after 2s of silence — triggers finalResult
      pauseFor: const Duration(seconds: 2),
      // Max listen time before auto-stop
      listenFor: const Duration(seconds: 30),
    );
  }

  /// Stop listening.
  Future<void> stopListening() async {
    _isListening = false;
    await _speech.stop();
  }

  /// Speak the given text aloud.
  Future<void> speak(String text) async {
    final normalized = text.trim();
    if (normalized.isEmpty) return;
    if (!_voiceEnabled) return;

    // Avoid audio-session conflicts between STT input and TTS output.
    if (_isListening) {
      await stopListening();
      await Future.delayed(const Duration(milliseconds: 80));
    }

    // If there is already speech in progress, restart cleanly.
    if (_isSpeaking) {
      await _tts.stop();
      await Future.delayed(const Duration(milliseconds: 60));
    }

    _isSpeaking = true;
    try {
      await _tts.speak(normalized);
    } catch (_) {
      _isSpeaking = false;
      rethrow;
    }
  }

  /// Stop any ongoing speech.
  Future<void> stopSpeaking() async {
    _isSpeaking = false;
    await _tts.stop();
  }

  // ── Custom Voice Upload ──

  /// Save a custom voice sample file.
  /// Copies the source file to the app's documents directory so it persists.
  /// Returns the stored path on success, null on failure.
  Future<String?> saveCustomVoice(File sourceFile) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final voiceDir = Directory('${dir.path}/custom_voices');
      if (!await voiceDir.exists()) {
        await voiceDir.create(recursive: true);
      }

      final ext = sourceFile.path.split('.').last;
      final destPath = '${voiceDir.path}/custom_voice.$ext';
      await sourceFile.copy(destPath);

      _customVoicePath = destPath;
      await _saveSettings();
      debugPrint('[SpeechService] Custom voice saved: $destPath');
      return destPath;
    } catch (e) {
      debugPrint('[SpeechService] Failed to save custom voice: $e');
      return null;
    }
  }

  /// Remove the stored custom voice sample.
  Future<void> removeCustomVoice() async {
    if (_customVoicePath != null) {
      try {
        final file = File(_customVoicePath!);
        if (await file.exists()) {
          await file.delete();
        }
      } catch (_) {}
      _customVoicePath = null;
      await _saveSettings();
    }
  }

  // ── Persistence ──

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    _rate = prefs.getDouble('tts_rate') ?? 0.5;
    _pitch = prefs.getDouble('tts_pitch') ?? 1.0;
    _volume = prefs.getDouble('tts_volume') ?? 1.0;
    _language = prefs.getString('tts_language') ?? 'en-US';
    _voiceEnabled = prefs.getBool('voice_enabled') ?? true;
    _customVoicePath = prefs.getString('tts_custom_voice_path');
    final voiceName = prefs.getString('tts_voice_name');
    final voiceLocale = prefs.getString('tts_voice_locale');
    if (voiceName != null && voiceLocale != null) {
      _voice = {'name': voiceName, 'locale': voiceLocale};
    }
  }

  Future<void> _applySettings() async {
    await _tts.setLanguage(_language);
    await _tts.setSpeechRate(_rate);
    await _tts.setPitch(_pitch);
    await _tts.setVolume(_volume);
    if (_voice != null) {
      await _tts.setVoice(_voice!);
    }
  }

  Future<void> _saveSettings() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble('tts_rate', _rate);
    await prefs.setDouble('tts_pitch', _pitch);
    await prefs.setDouble('tts_volume', _volume);
    await prefs.setString('tts_language', _language);
    if (_voice != null) {
      await prefs.setString('tts_voice_name', _voice!['name'] ?? '');
      await prefs.setString('tts_voice_locale', _voice!['locale'] ?? '');
    }
    if (_customVoicePath != null) {
      await prefs.setString('tts_custom_voice_path', _customVoicePath!);
    } else {
      await prefs.remove('tts_custom_voice_path');
    }
  }
}
