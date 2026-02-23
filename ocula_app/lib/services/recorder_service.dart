import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'ai_manager.dart';

/// Recording mode — controls the AI summarization prompt.
enum RecorderMode { meeting, lecture, notes }

/// Continuous speech-to-text recorder with AI summarization.
///
/// Usage:
///   final svc = RecorderService();
///   await svc.start();
///   // listen to svc.transcriptStream for live updates
///   await svc.stop();
///   final summary = await svc.summarize(RecorderMode.meeting);
///   svc.dispose();
class RecorderService {
  final stt.SpeechToText _speech = stt.SpeechToText();
  final AIManager _aiManager;

  bool _isRecording = false;
  bool _sttInitialized = false;
  bool _stopped = false;

  final StringBuffer _transcript = StringBuffer();
  String _currentPartial = '';
  final Stopwatch _stopwatch = Stopwatch();

  final StreamController<String> _transcriptController =
      StreamController<String>.broadcast();

  Stream<String> get transcriptStream => _transcriptController.stream;
  bool get isRecording => _isRecording;

  /// The committed transcript (all finalized speech, excluding the in-progress partial).
  String get transcript => _transcript.toString().trim();

  /// Full transcript including any uncommitted partial.
  String get fullTranscript {
    final base = _transcript.toString().trim();
    final partial = _currentPartial.trim();
    if (partial.isEmpty) return base;
    return base.isEmpty ? partial : '$base $partial';
  }

  Duration get elapsed => _stopwatch.elapsed;

  RecorderService({AIManager? aiManager})
      : _aiManager = aiManager ?? AIManager();

  // ── Lifecycle ──

  /// Initialize STT and start continuous recording.
  /// Returns false if the microphone / STT is unavailable.
  Future<bool> start() async {
    if (_isRecording) return true;

    if (!_sttInitialized) {
      final available = await _speech.initialize(
        onError: (error) {
          debugPrint('[RecorderService] STT error: $error');
          // If the session ends permanently and we're still recording, restart.
          if (error.permanent && _isRecording) {
            Future.delayed(const Duration(milliseconds: 200), _startSession);
          }
        },
        onStatus: (status) {
          debugPrint('[RecorderService] STT status: $status');
          // 'done' / 'notListening' fires at the end of each session.
          // Restart immediately if we're still recording.
          if (_isRecording &&
              (status == 'done' || status == 'notListening')) {
            Future.delayed(const Duration(milliseconds: 150), _startSession);
          }
        },
      );
      if (!available) return false;
      _sttInitialized = true;
    }

    _transcript.clear();
    _currentPartial = '';
    _stopped = false;
    _isRecording = true;
    _stopwatch
      ..reset()
      ..start();
    _startSession();
    return true;
  }

  /// Stop recording and return the committed transcript.
  Future<String> stop() async {
    _stopped = true;
    _isRecording = false;
    _stopwatch.stop();
    await _speech.stop();
    // Flush any uncommitted partial into the main buffer.
    if (_currentPartial.trim().isNotEmpty) {
      _transcript.write(' ${_currentPartial.trim()}');
      _currentPartial = '';
    }
    _transcriptController.add(fullTranscript);
    return transcript;
  }

  // ── Internal STT session loop ──

  void _startSession() {
    if (_stopped || !_isRecording) return;

    _speech.listen(
      onResult: (val) {
        if (!_isRecording) return;
        final text = val.recognizedWords;

        if (val.finalResult) {
          // Commit finalized text.
          if (text.trim().isNotEmpty) {
            if (_transcript.isNotEmpty) _transcript.write(' ');
            _transcript.write(text.trim());
          }
          _currentPartial = '';
          _transcriptController.add(fullTranscript);

          // Restart session to keep recording.
          if (_isRecording && !_stopped) {
            Future.delayed(const Duration(milliseconds: 100), _startSession);
          }
        } else {
          // Live partial — update without committing.
          _currentPartial = text;
          _transcriptController.add(fullTranscript);
        }
      },
      // 12s silence → finalResult triggers restart
      pauseFor: const Duration(seconds: 12),
      // 3 min max per session
      listenFor: const Duration(minutes: 3),
      listenOptions: stt.SpeechListenOptions(
        cancelOnError: false,
        partialResults: true,
      ),
    );
  }

  // ── Summarization ──

  /// Summarize the recorded transcript using the best available model.
  ///
  /// Upgrade order: Pro → Plus → current (Free).
  /// The engine is NOT rolled back after summarization, so the user keeps
  /// the better model for subsequent queries.
  Future<String> summarize(RecorderMode mode) async {
    final text = fullTranscript;
    if (text.trim().isEmpty) {
      return 'No transcript to summarize.';
    }

    // Try to upgrade to a more capable model for summarization.
    await _tryUpgradeForSummary();

    // Truncate to protect context window (~8000 chars max).
    final truncated = _truncate(text, 8000);
    final prompt = _buildPrompt(mode, truncated);

    try {
      return await _aiManager.ask(prompt);
    } catch (e) {
      debugPrint('[RecorderService] summarize error: $e');
      return 'Error generating summary: $e';
    }
  }

  /// Attempt to switch to the best available tier for a high-quality summary.
  Future<void> _tryUpgradeForSummary() async {
    for (final tier in [AITier.pro, AITier.plus]) {
      try {
        if (await _aiManager.isTierDownloaded(tier)) {
          await _aiManager.switchEngine(tier);
          debugPrint('[RecorderService] Upgraded to ${tier.name} for summary');
          return;
        }
      } catch (e) {
        debugPrint('[RecorderService] Could not upgrade to ${tier.name}: $e');
      }
    }
    debugPrint('[RecorderService] Using current tier for summary');
  }

  String _truncate(String text, int maxChars) {
    if (text.length <= maxChars) return text;
    // Truncate from the middle: keep first 60% and last 40%.
    final keep1 = (maxChars * 0.6).round();
    final keep2 = maxChars - keep1;
    return '${text.substring(0, keep1)}\n\n[... transcript truncated ...]\n\n'
        '${text.substring(text.length - keep2)}';
  }

  String _buildPrompt(RecorderMode mode, String text) {
    switch (mode) {
      case RecorderMode.meeting:
        return '''You are a professional meeting summariser. Analyse the following meeting transcript and produce a concise, structured summary in Markdown.

Use these sections (omit any that are not applicable):
## Overview
## Discussion Points
## Decisions Made
## Action Items  (use a checklist: - [ ] Owner: task)
## Next Steps

Keep each section brief and factual. Do not invent details not present in the transcript.

--- TRANSCRIPT ---
$text
--- END ---''';

      case RecorderMode.lecture:
        return '''You are an expert academic note-taker. Analyse the following lecture transcript and produce clear, structured study notes in Markdown.

Use these sections (omit any that are not applicable):
## Topic & Objectives
## Key Concepts
## Important Details & Examples
## Key Terms  (term: definition)
## Study Notes / Summary

Write in a style suitable for revision. Be concise but thorough.

--- TRANSCRIPT ---
$text
--- END ---''';

      case RecorderMode.notes:
        return '''You are a smart personal assistant. Organise the following voice notes into a clean, structured document in Markdown.

Use these sections (omit any that are not applicable):
## Summary
## Main Points
## To-Do / Follow-Ups  (use a checklist: - [ ] item)

Remove filler words, repetition, and false starts. Keep the meaning intact.

--- TRANSCRIPT ---
$text
--- END ---''';
    }
  }

  // ── Dispose ──

  Future<void> dispose() async {
    if (_isRecording) await stop();
    await _transcriptController.close();
  }
}
