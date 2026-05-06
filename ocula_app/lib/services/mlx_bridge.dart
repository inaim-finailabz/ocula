import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Dart bridge to the macOS MLX inference engine (MLXEngine.swift).
/// Only functional on macOS — all methods are no-ops on other platforms.
class MLXBridge {
  static final MLXBridge _instance = MLXBridge._();
  factory MLXBridge() => _instance;
  MLXBridge._();

  static const _method = MethodChannel('com.finailabz.ai.ocula/mlx');
  static const _events = EventChannel('com.finailabz.ai.ocula/mlx_stream');

  bool _isLoaded = false;
  bool get isLoaded => _isLoaded;

  /// Load an MLX model from a local directory path.
  /// Returns true on success.
  Future<bool> loadModel(String dirPath, {void Function(double)? onProgress}) async {
    if (!Platform.isMacOS) return false;
    try {
      // Listen for load progress events
      StreamSubscription? sub;
      if (onProgress != null) {
        sub = _events.receiveBroadcastStream().listen((event) {
          if (event is Map && event['loadProgress'] != null) {
            onProgress((event['loadProgress'] as num).toDouble());
          }
        });
      }
      final ok = await _method.invokeMethod<bool>('loadModel', {'path': dirPath});
      await sub?.cancel();
      _isLoaded = ok == true;
      debugPrint('[MLXBridge] loadModel: $_isLoaded ($dirPath)');
      return _isLoaded;
    } on PlatformException catch (e) {
      debugPrint('[MLXBridge] loadModel error: ${e.message}');
      return false;
    }
  }

  Future<void> unload() async {
    if (!Platform.isMacOS) return;
    try {
      await _method.invokeMethod('unload');
      _isLoaded = false;
    } catch (_) {}
  }

  /// Stream tokens from the MLX engine for the given prompt.
  /// Yields partial response strings as they arrive, then the final complete string.
  Stream<String> generateStream(
    String prompt, {
    int maxTokens = 1024,
    double temperature = 0.7,
  }) async* {
    if (!Platform.isMacOS || !_isLoaded) {
      yield '';
      return;
    }

    // Start generation — result is nil (tokens come on event channel).
    unawaited(_method.invokeMethod('generate', {
      'prompt': prompt,
      'maxTokens': maxTokens,
      'temperature': temperature,
    }));

    final buffer = StringBuffer();
    final completer = Completer<void>();

    // Listen to the event channel for token stream.
    StreamSubscription? sub;
    sub = _events.receiveBroadcastStream().listen(
      (event) {
        if (event is! Map) return;
        final token = event['token'] as String? ?? '';
        final done = event['done'] as bool? ?? false;
        final error = event['error'] as String?;

        if (error != null) {
          debugPrint('[MLXBridge] generate error: $error');
          completer.complete();
          sub?.cancel();
          return;
        }

        buffer.write(token);
        if (done) {
          completer.complete();
          sub?.cancel();
        }
      },
      onError: (e) {
        debugPrint('[MLXBridge] stream error: $e');
        completer.complete();
      },
    );

    // Yield tokens as they arrive by polling buffer changes.
    int lastLen = 0;
    while (!completer.isCompleted) {
      await Future.delayed(const Duration(milliseconds: 30));
      final current = buffer.toString();
      if (current.length > lastLen) {
        yield _cleanOutput(current);
        lastLen = current.length;
      }
    }

    // Yield final cleaned output.
    yield _cleanOutput(buffer.toString());
  }

  /// Cancel any in-progress generation.
  Future<void> cancel() async {
    if (!Platform.isMacOS) return;
    try {
      await _method.invokeMethod('cancelGeneration');
    } catch (_) {}
  }

  /// Strip ChatML artifacts and think blocks from model output.
  String _cleanOutput(String text) {
    var t = text
        .replaceAll('<|im_end|>', '')
        .replaceAll('<|im_start|>', '')
        .trimLeft();
    t = t.replaceAll(
      RegExp(
        r'<\s*(think|thinking|reasoning)\b[^>]*>.*?<\s*/\s*\1\s*>',
        caseSensitive: false,
        dotAll: true,
      ),
      '',
    );
    t = t.replaceAll(
      RegExp(
        r'```\s*(thinking|reasoning)\b[\s\S]*?```',
        caseSensitive: false,
        dotAll: true,
      ),
      '',
    );
    final openTag = RegExp(
      r'<\s*(think|thinking|reasoning)\b[^>]*>',
      caseSensitive: false,
    ).firstMatch(t);
    if (openTag != null) t = t.substring(0, openTag.start).trimRight();
    return t.trim();
  }
}
