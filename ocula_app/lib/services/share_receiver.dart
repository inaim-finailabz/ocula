import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:receive_sharing_intent/receive_sharing_intent.dart';
import 'indexer.dart';
import 'local_data.dart';

/// Receives content shared from other apps (share sheet / send-to).
/// Auto-indexes shared content and builds a smart prompt for the assistant.
///
/// Handles: text, URLs, images, files (PDF, docs, etc.)
class ShareReceiver {
  static final ShareReceiver _instance = ShareReceiver._();
  factory ShareReceiver() => _instance;
  ShareReceiver._();

  final _indexer = Indexer();
  final _local = LocalData();

  StreamSubscription? _intentSub;

  /// Callback when shared content is ready for the assistant.
  /// Returns: (displayText, queryForModel)
  void Function(String displayText, String queryForModel)? onSharedContent;

  /// Start listening for incoming shared content.
  /// Call once from initState. No-op on desktop platforms.
  void init() {
    if (!Platform.isAndroid && !Platform.isIOS) return;

    // Listen for all shared content (text, URLs, images, files) while app is running
    _intentSub = ReceiveSharingIntent.instance.getMediaStream().listen(
      (List<SharedMediaFile> files) {
        if (files.isNotEmpty) _handleMediaFiles(files);
      },
      onError: (e) {
        if (kDebugMode) print('[ShareReceiver] Media stream error: $e');
      },
    );

    // Check for content shared while app was closed
    _checkInitialShare();

    // Also check iOS App Group for share extension data
    if (Platform.isIOS) {
      _checkiOSShareExtension();
    }
  }

  /// Check if app was launched via share intent.
  Future<void> _checkInitialShare() async {
    final initialMedia = await ReceiveSharingIntent.instance.getInitialMedia();
    if (initialMedia.isNotEmpty) {
      _handleMediaFiles(initialMedia);
    }
  }

  /// Check iOS App Group for content shared via the Share Extension.
  Future<void> _checkiOSShareExtension() async {
    try {
      // Read from shared UserDefaults via platform channel
      // The Share Extension saves JSON to "pending_shared_content"
      const channel = MethodChannel('com.finailabz.ai.ocula/share');
      final jsonString = await channel.invokeMethod<String>('getPendingShare');
      if (jsonString == null || jsonString.isEmpty) return;

      final List items = jsonDecode(jsonString);
      for (final item in items) {
        final type = item['type'] as String? ?? '';
        final content = item['content'] as String? ?? '';
        final name = item['name'] as String?;

        switch (type) {
          case 'text':
            _handleSharedText(content);
            break;
          case 'url':
            _handleSharedText(content);
            break;
          case 'image':
            _handleSharedImage(content);
            break;
          case 'file':
            _handleSharedFile(content, name);
            break;
        }
      }

      // Clear pending content
      await channel.invokeMethod('clearPendingShare');
    } catch (e) {
      if (kDebugMode) print('[ShareReceiver] iOS share check: $e');
    }
  }

  /// Handle shared text (email body, note, link, etc.)
  void _handleSharedText(String text) {
    if (kDebugMode) print('[ShareReceiver] Text received (${text.length} chars)');

    // Index the shared text
    _indexer.indexChatTurn('User shared text', text).catchError((_) {});

    // Determine what kind of content it is
    final isUrl = text.startsWith('http://') || text.startsWith('https://');
    final isEmail = text.contains('From:') && text.contains('Subject:');

    String displayText;
    String query;

    if (isUrl) {
      displayText = 'Shared link: $text';
      query = 'The user shared this link with you: $text\n\n'
          'Provide a brief summary of what this link is about. '
          'Then ask: "Would you like me to explain more, or help you with something specific about this?"';
    } else if (isEmail) {
      displayText = 'Shared email';
      final truncated = text.length > 3000 ? text.substring(0, 3000) : text;
      query = 'The user shared this email with you:\n$truncated\n\n'
          'Summarize the key points in 2-3 sentences. '
          'Then ask: "Would you like me to draft a reply, extract action items, or clarify something?"';
    } else {
      displayText = text.length > 100 ? '${text.substring(0, 100)}...' : text;
      final truncated = text.length > 3000 ? text.substring(0, 3000) : text;
      query = 'The user shared this content with you:\n$truncated\n\n'
          'Summarize this briefly. '
          'Then ask: "Would you like me to enhance, clarify, or help you with this?"';
    }

    onSharedContent?.call(displayText, query);
  }

  /// Handle a shared image file path.
  void _handleSharedImage(String path) {
    if (kDebugMode) print('[ShareReceiver] Image received: $path');

    final displayText = 'Shared image';
    final query = 'The user shared an image with you. '
        'Describe what you see and ask: "Would you like me to explain more about this image?"';

    onSharedContent?.call(displayText, query);
  }

  /// Handle a shared file (PDF, doc, etc.)
  void _handleSharedFile(String path, String? name) async {
    if (kDebugMode) print('[ShareReceiver] File received: $path');

    final fileName = name ?? path.split('/').last;

    // Index the file for RAG
    await _indexer.indexFilePath(path);

    // Try to read content
    final content = await _local.readFileContent(path);
    final truncated = content != null && content.length > 3000
        ? content.substring(0, 3000)
        : content ?? '';

    final displayText = 'Shared file: $fileName';
    String query;

    if (truncated.isNotEmpty) {
      query = 'The user shared a file "$fileName" with you:\n$truncated\n\n'
          'Summarize the key points. '
          'Then ask: "Would you like me to enhance, clarify, or extract specific information from this?"';
    } else {
      query = 'The user shared a file "$fileName" with you. '
          'Let them know you received it and ask what they\'d like to do with it.';
    }

    onSharedContent?.call(displayText, query);
  }

  /// Handle media files from receive_sharing_intent.
  void _handleMediaFiles(List<SharedMediaFile> files) {
    for (final file in files) {
      final path = file.path;
      final type = file.type;

      switch (type) {
        case SharedMediaType.text:
          _handleSharedText(path);
          break;
        case SharedMediaType.url:
          _handleSharedText(path);
          break;
        case SharedMediaType.image:
          _handleSharedImage(path);
          break;
        case SharedMediaType.video:
          _handleSharedFile(path, null);
          break;
        case SharedMediaType.file:
          _handleSharedFile(path, null);
          break;
      }
    }
  }

  void dispose() {
    _intentSub?.cancel();
  }
}
