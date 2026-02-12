import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'rag_engine.dart';
import 'local_data.dart';
import 'ocula_db.dart';

/// Background indexer v3 — lifecycle-aware, asset-linking, all-platform.
///
/// Improvements over v2:
/// - WidgetsBindingObserver: re-indexes on app resume
/// - Periodic timer: re-indexes every 15 minutes while app is active
/// - Asset linking: detects phone numbers, emails, file paths, URLs in
///   indexed content and stores them as asset_links for surfacing in chat
/// - Identical behavior on iOS, Android, macOS, Linux, Windows
class Indexer with WidgetsBindingObserver {
  static final Indexer _instance = Indexer._internal();
  factory Indexer() => _instance;

  final RAGEngine _rag;
  final LocalData _local;
  final OculaDB _db;

  bool _isRunning = false;
  double _progress = 0.0;
  int _filesIndexed = 0;
  int _filesSkipped = 0;
  int _filesErrored = 0;
  Timer? _periodicTimer;
  bool _lifecycleRegistered = false;
  DateTime? _lastFullIndex;

  /// Minimum interval between full index runs (avoid battery drain).
  static const _minIndexInterval = Duration(minutes: 15);

  bool get isRunning => _isRunning;
  double get progress => _progress;
  int get filesIndexed => _filesIndexed;
  int get filesSkipped => _filesSkipped;

  /// Optional progress callback for UI updates.
  void Function(double progress, String status)? onProgress;

  Indexer._internal({RAGEngine? rag, LocalData? local, OculaDB? db})
      : _rag = rag ?? RAGEngine(),
        _local = local ?? LocalData(),
        _db = db ?? OculaDB();

  /// Start lifecycle-aware background indexing.
  /// Call once from initState. Safe to call multiple times.
  void startBackgroundIndexing() {
    if (!_lifecycleRegistered) {
      WidgetsBinding.instance.addObserver(this);
      _lifecycleRegistered = true;
    }

    // Kick off the first index
    runFullIndex();

    // Periodic re-index while app is foregrounded
    _periodicTimer?.cancel();
    _periodicTimer = Timer.periodic(_minIndexInterval, (_) {
      runFullIndex();
    });
  }

  /// Stop background indexing (call from dispose).
  void stopBackgroundIndexing() {
    _periodicTimer?.cancel();
    _periodicTimer = null;
    if (_lifecycleRegistered) {
      WidgetsBinding.instance.removeObserver(this);
      _lifecycleRegistered = false;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      // App came back to foreground — re-index if enough time has passed
      if (_lastFullIndex == null ||
          DateTime.now().difference(_lastFullIndex!) > _minIndexInterval) {
        runFullIndex();
      }
    }
  }

  /// Run a full index pass with smart incremental updates.
  ///
  /// Files are checked by fingerprint (mtime + size). Only new or changed
  /// files are actually re-indexed. Unchanged files are skipped instantly.
  /// Links phone numbers, emails, file paths found in content.
  Future<IndexResult> runFullIndex() async {
    if (_isRunning) return IndexResult.skipped();

    // Throttle: don't re-index too frequently
    if (_lastFullIndex != null &&
        DateTime.now().difference(_lastFullIndex!) < const Duration(minutes: 5)) {
      return IndexResult.skipped();
    }

    _isRunning = true;
    _progress = 0.0;
    _filesIndexed = 0;
    _filesSkipped = 0;
    _filesErrored = 0;

    await _rag.init();

    final stopwatch = Stopwatch()..start();

    // Step 1: Files (60% of progress) — most impactful for RAG
    _reportProgress(0.0, 'Scanning files...');
    await _indexFiles();
    _reportProgress(0.6, 'Files done. Indexing emails...');

    // Step 2: Emails (15% of progress)
    await _indexEmails();
    _reportProgress(0.75, 'Emails done. Indexing photos...');

    // Step 3: Photos (15% of progress)
    await _indexPhotos();
    _reportProgress(0.9, 'Photos done. Indexing calendar...');

    // Step 4: Calendar (10% of progress)
    await _indexCalendar();
    _reportProgress(0.95, 'Saving index...');

    // Single batch save at the end
    await _rag.save();
    _reportProgress(1.0, 'Complete');

    stopwatch.stop();
    _isRunning = false;
    _lastFullIndex = DateTime.now();

    final result = IndexResult(
      filesIndexed: _filesIndexed,
      filesSkipped: _filesSkipped,
      filesErrored: _filesErrored,
      totalEntries: _rag.entryCount,
      durationMs: stopwatch.elapsedMilliseconds,
    );

    if (kDebugMode) {
      print('[Indexer] $result');
    }

    return result;
  }

  /// Index a single chat conversation turn (call after each AI response).
  /// Also detects and links phone numbers, emails, URLs mentioned in chat.
  Future<void> indexChatTurn(String userMessage, String assistantResponse) async {
    await _rag.init();
    final sourceId = 'chat:${DateTime.now().millisecondsSinceEpoch}';
    await _rag.indexChat(
      userMessage: userMessage,
      assistantResponse: assistantResponse,
    );
    // Detect phone numbers, emails, URLs in both user and assistant text
    await _db.linkDetectedAssets(
      '$userMessage\n$assistantResponse',
      'chat',
      sourceId,
    );
    await _rag.save();
  }

  /// Manually index a specific file path.
  Future<bool> indexFilePath(String path) async {
    await _rag.init();
    final content = await _local.readFileContent(path);
    if (content == null || content.isEmpty) return false;

    final name = path.split('/').last;
    final fingerprint = await _local.fileFingerprint(path);

    await _rag.indexFile(
      fileName: name,
      textContent: content,
      modified: DateTime.now(),
      fingerprint: fingerprint,
    );
    await _rag.save();

    if (kDebugMode) {
      print('[Indexer] Indexed file: $name (${content.length} chars)');
    }
    return true;
  }

  void _reportProgress(double progress, String status) {
    _progress = progress;
    onProgress?.call(progress, status);
  }

  Future<void> _indexFiles() async {
    try {
      final files = await _local.searchFiles('');
      if (kDebugMode) {
        print('[Indexer] Found ${files.length} text files to scan');
      }

      for (int i = 0; i < files.length; i++) {
        final file = files[i];

        // Progress within file step (0.0 → 0.6)
        _progress = 0.6 * (i / files.length);

        // Check fingerprint — skip unchanged files
        final sourceId = 'file:${file.name}';
        final check = await _rag.checkFingerprintAsync(sourceId, file.fingerprint);
        if (check == true) {
          _filesSkipped++;
          continue; // Unchanged — skip
        }

        // Read & index
        try {
          final content = await _local.readFileContent(file.path);
          if (content != null && content.isNotEmpty) {
            final sourceIdFull = 'file:${file.name}';
            final indexed = await _rag.indexFile(
              fileName: file.name,
              textContent: content,
              modified: file.modified,
              fingerprint: file.fingerprint,
            );

            if (indexed) {
              _filesIndexed++;
              // Link the file itself as an openable asset
              await _db.linkAsset(
                sourceId: sourceIdFull,
                assetType: 'file',
                assetRef: file.path,
                label: file.name,
              );
              // Detect phone numbers, emails, URLs in content
              await _db.linkDetectedAssets(content, 'file', sourceIdFull);
            } else {
              _filesSkipped++;
            }
          }
        } catch (e) {
          _filesErrored++;
          if (kDebugMode) {
            print('[Indexer] Error indexing ${file.name}: $e');
          }
        }
      }
    } catch (e) {
      if (kDebugMode) print('[Indexer] File indexing error: $e');
    }
  }

  Future<void> _indexEmails() async {
    try {
      final emails = await _local.recentEmails(limit: 50);
      for (final email in emails) {
        final sourceId = 'email:${email.subject}:${email.date.toIso8601String()}';
        await _rag.indexEmail(
          from: email.from,
          subject: email.subject,
          body: email.body,
          date: email.date,
        );
        // Link sender as contact
        await _db.linkAsset(
          sourceId: sourceId,
          assetType: 'email',
          assetRef: 'mailto:${email.from}',
          label: '${email.from} — ${email.subject}',
        );
        // Link attachments
        for (final att in email.attachments) {
          await _db.linkAsset(
            sourceId: sourceId,
            assetType: 'file',
            assetRef: att,
            label: att.split('/').last,
          );
        }
        // Detect phone numbers, emails, URLs in body
        await _db.linkDetectedAssets(email.body, 'email', sourceId);
      }
    } catch (e) {
      if (kDebugMode) print('[Indexer] Email indexing skipped: $e');
    }
  }

  Future<void> _indexPhotos() async {
    try {
      final photos = await _local.recentPhotos(limit: 100);
      for (final photo in photos) {
        if (photo.label != null) {
          final sourceId = 'photo:${photo.path}';
          await _rag.indexPhoto(
            path: photo.path,
            label: photo.label!,
            date: photo.date,
          );
          // Link photo file for inline display
          await _db.linkAsset(
            sourceId: sourceId,
            assetType: 'photo',
            assetRef: photo.path,
            label: photo.label ?? photo.path.split('/').last,
          );
        }
      }
    } catch (e) {
      if (kDebugMode) print('[Indexer] Photo indexing skipped: $e');
    }
  }

  Future<void> _indexCalendar() async {
    try {
      final now = DateTime.now();
      final events = await _local.getEvents(
        now.subtract(const Duration(days: 7)),
        now.add(const Duration(days: 7)),
      );
      for (final event in events) {
        final sourceId = 'cal:${event.title}:${event.start.toIso8601String()}';
        final content = '${event.title} at ${event.location ?? "no location"} '
            'on ${event.start}';
        await _rag.index(
          content: content,
          source: 'calendar',
          sourceId: sourceId,
          timestamp: event.start,
        );
        // Link calendar event
        await _db.linkAsset(
          sourceId: sourceId,
          assetType: 'calendar',
          assetRef: 'cal:${event.title}',
          label: '${event.title} — ${event.start.toLocal().toString().substring(0, 16)}',
        );
        // Detect phone/email/URL in event details
        final details = '${event.title} ${event.location ?? ''}';
        await _db.linkDetectedAssets(details, 'calendar', sourceId);
      }
    } catch (e) {
      if (kDebugMode) print('[Indexer] Calendar indexing skipped: $e');
    }
  }
}

/// Result of an indexing run.
class IndexResult {
  final int filesIndexed;
  final int filesSkipped;
  final int filesErrored;
  final int totalEntries;
  final int durationMs;
  final bool wasSkipped;

  IndexResult({
    required this.filesIndexed,
    required this.filesSkipped,
    required this.filesErrored,
    required this.totalEntries,
    required this.durationMs,
    this.wasSkipped = false,
  });

  factory IndexResult.skipped() => IndexResult(
        filesIndexed: 0,
        filesSkipped: 0,
        filesErrored: 0,
        totalEntries: 0,
        durationMs: 0,
        wasSkipped: true,
      );

  @override
  String toString() =>
      'IndexResult(indexed=$filesIndexed, skipped=$filesSkipped, '
      'errors=$filesErrored, total=$totalEntries, ${durationMs}ms)';
}
