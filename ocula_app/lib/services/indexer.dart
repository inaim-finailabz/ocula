import 'package:flutter/foundation.dart';
import 'rag_engine.dart';
import 'local_data.dart';

/// Background indexer v2 — crawls local data into the RAG engine.
///
/// Improvements:
/// - Fingerprint-based change detection: only re-indexes modified files
/// - Batch saves: persists once at end, not after every file
/// - Progress reporting with file counts
/// - Incremental updates: new/changed files only
/// - Resilient: individual file errors don't abort the run
class Indexer {
  final RAGEngine _rag;
  final LocalData _local;

  bool _isRunning = false;
  double _progress = 0.0;
  int _filesIndexed = 0;
  int _filesSkipped = 0;
  int _filesErrored = 0;

  bool get isRunning => _isRunning;
  double get progress => _progress;
  int get filesIndexed => _filesIndexed;
  int get filesSkipped => _filesSkipped;

  /// Optional progress callback for UI updates.
  void Function(double progress, String status)? onProgress;

  Indexer({RAGEngine? rag, LocalData? local, this.onProgress})
      : _rag = rag ?? RAGEngine(),
        _local = local ?? LocalData();

  /// Run a full index pass with smart incremental updates.
  ///
  /// Files are checked by fingerprint (mtime + size). Only new or changed
  /// files are actually re-indexed. Unchanged files are skipped instantly.
  Future<IndexResult> runFullIndex() async {
    if (_isRunning) return IndexResult.skipped();
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
  /// Does NOT save immediately — relies on periodic/app-lifecycle saves.
  Future<void> indexChatTurn(String userMessage, String assistantResponse) async {
    await _rag.init();
    await _rag.indexChat(
      userMessage: userMessage,
      assistantResponse: assistantResponse,
    );
    // Defer save — will be persisted on next runFullIndex or app pause
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
            final indexed = await _rag.indexFile(
              fileName: file.name,
              textContent: content,
              modified: file.modified,
              fingerprint: file.fingerprint,
            );

            if (indexed) {
              _filesIndexed++;
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
        await _rag.indexEmail(
          from: email.from,
          subject: email.subject,
          body: email.body,
          date: email.date,
        );
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
          await _rag.indexPhoto(
            path: photo.path,
            label: photo.label!,
            date: photo.date,
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
        await _rag.index(
          content: '${event.title} at ${event.location ?? "no location"} '
              'on ${event.start}',
          source: 'calendar',
          sourceId: 'cal:${event.title}:${event.start.toIso8601String()}',
          timestamp: event.start,
        );
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
