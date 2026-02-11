import 'rag_engine.dart';
import 'local_data.dart';

/// Background indexer that crawls local data and feeds it into the RAG engine.
/// Runs once on first launch, then incrementally on each app open.
///
/// Priority order:
/// 1. Recent emails (last 7 days) — most likely to be asked about
/// 2. Recent photos (last 7 days) — with AI labels
/// 3. Files in Downloads/Documents
/// 4. Contacts
/// 5. Calendar events (next 7 days)
class Indexer {
  final RAGEngine _rag;
  final LocalData _local;

  bool _isRunning = false;
  double _progress = 0.0;

  bool get isRunning => _isRunning;
  double get progress => _progress;

  Indexer({RAGEngine? rag, LocalData? local})
      : _rag = rag ?? RAGEngine(),
        _local = local ?? LocalData();

  /// Run a full index pass. Safe to call multiple times — skips already-indexed items.
  /// Call this on app start in an isolate so it doesn't block the UI.
  Future<void> runFullIndex() async {
    if (_isRunning) return;
    _isRunning = true;
    _progress = 0.0;

    await _rag.init();

    // Step 1: Emails (40% of progress)
    await _indexEmails();
    _progress = 0.4;

    // Step 2: Photos (30% of progress)
    await _indexPhotos();
    _progress = 0.7;

    // Step 3: Files (20% of progress)
    await _indexFiles();
    _progress = 0.9;

    // Step 4: Calendar (10% of progress)
    await _indexCalendar();
    _progress = 1.0;

    _isRunning = false;
  }

  Future<void> _indexEmails() async {
    final emails = await _local.recentEmails(limit: 50);
    for (final email in emails) {
      await _rag.indexEmail(
        from: email.from,
        subject: email.subject,
        body: email.body,
        date: email.date,
      );
    }
  }

  Future<void> _indexPhotos() async {
    final photos = await _local.recentPhotos(limit: 100);
    for (final photo in photos) {
      // Each photo should already have a label from SmolVLM
      // If not, queue it for labeling
      if (photo.label != null) {
        await _rag.indexPhoto(
          path: photo.path,
          label: photo.label!,
          date: photo.date,
        );
      }
    }
  }

  Future<void> _indexFiles() async {
    final files = await _local.searchFiles('');
    for (final file in files) {
      // TODO: Extract text from PDFs and docs
      // For now, index the filename and path
      await _rag.indexFile(
        fileName: file.name,
        textContent: file.name,
        modified: file.modified,
      );
    }
  }

  Future<void> _indexCalendar() async {
    final now = DateTime.now();
    final events = await _local.getEvents(
      now.subtract(const Duration(days: 7)),
      now.add(const Duration(days: 7)),
    );
    for (final event in events) {
      await _rag.index(
        content: '${event.title} at ${event.location ?? "no location"} on ${event.start}',
        source: 'calendar',
        sourceId: event.title,
        timestamp: event.start,
      );
    }
  }
}
