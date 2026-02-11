import 'dart:math';

/// On-device RAG (Retrieval Augmented Generation) engine.
/// Indexes local data into vectors, retrieves relevant chunks at query time.
///
/// Stack:
/// - Embedding: all-MiniLM-L6-v2 (22MB ONNX) — runs on any phone
/// - Vector Store: ObjectBox with HNSW vector search
/// - Everything stays on-device. Zero network calls.

class RAGEngine {
  static final RAGEngine _instance = RAGEngine._();
  factory RAGEngine() => _instance;
  RAGEngine._();

  // TODO: Replace with actual ObjectBox store
  // late final Store _store;
  // late final Box<DocumentChunk> _chunkBox;

  /// In-memory store for development. Replace with ObjectBox in production.
  final List<_VectorEntry> _store = [];

  bool _isInitialized = false;
  final bool _isIndexing = false;

  bool get isInitialized => _isInitialized;
  bool get isIndexing => _isIndexing;

  /// Initialize the RAG engine. Call once on app start.
  Future<void> init() async {
    if (_isInitialized) return;

    // TODO: Load the MiniLM embedding model (ONNX format)
    // _embedder = OnnxRuntime.load('assets/models/minilm-l6-v2.onnx');

    // TODO: Open ObjectBox store
    // _store = await openStore();
    // _chunkBox = _store.box<DocumentChunk>();

    _isInitialized = true;
  }

  // ──────────────────────────────────────────
  // INDEXING — runs in background
  // ──────────────────────────────────────────

  /// Index a piece of content. Chunks it, embeds it, stores the vectors.
  Future<void> index({
    required String content,
    required String source,     // e.g. "email", "file", "photo"
    required String sourceId,   // e.g. email subject, file path
    DateTime? timestamp,
  }) async {
    final chunks = _chunk(content);

    for (final chunk in chunks) {
      final vector = await _embed(chunk);
      _store.add(_VectorEntry(
        text: chunk,
        vector: vector,
        source: source,
        sourceId: sourceId,
        timestamp: timestamp ?? DateTime.now(),
      ));
    }
  }

  /// Index an email.
  Future<void> indexEmail({
    required String from,
    required String subject,
    required String body,
    required DateTime date,
  }) async {
    final content = 'From: $from\nSubject: $subject\nDate: $date\n\n$body';
    await index(
      content: content,
      source: 'email',
      sourceId: subject,
      timestamp: date,
    );
  }

  /// Index a file's text content.
  Future<void> indexFile({
    required String fileName,
    required String textContent,
    required DateTime modified,
  }) async {
    await index(
      content: textContent,
      source: 'file',
      sourceId: fileName,
      timestamp: modified,
    );
  }

  /// Index a photo with its AI-generated label.
  Future<void> indexPhoto({
    required String path,
    required String label,
    required DateTime date,
  }) async {
    await index(
      content: 'Photo: $label. Location: $path',
      source: 'photo',
      sourceId: path,
      timestamp: date,
    );
  }

  // ──────────────────────────────────────────
  // RETRIEVAL — called when user asks a question
  // ──────────────────────────────────────────

  /// Find the most relevant chunks for a query.
  /// Returns up to [topK] results, sorted by relevance.
  Future<List<RAGResult>> search(String query, {int topK = 5}) async {
    final queryVector = await _embed(query);

    // Score every entry by cosine similarity
    final scored = _store.map((entry) {
      final score = _cosineSimilarity(queryVector, entry.vector);
      return RAGResult(
        text: entry.text,
        source: entry.source,
        sourceId: entry.sourceId,
        score: score,
        timestamp: entry.timestamp,
      );
    }).toList();

    // Sort by score descending, return top K
    scored.sort((a, b) => b.score.compareTo(a.score));
    return scored.take(topK).where((r) => r.score > 0.3).toList();
  }

  /// Build a context string from RAG results for the LLM prompt.
  Future<String> getContext(String query) async {
    final results = await search(query);
    if (results.isEmpty) return '';

    return results
        .map((r) => '[${r.source}] ${r.text}')
        .join('\n---\n');
  }

  /// Get stats about what's been indexed.
  Map<String, int> get stats {
    final counts = <String, int>{};
    for (final entry in _store) {
      counts[entry.source] = (counts[entry.source] ?? 0) + 1;
    }
    return counts;
  }

  // ──────────────────────────────────────────
  // INTERNALS
  // ──────────────────────────────────────────

  /// Split text into chunks of ~200 words with overlap.
  List<String> _chunk(String text, {int chunkSize = 200, int overlap = 50}) {
    final words = text.split(RegExp(r'\s+'));
    if (words.length <= chunkSize) return [text];

    final chunks = <String>[];
    int start = 0;
    while (start < words.length) {
      final end = (start + chunkSize).clamp(0, words.length);
      chunks.add(words.sublist(start, end).join(' '));
      start += chunkSize - overlap;
    }
    return chunks;
  }

  /// Embed text into a vector using MiniLM.
  Future<List<double>> _embed(String text) async {
    // TODO: Replace with actual ONNX inference
    // final input = _tokenizer.encode(text);
    // final output = await _model.run(input);
    // return output.toList();

    // Stub: random vector for development
    final rng = Random(text.hashCode);
    return List.generate(384, (_) => rng.nextDouble() * 2 - 1);
  }

  /// Cosine similarity between two vectors.
  double _cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) return 0.0;

    double dot = 0, normA = 0, normB = 0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    final denom = sqrt(normA) * sqrt(normB);
    return denom == 0 ? 0.0 : dot / denom;
  }
}

/// A single RAG search result.
class RAGResult {
  final String text;
  final String source;
  final String sourceId;
  final double score;
  final DateTime timestamp;

  RAGResult({
    required this.text,
    required this.source,
    required this.sourceId,
    required this.score,
    required this.timestamp,
  });
}

/// Internal vector entry in the store.
class _VectorEntry {
  final String text;
  final List<double> vector;
  final String source;
  final String sourceId;
  final DateTime timestamp;

  _VectorEntry({
    required this.text,
    required this.vector,
    required this.source,
    required this.sourceId,
    required this.timestamp,
  });
}
