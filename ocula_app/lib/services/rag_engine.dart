import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter_llama/flutter_llama.dart';
import 'ocula_db.dart';

/// On-device RAG v3 — SQLite-backed Hybrid Search Engine.
///
/// Combines FTS5 BM25 (keyword) with vector cosine similarity (semantic)
/// for dramatically better retrieval. Everything stays on-device.
///
/// v3 improvements over v2:
/// - SQLite-backed: no more loading entire JSON into RAM
/// - FTS5 with porter stemmer: proper BM25 scoring built-in
/// - Vectors stored as BLOB: Float32List encoded, ~4× smaller than JSON
/// - Incremental: each index() is an INSERT, not a full file rewrite
/// - Knowledge graph integration for entity-relationship context
/// - Auto-migrates from rag_index_v2.json on first run

class RAGEngine {
  static final RAGEngine _instance = RAGEngine._();
  factory RAGEngine() => _instance;
  RAGEngine._();

  static const _maxEntries = 10000;

  // ── Hybrid search tuning ──
  static const _vectorWeight = 0.55;

  final OculaDB _db = OculaDB();

  bool _isInitialized = false;
  bool _isIndexing = false;
  int _embeddingDim = 0;

  bool get isInitialized => _isInitialized;
  bool get isIndexing => _isIndexing;

  /// Entry count (reads from DB).
  Future<int> get entryCountAsync => _db.chunkCount;

  /// Synchronous entry count approximation (for backward compat).
  /// Returns 0 until first async call populates it.
  int _cachedCount = 0;
  int get entryCount => _cachedCount;

  /// Initialize the RAG engine. Call once on app start.
  Future<void> init() async {
    if (_isInitialized) return;

    // Touch the DB to trigger open + migration from JSON
    final d = await _db.db;
    _cachedCount = await _db.chunkCount;

    try {
      // Prefer dedicated embedding model dimension
      final llama = FlutterLlama.instance;
      if (llama.isEmbeddingModelLoaded) {
        _embeddingDim = await llama.getEmbeddingModelDim();
      } else {
        _embeddingDim = await llama.getEmbeddingDim();
      }
      if (kDebugMode) {
        print('[RAG] Init: $_cachedCount entries in DB, dim=$_embeddingDim');
      }
    } catch (e) {
      if (kDebugMode) print('[RAG] Model not loaded — keyword-only mode');
    }

    _isInitialized = true;
  }

  /// Check if existing embeddings need re-indexing (e.g. embedding model changed).
  /// Call after loading the embedding model for the first time.
  Future<bool> needsReindex() async {
    final meta = await _db.getMeta('embedding_model');
    final llama = FlutterLlama.instance;
    final currentModel = llama.isEmbeddingModelLoaded ? 'minilm-v2' : 'generative';
    if (meta != currentModel && _cachedCount > 0) {
      return true; // Embedding model changed — vectors are incompatible
    }
    return false;
  }

  /// Mark the current embedding model version in metadata.
  Future<void> markEmbeddingModel() async {
    final llama = FlutterLlama.instance;
    final model = llama.isEmbeddingModelLoaded ? 'minilm-v2' : 'generative';
    await _db.setMeta('embedding_model', model);
  }

  // ══════════════════════════════════════════
  // INDEXING
  // ══════════════════════════════════════════

  /// Index content with optional fingerprint for change detection.
  Future<bool> index({
    required String content,
    required String source,
    required String sourceId,
    DateTime? timestamp,
    String? fingerprint,
  }) async {
    if (content.trim().isEmpty) return false;

    // ── Change detection ──
    if (fingerprint != null) {
      final existing = await _db.getFingerprint(sourceId);
      if (existing == fingerprint) return false; // Unchanged
      await _db.setFingerprint(sourceId, fingerprint);

      // Remove old entries for this sourceId (re-index)
      await _db.deleteChunksBySource(sourceId);
    } else {
      // No fingerprint — de-dup by sourceId
      if (await _db.hasChunksForSource(sourceId)) {
        return false;
      }
    }

    final chunks = _chunkSentences(content);
    _isIndexing = true;

    for (int i = 0; i < chunks.length; i++) {
      final chunk = chunks[i];
      final vector = await _embed(chunk);

      await _db.insertChunk(
        text: chunk,
        vector: vector,
        source: source,
        sourceId: sourceId,
        chunkIdx: i,
        timestamp: timestamp,
      );
    }

    // Evict oldest entries if over limit
    await _db.trimChunks(maxEntries: _maxEntries);

    _cachedCount = await _db.chunkCount;
    _isIndexing = false;
    return true;
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
      sourceId: 'email:$subject:${date.toIso8601String()}',
      timestamp: date,
    );
  }

  /// Index a file. Pass [fingerprint] ("mtime_ms:size") for incremental updates.
  Future<bool> indexFile({
    required String fileName,
    required String textContent,
    required DateTime modified,
    String? fingerprint,
  }) async {
    return await index(
      content: 'File: $fileName\n\n$textContent',
      source: 'file',
      sourceId: 'file:$fileName',
      timestamp: modified,
      fingerprint: fingerprint,
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
      sourceId: 'photo:$path',
      timestamp: date,
    );
  }

  /// Index a chat conversation turn.
  Future<void> indexChat({
    required String userMessage,
    required String assistantResponse,
    DateTime? timestamp,
  }) async {
    final content = 'User: $userMessage\nAssistant: $assistantResponse';
    await index(
      content: content,
      source: 'chat',
      sourceId: 'chat:${timestamp?.toIso8601String() ?? DateTime.now().toIso8601String()}',
      timestamp: timestamp,
    );
  }

  // ══════════════════════════════════════════
  // HYBRID RETRIEVAL — delegates to OculaDB
  // ══════════════════════════════════════════

  /// Hybrid search: FTS5 BM25 + vector cosine similarity.
  Future<List<RAGResult>> search(
    String query, {
    int topK = 5,
    double minScore = 0.15,
    String? sourceHint,
  }) async {
    final queryVector = await _embed(query);

    final results = await _db.hybridSearch(
      query,
      queryVector,
      topK: topK,
      minScore: minScore,
      sourceHint: sourceHint,
      vectorWeight: _vectorWeight,
    );

    return results.map((r) => RAGResult(
      text: r.text,
      source: r.source,
      sourceId: r.sourceId,
      score: r.score,
      timestamp: r.timestamp,
    )).toList();
  }

  /// Build a context string from RAG results for the LLM prompt.
  /// Formats each source type clearly so the model cites specific details.
  Future<String> getContext(String query, {String? sourceHint}) async {
    final results = await search(query, sourceHint: sourceHint);
    if (results.isEmpty) return '';

    return results.map((r) {
      final label = _sourceLabel(r.source);
      return '$label: ${r.text}';
    }).join('\n\n');
  }

  String _sourceLabel(String source) {
    switch (source) {
      case 'contact': return 'CONTACT';
      case 'calendar': return 'CALENDAR EVENT';
      case 'photo': return 'PHOTO';
      case 'file': return 'FILE';
      case 'email': return 'EMAIL';
      case 'chat': return 'PREVIOUS CONVERSATION';
      default: return source.toUpperCase();
    }
  }

  /// Get stats about what's been indexed.
  Future<Map<String, int>> get statsAsync async {
    return _db.stats();
  }

  /// Synchronous stats stub for backward compat.
  Map<String, int> get stats => {'_total': _cachedCount};

  /// Check if a sourceId has been indexed and whether its fingerprint matches.
  /// Returns: null = not indexed, true = indexed & unchanged, false = indexed & changed.
  Future<bool?> checkFingerprintAsync(String sourceId, String fingerprint) async {
    final stored = await _db.getFingerprint(sourceId);
    if (stored == null) return null;
    return stored == fingerprint;
  }

  /// Synchronous wrapper (backward compat) — always returns null (use async version).
  bool? checkFingerprint(String sourceId, String fingerprint) {
    // Deprecated: use checkFingerprintAsync instead.
    // Returns null to force re-index (safe fallback).
    return null;
  }

  /// No-op — SQLite auto-persists. Kept for API compatibility.
  Future<void> save() async {
    // SQLite transactions are auto-committed. Nothing to do.
  }

  /// Clear the entire index.
  Future<void> clear() async {
    final d = await _db.db;
    await d.delete('rag_chunks');
    _cachedCount = 0;
  }

  // ══════════════════════════════════════════
  // TEXT PROCESSING
  // ══════════════════════════════════════════

  /// Sentence-aware chunking with overlap.
  List<String> _chunkSentences(String text, {int maxChars = 800, int overlapSentences = 2}) {
    if (text.length <= maxChars) return [text];

    final sentences = text
        .split(RegExp(r'(?<=[.!?\n])\s+'))
        .where((s) => s.trim().isNotEmpty)
        .toList();

    if (sentences.length <= 1) {
      return _chunkWords(text, chunkSize: 200, overlap: 50);
    }

    final chunks = <String>[];
    final buffer = StringBuffer();
    final recentSentences = <String>[];

    for (final sentence in sentences) {
      if (buffer.length + sentence.length > maxChars && buffer.length > 100) {
        chunks.add(buffer.toString().trim());
        buffer.clear();

        final overlap = recentSentences.length > overlapSentences
            ? recentSentences.sublist(recentSentences.length - overlapSentences)
            : List<String>.from(recentSentences);
        recentSentences.clear();
        for (final s in overlap) {
          buffer.write(s);
          buffer.write(' ');
          recentSentences.add(s);
        }
      }
      buffer.write(sentence);
      buffer.write(' ');
      recentSentences.add(sentence);
    }

    if (buffer.isNotEmpty) {
      chunks.add(buffer.toString().trim());
    }

    return chunks.isEmpty ? [text] : chunks;
  }

  List<String> _chunkWords(String text, {int chunkSize = 200, int overlap = 50}) {
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

  // ══════════════════════════════════════════
  // EMBEDDINGS
  // ══════════════════════════════════════════

  /// Embed text via llama.cpp.
  ///
  /// Priority:
  /// 1. Dedicated embedding model (all-MiniLM-L6-v2) — high-quality sentence embeddings
  /// 2. Generative model fallback — lower quality but better than nothing
  /// 3. Empty list — keyword-only mode (FTS5 BM25 still works)
  Future<List<double>> _embed(String text) async {
    try {
      final llama = FlutterLlama.instance;

      // Prefer dedicated embedding model (trained for sentence similarity)
      if (llama.isEmbeddingModelLoaded) {
        final embedding = await llama.getEmbeddingV2(text);
        if (embedding != null && embedding.isNotEmpty) {
          _embeddingDim = embedding.length;
          return embedding;
        }
      }

      // Fallback: generative model embeddings (lower quality)
      if (llama.isModelLoaded) {
        final embedding = await llama.getEmbedding(text);
        if (embedding != null && embedding.isNotEmpty) {
          _embeddingDim = embedding.length;
          return embedding;
        }
      }
    } catch (e) {
      if (kDebugMode) {
        print('[RAG] Embed failed, keyword-only: $e');
      }
    }
    return [];
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
