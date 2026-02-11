import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_llama/flutter_llama.dart';

/// On-device RAG v2 — Hybrid Search Engine.
///
/// Combines vector cosine similarity (semantic) with BM25 keyword scoring
/// for dramatically better retrieval. Everything stays on-device.
///
/// Improvements over v1:
/// - Hybrid search: α·cosine + (1-α)·BM25
/// - Sentence-aware chunking with overlap
/// - Pre-computed term frequencies for instant BM25
/// - Content fingerprint tracking for smart re-indexing
/// - Stopword filtering for cleaner keyword matching
/// - Automatic recency boost for recent documents
/// - Source-type boost (files score higher for file queries)

class RAGEngine {
  static final RAGEngine _instance = RAGEngine._();
  factory RAGEngine() => _instance;
  RAGEngine._();

  static const _indexFileName = 'rag_index_v2.json';
  static const _fingerprintFileName = 'rag_fingerprints.json';
  static const _maxEntries = 10000;

  // ── Hybrid search tuning ──
  // Weight balance: 0.0 = pure BM25, 1.0 = pure vector
  static const _vectorWeight = 0.55;
  static const _bm25Weight = 0.45;

  // BM25 parameters (Okapi BM25)
  static const _k1 = 1.2;
  static const _b = 0.75;

  // Recency boost: recent docs within this window get a bonus
  static const _recencyWindowDays = 7;
  static const _recencyBoost = 0.05;

  // ── Storage ──
  final List<_VectorEntry> _store = [];

  /// Content fingerprints: sourceId → "mtime_ms:size" for change detection.
  final Map<String, String> _fingerprints = {};

  /// Aggregate document frequency: term → number of entries containing it.
  final Map<String, int> _docFreqs = {};

  /// Total term count across all entries (for avgDocLen).
  int _totalTerms = 0;

  bool _isInitialized = false;
  bool _isIndexing = false;
  bool _dirty = false;
  int _embeddingDim = 0;

  bool get isInitialized => _isInitialized;
  bool get isIndexing => _isIndexing;
  int get entryCount => _store.length;

  /// Initialize the RAG engine. Call once on app start.
  Future<void> init() async {
    if (_isInitialized) return;

    await _loadFromDisk();
    _rebuildBM25Stats();

    try {
      _embeddingDim = await FlutterLlama.instance.getEmbeddingDim();
      if (kDebugMode) {
        print('[RAG] Init: ${_store.length} entries, dim=$_embeddingDim, '
            '${_fingerprints.length} fingerprints');
      }
    } catch (e) {
      if (kDebugMode) print('[RAG] Model not loaded — keyword-only mode');
    }

    _isInitialized = true;
  }

  // ══════════════════════════════════════════
  // INDEXING
  // ══════════════════════════════════════════

  /// Index content with optional fingerprint for change detection.
  /// If [fingerprint] is provided and matches the stored one, indexing is
  /// skipped (content unchanged).
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
      final existing = _fingerprints[sourceId];
      if (existing == fingerprint) return false; // Unchanged
      _fingerprints[sourceId] = fingerprint;

      // Remove old entries for this sourceId (re-index)
      _store.removeWhere((e) => e.sourceId == sourceId && e.source == source);
    } else {
      // No fingerprint — de-dup by sourceId
      if (_store.any((e) => e.sourceId == sourceId && e.source == source)) {
        return false;
      }
    }

    final chunks = _chunkSentences(content);
    _isIndexing = true;

    for (int i = 0; i < chunks.length; i++) {
      final chunk = chunks[i];
      final vector = await _embed(chunk);
      final termFreqs = _computeTermFreqs(chunk);

      _store.add(_VectorEntry(
        text: chunk,
        vector: vector,
        termFreqs: termFreqs,
        termCount: termFreqs.values.fold(0, (a, b) => a + b),
        source: source,
        sourceId: sourceId,
        chunkIndex: i,
        timestamp: timestamp ?? DateTime.now(),
      ));

      // Update aggregate doc-freq counts
      for (final term in termFreqs.keys) {
        _docFreqs[term] = (_docFreqs[term] ?? 0) + 1;
      }
      _totalTerms += termFreqs.values.fold(0, (a, b) => a + b);
    }

    // Evict oldest entries if over limit
    _evictIfNeeded();

    _dirty = true;
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
  /// Returns true if content was actually indexed (new or changed).
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
  // HYBRID RETRIEVAL
  // ══════════════════════════════════════════

  /// Hybrid search: combines vector similarity with BM25 keyword scoring.
  /// Optionally boost results matching [sourceHint] type (e.g. "file").
  Future<List<RAGResult>> search(
    String query, {
    int topK = 5,
    double minScore = 0.15,
    String? sourceHint,
  }) async {
    if (_store.isEmpty) return [];

    final queryVector = await _embed(query);
    final queryTerms = _tokenize(query);
    final hasVectors = queryVector.isNotEmpty &&
        _store.any((e) => e.vector.isNotEmpty && e.vector.length == queryVector.length);
    final now = DateTime.now();

    // ── Score all entries ──
    double maxBM25 = 0;
    double maxCosine = 0;

    // First pass: compute raw scores
    final rawScores = <_RawScore>[];
    for (int i = 0; i < _store.length; i++) {
      final entry = _store[i];

      // BM25 score
      final bm25 = _bm25Score(queryTerms, entry);

      // Cosine similarity (if vectors available)
      double cosine = 0;
      if (hasVectors && entry.vector.isNotEmpty &&
          entry.vector.length == queryVector.length) {
        cosine = _cosineSimilarity(queryVector, entry.vector);
      }

      if (bm25 > maxBM25) maxBM25 = bm25;
      if (cosine > maxCosine) maxCosine = cosine;

      rawScores.add(_RawScore(
        index: i,
        bm25: bm25,
        cosine: cosine,
      ));
    }

    // ── Normalize and combine ──
    final results = <RAGResult>[];
    for (final raw in rawScores) {
      final entry = _store[raw.index];

      // Normalize to [0, 1]
      final normBM25 = maxBM25 > 0 ? raw.bm25 / maxBM25 : 0.0;
      final normCosine = maxCosine > 0 ? raw.cosine / maxCosine : 0.0;

      double score;
      if (hasVectors) {
        score = _vectorWeight * normCosine + _bm25Weight * normBM25;
      } else {
        // Pure keyword mode
        score = normBM25;
      }

      // Recency boost: docs from the last week get a small bonus
      final ageDays = now.difference(entry.timestamp).inDays;
      if (ageDays <= _recencyWindowDays) {
        score += _recencyBoost * (1 - ageDays / _recencyWindowDays);
      }

      // Source-type boost when orchestrator tells us the intent
      if (sourceHint != null && entry.source == sourceHint) {
        score += 0.05;
      }

      if (score < minScore) continue;

      results.add(RAGResult(
        text: entry.text,
        source: entry.source,
        sourceId: entry.sourceId,
        score: score,
        timestamp: entry.timestamp,
      ));
    }

    // Sort descending, deduplicate by sourceId (keep best chunk)
    results.sort((a, b) => b.score.compareTo(a.score));
    final seen = <String>{};
    final deduped = <RAGResult>[];
    for (final r in results) {
      if (seen.add(r.sourceId) || r.source == 'chat') {
        deduped.add(r);
      }
      if (deduped.length >= topK) break;
    }

    return deduped;
  }

  /// Build a context string from RAG results for the LLM prompt.
  /// Optionally pass [sourceHint] to boost relevant source types.
  Future<String> getContext(String query, {String? sourceHint}) async {
    final results = await search(query, sourceHint: sourceHint);
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
    counts['_total'] = _store.length;
    counts['_fingerprints'] = _fingerprints.length;
    return counts;
  }

  /// Check if a sourceId has been indexed and whether its fingerprint matches.
  /// Returns: null = not indexed, true = indexed & unchanged, false = indexed & changed.
  bool? checkFingerprint(String sourceId, String fingerprint) {
    final stored = _fingerprints[sourceId];
    if (stored == null) return null;
    return stored == fingerprint;
  }

  /// Persist the current index to disk.
  Future<void> save() async {
    if (!_dirty) return;
    await _saveToDisk();
    _dirty = false;
  }

  /// Clear the entire index.
  Future<void> clear() async {
    _store.clear();
    _fingerprints.clear();
    _docFreqs.clear();
    _totalTerms = 0;
    _dirty = true;
    await save();
  }

  // ══════════════════════════════════════════
  // BM25 SCORING
  // ══════════════════════════════════════════

  /// Okapi BM25 score for a query against one document entry.
  double _bm25Score(List<String> queryTerms, _VectorEntry entry) {
    if (queryTerms.isEmpty || entry.termFreqs.isEmpty) return 0;

    final n = _store.length;
    final dl = entry.termCount.toDouble();
    final avgDl = _store.isEmpty ? 1.0 : _totalTerms / _store.length;
    double score = 0;

    for (final term in queryTerms) {
      final df = _docFreqs[term] ?? 0;
      if (df == 0) continue;

      final tf = entry.termFreqs[term] ?? 0;
      if (tf == 0) continue;

      // IDF component: log((N - df + 0.5) / (df + 0.5) + 1)
      final idf = log((n - df + 0.5) / (df + 0.5) + 1);

      // TF normalization with length penalty
      final tfNorm = (tf * (_k1 + 1)) / (tf + _k1 * (1 - _b + _b * dl / avgDl));

      score += idf * tfNorm;
    }

    return score;
  }

  /// Rebuild BM25 aggregate stats from _store (called after load).
  void _rebuildBM25Stats() {
    _docFreqs.clear();
    _totalTerms = 0;

    for (final entry in _store) {
      _totalTerms += entry.termCount;
      for (final term in entry.termFreqs.keys) {
        _docFreqs[term] = (_docFreqs[term] ?? 0) + 1;
      }
    }
  }

  // ══════════════════════════════════════════
  // TEXT PROCESSING
  // ══════════════════════════════════════════

  /// Sentence-aware chunking with overlap for better semantic boundaries.
  /// Splits on sentence endings, then groups into chunks of ~[maxChars].
  List<String> _chunkSentences(String text, {int maxChars = 800, int overlapSentences = 2}) {
    if (text.length <= maxChars) return [text];

    // Split by sentence-like boundaries
    final sentences = text
        .split(RegExp(r'(?<=[.!?\n])\s+'))
        .where((s) => s.trim().isNotEmpty)
        .toList();

    if (sentences.length <= 1) {
      // Fallback: split by words if no sentence boundaries
      return _chunkWords(text, chunkSize: 200, overlap: 50);
    }

    final chunks = <String>[];
    final buffer = StringBuffer();
    final recentSentences = <String>[];

    for (final sentence in sentences) {
      if (buffer.length + sentence.length > maxChars && buffer.length > 100) {
        chunks.add(buffer.toString().trim());
        buffer.clear();

        // Overlap: carry last N sentences forward
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

  /// Fallback word-based chunking when text has no sentence structure.
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

  /// Tokenize text into normalized terms, filtering stopwords.
  List<String> _tokenize(String text) {
    return text
        .toLowerCase()
        .replaceAll(RegExp(r'[^\w\s]'), ' ')
        .split(RegExp(r'\s+'))
        .where((w) => w.length > 2 && !_stopwords.contains(w))
        .toList();
  }

  /// Compute term frequency map for a text chunk.
  Map<String, int> _computeTermFreqs(String text) {
    final freqs = <String, int>{};
    for (final term in _tokenize(text)) {
      freqs[term] = (freqs[term] ?? 0) + 1;
    }
    return freqs;
  }

  // ══════════════════════════════════════════
  // EMBEDDINGS
  // ══════════════════════════════════════════

  /// Embed text via llama.cpp. Falls back to empty (keyword-only mode).
  Future<List<double>> _embed(String text) async {
    try {
      final llama = FlutterLlama.instance;
      if (!llama.isModelLoaded) return [];

      final embedding = await llama.getEmbedding(text);
      if (embedding != null && embedding.isNotEmpty) {
        _embeddingDim = embedding.length;
        return embedding;
      }
    } catch (e) {
      if (kDebugMode) {
        print('[RAG] Embed failed, keyword-only: $e');
      }
    }
    return [];
  }

  /// Cosine similarity between two L2-normalized vectors.
  double _cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length || a.isEmpty) return 0.0;

    double dot = 0, normA = 0, normB = 0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    final denom = sqrt(normA) * sqrt(normB);
    return denom == 0 ? 0.0 : dot / denom;
  }

  // ══════════════════════════════════════════
  // PERSISTENCE — JSON file storage
  // ══════════════════════════════════════════

  Future<void> _loadFromDisk() async {
    try {
      final dir = await getApplicationDocumentsDirectory();

      // Load main index
      final file = File('${dir.path}/$_indexFileName');
      if (await file.exists()) {
        final raw = await file.readAsString();
        final data = jsonDecode(raw);

        // Support v2 format (object with entries array) and v1 (plain list)
        List entries;
        if (data is Map) {
          entries = data['entries'] as List? ?? [];
          // Load fingerprints inline
          final fp = data['fingerprints'] as Map<String, dynamic>?;
          if (fp != null) {
            _fingerprints.addAll(fp.cast<String, String>());
          }
        } else {
          entries = data as List;
        }

        _store.clear();
        for (final item in entries) {
          final map = item as Map<String, dynamic>;
          final text = map['text'] as String;

          // Rebuild term freqs from text (not persisted to save space)
          final termFreqs = _computeTermFreqs(text);

          _store.add(_VectorEntry(
            text: text,
            vector: (map['vector'] as List?)
                    ?.cast<num>()
                    .map((n) => n.toDouble())
                    .toList() ??
                [],
            termFreqs: termFreqs,
            termCount: termFreqs.values.fold(0, (a, b) => a + b),
            source: map['source'] as String,
            sourceId: map['sourceId'] as String,
            chunkIndex: map['chunkIdx'] as int? ?? 0,
            timestamp: DateTime.parse(map['timestamp'] as String),
          ));
        }

        if (kDebugMode) {
          print('[RAG] Loaded ${_store.length} entries from disk');
        }
      }

      // Load fingerprints (separate file for v1 compatibility)
      final fpFile = File('${dir.path}/$_fingerprintFileName');
      if (await fpFile.exists()) {
        final raw = await fpFile.readAsString();
        final map = jsonDecode(raw) as Map<String, dynamic>;
        _fingerprints.addAll(map.cast<String, String>());
      }
    } catch (e) {
      if (kDebugMode) print('[RAG] Load failed: $e');
    }
  }

  Future<void> _saveToDisk() async {
    try {
      final dir = await getApplicationDocumentsDirectory();

      // Save main index (v2 format)
      final file = File('${dir.path}/$_indexFileName');
      final data = {
        'version': 2,
        'savedAt': DateTime.now().toIso8601String(),
        'fingerprints': _fingerprints,
        'entries': _store.map((e) => {
          'text': e.text,
          'vector': e.vector,
          'source': e.source,
          'sourceId': e.sourceId,
          'chunkIdx': e.chunkIndex,
          'timestamp': e.timestamp.toIso8601String(),
          // termFreqs are NOT persisted — rebuilt on load from text
        }).toList(),
      };
      await file.writeAsString(jsonEncode(data));

      if (kDebugMode) {
        print('[RAG] Saved ${_store.length} entries, '
            '${_fingerprints.length} fingerprints');
      }
    } catch (e) {
      if (kDebugMode) print('[RAG] Save failed: $e');
    }
  }

  // ══════════════════════════════════════════
  // EVICTION
  // ══════════════════════════════════════════

  void _evictIfNeeded() {
    if (_store.length <= _maxEntries) return;

    // Sort by timestamp ascending, remove oldest
    _store.sort((a, b) => a.timestamp.compareTo(b.timestamp));
    final removeCount = _store.length - _maxEntries;
    _store.removeRange(0, removeCount);

    // Rebuild BM25 stats (indices shifted)
    _rebuildBM25Stats();

    if (kDebugMode) {
      print('[RAG] Evicted $removeCount oldest entries');
    }
  }

  // ══════════════════════════════════════════
  // STOPWORDS — filtered from BM25 to focus on meaning-carrying terms
  // ══════════════════════════════════════════

  static const _stopwords = <String>{
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
    'can', 'her', 'was', 'one', 'our', 'out', 'has', 'had',
    'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see',
    'way', 'who', 'did', 'get', 'let', 'say', 'she', 'too',
    'use', 'this', 'that', 'with', 'have', 'from', 'they',
    'been', 'said', 'each', 'make', 'like', 'long', 'look',
    'many', 'then', 'them', 'than', 'some', 'what', 'when',
    'will', 'more', 'into', 'over', 'such', 'take', 'also',
    'back', 'just', 'only', 'come', 'could', 'about', 'would',
    'there', 'their', 'which', 'other', 'after',
    'being', 'those', 'still', 'these', 'where', 'should',
    'very', 'much', 'here', 'does', 'were', 'your',
  };
}

/// Raw score pair used during hybrid scoring normalization.
class _RawScore {
  final int index;
  final double bm25;
  final double cosine;
  _RawScore({required this.index, required this.bm25, required this.cosine});
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

/// Internal vector entry with pre-computed BM25 term frequencies.
class _VectorEntry {
  final String text;
  final List<double> vector;
  final Map<String, int> termFreqs;  // Pre-computed for BM25
  final int termCount;               // Total terms in this chunk
  final String source;
  final String sourceId;
  final int chunkIndex;
  final DateTime timestamp;

  _VectorEntry({
    required this.text,
    required this.vector,
    required this.termFreqs,
    required this.termCount,
    required this.source,
    required this.sourceId,
    required this.chunkIndex,
    required this.timestamp,
  });
}
