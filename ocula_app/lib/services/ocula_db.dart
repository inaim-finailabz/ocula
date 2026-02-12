import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart' as p;

/// Unified on-device database for Ocula.
///
/// Replaces the old JSON files (episodic_memory.json, rag_index_v2.json)
/// with a single SQLite database. Zero additional binary — iOS and Android
/// ship with SQLite built in.
///
/// Tables:
///   chat_turns    — conversation history (replaces EpisodicMemory JSON)
///   rag_chunks    — text chunks with vectors + BM25 data
///   rag_meta      — fingerprints + index metadata
///   knowledge     — (subject, predicate, object) triples for graph search
///   assets        — indexed phone assets tracking
class OculaDB {
  static final OculaDB _instance = OculaDB._();
  factory OculaDB() => _instance;
  OculaDB._();

  static const _dbName = 'ocula.db';
  static const _dbVersion = 1;

  Database? _db;
  bool _migrated = false;

  /// Get the database, creating/opening it if needed.
  Future<Database> get db async {
    if (_db != null && _db!.isOpen) return _db!;
    _db = await _open();
    if (!_migrated) {
      await _migrateFromJson();
      _migrated = true;
    }
    return _db!;
  }

  Future<Database> _open() async {
    final dir = await getApplicationDocumentsDirectory();
    final path = p.join(dir.path, _dbName);

    return openDatabase(
      path,
      version: _dbVersion,
      onCreate: _onCreate,
      onUpgrade: _onUpgrade,
    );
  }

  Future<void> _onCreate(Database db, int version) async {
    final batch = db.batch();

    // ── Chat turns (replaces episodic_memory.json) ──
    batch.execute('''
      CREATE TABLE chat_turns (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        query      TEXT NOT NULL,
        response   TEXT NOT NULL,
        intent     TEXT DEFAULT 'chat',
        model_used TEXT DEFAULT 'free',
        steps      TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      )
    ''');
    batch.execute('CREATE INDEX idx_chat_created ON chat_turns(created_at DESC)');

    // ── RAG chunks (replaces rag_index_v2.json entries) ──
    batch.execute('''
      CREATE TABLE rag_chunks (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        text       TEXT NOT NULL,
        vector     BLOB,
        source     TEXT NOT NULL,
        source_id  TEXT NOT NULL,
        chunk_idx  INTEGER DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      )
    ''');
    batch.execute('CREATE INDEX idx_rag_source_id ON rag_chunks(source_id)');
    batch.execute('CREATE INDEX idx_rag_source ON rag_chunks(source)');

    // Full-text search on chunk text (BM25 built-in)
    batch.execute('''
      CREATE VIRTUAL TABLE rag_fts USING fts5(
        text,
        source,
        source_id,
        content='rag_chunks',
        content_rowid='id',
        tokenize='porter unicode61'
      )
    ''');

    // Triggers to keep FTS in sync with rag_chunks
    batch.execute('''
      CREATE TRIGGER rag_fts_insert AFTER INSERT ON rag_chunks BEGIN
        INSERT INTO rag_fts(rowid, text, source, source_id)
          VALUES (new.id, new.text, new.source, new.source_id);
      END
    ''');
    batch.execute('''
      CREATE TRIGGER rag_fts_delete AFTER DELETE ON rag_chunks BEGIN
        INSERT INTO rag_fts(rag_fts, rowid, text, source, source_id)
          VALUES ('delete', old.id, old.text, old.source, old.source_id);
      END
    ''');
    batch.execute('''
      CREATE TRIGGER rag_fts_update AFTER UPDATE ON rag_chunks BEGIN
        INSERT INTO rag_fts(rag_fts, rowid, text, source, source_id)
          VALUES ('delete', old.id, old.text, old.source, old.source_id);
        INSERT INTO rag_fts(rowid, text, source, source_id)
          VALUES (new.id, new.text, new.source, new.source_id);
      END
    ''');

    // ── RAG metadata (fingerprints, stats) ──
    batch.execute('''
      CREATE TABLE rag_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
      )
    ''');

    // ── Knowledge graph (subject → predicate → object triples) ──
    batch.execute('''
      CREATE TABLE knowledge (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        subject    TEXT NOT NULL,
        predicate  TEXT NOT NULL,
        object     TEXT NOT NULL,
        source     TEXT,
        confidence REAL DEFAULT 1.0,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      )
    ''');
    batch.execute('CREATE INDEX idx_kg_subject ON knowledge(subject)');
    batch.execute('CREATE INDEX idx_kg_object ON knowledge(object)');
    batch.execute('CREATE INDEX idx_kg_predicate ON knowledge(predicate)');

    // ── Asset tracking ──
    batch.execute('''
      CREATE TABLE assets (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        path         TEXT NOT NULL UNIQUE,
        type         TEXT NOT NULL,
        fingerprint  TEXT,
        last_indexed TEXT,
        metadata     TEXT
      )
    ''');
    batch.execute('CREATE INDEX idx_assets_type ON assets(type)');

    await batch.commit(noResult: true);
  }

  Future<void> _onUpgrade(Database db, int oldVersion, int newVersion) async {
    // Future migrations go here
  }

  // ════════════════════════════════════════════════════════════════════
  // CHAT TURNS
  // ════════════════════════════════════════════════════════════════════

  /// Log a completed chat turn.
  Future<int> logChat({
    required String query,
    required String response,
    String intent = 'chat',
    String modelUsed = 'free',
    List<String>? steps,
  }) async {
    final d = await db;
    return d.insert('chat_turns', {
      'query': query,
      'response': response.length > 2000
          ? response.substring(0, 2000)
          : response,
      'intent': intent,
      'model_used': modelUsed,
      'steps': steps != null ? jsonEncode(steps) : null,
      'created_at': DateTime.now().toIso8601String(),
    });
  }

  /// Recall recent conversations matching keywords.
  /// Uses FTS if terms are long enough, else LIKE fallback.
  Future<String> recallChat(String query, {int limit = 3}) async {
    final d = await db;
    final keywords = query
        .toLowerCase()
        .split(RegExp(r'\s+'))
        .where((w) => w.length > 3)
        .toList();

    if (keywords.isEmpty) return '';

    // Build a simple OR query
    final where = keywords
        .map((_) => "(query LIKE ? OR response LIKE ?)")
        .join(' OR ');
    final args = keywords
        .expand((k) => ['%$k%', '%$k%'])
        .toList();

    final rows = await d.query(
      'chat_turns',
      where: where,
      whereArgs: args,
      orderBy: 'created_at DESC',
      limit: limit,
    );

    if (rows.isEmpty) return '';

    return rows.map((r) {
      return 'On ${r['created_at']}: User asked "${r['query']}" → '
          '${r['response']}';
    }).join('\n');
  }

  /// Get total chat turn count.
  Future<int> get chatCount async {
    final d = await db;
    final result = await d.rawQuery('SELECT COUNT(*) as c FROM chat_turns');
    return Sqflite.firstIntValue(result) ?? 0;
  }

  /// Trim chat_turns to the newest [maxEntries].
  Future<void> trimChat({int maxEntries = 2000}) async {
    final d = await db;
    await d.rawDelete('''
      DELETE FROM chat_turns WHERE id NOT IN (
        SELECT id FROM chat_turns ORDER BY created_at DESC LIMIT ?
      )
    ''', [maxEntries]);
  }

  /// Clear all chat history.
  Future<void> clearChat() async {
    final d = await db;
    await d.delete('chat_turns');
  }

  // ════════════════════════════════════════════════════════════════════
  // RAG CHUNKS — vector + FTS hybrid
  // ════════════════════════════════════════════════════════════════════

  /// Insert a chunk with its embedding vector.
  Future<int> insertChunk({
    required String text,
    required List<double> vector,
    required String source,
    required String sourceId,
    int chunkIdx = 0,
    DateTime? timestamp,
  }) async {
    final d = await db;
    return d.insert('rag_chunks', {
      'text': text,
      'vector': _encodeVector(vector),
      'source': source,
      'source_id': sourceId,
      'chunk_idx': chunkIdx,
      'created_at': (timestamp ?? DateTime.now()).toIso8601String(),
    });
  }

  /// Remove all chunks for a source_id (for re-indexing).
  Future<int> deleteChunksBySource(String sourceId) async {
    final d = await db;
    return d.delete('rag_chunks', where: 'source_id = ?', whereArgs: [sourceId]);
  }

  /// Check if a source_id already exists.
  Future<bool> hasChunksForSource(String sourceId) async {
    final d = await db;
    final result = await d.rawQuery(
      'SELECT 1 FROM rag_chunks WHERE source_id = ? LIMIT 1',
      [sourceId],
    );
    return result.isNotEmpty;
  }

  /// Hybrid search: FTS5 BM25 + vector cosine similarity.
  ///
  /// Returns top [topK] results scored by:
  ///   α·cosine(query_vec, chunk_vec) + (1−α)·bm25_normalized
  ///   + recency_boost + source_boost
  Future<List<RagSearchResult>> hybridSearch(
    String query,
    List<double> queryVector, {
    int topK = 5,
    double minScore = 0.1,
    String? sourceHint,
    double vectorWeight = 0.55,
  }) async {
    final d = await db;
    final now = DateTime.now();
    final hasVector = queryVector.isNotEmpty;

    // Step 1: FTS5 BM25 candidates (fast, indexed)
    final ftsResults = <int, double>{};
    try {
      final queryTerms = query
          .replaceAll(RegExp(r'[^\w\s]'), ' ')
          .split(RegExp(r'\s+'))
          .where((w) => w.length > 2)
          .join(' OR ');

      if (queryTerms.isNotEmpty) {
        final rows = await d.rawQuery('''
          SELECT rowid, bm25(rag_fts) as score
          FROM rag_fts
          WHERE rag_fts MATCH ?
          ORDER BY score
          LIMIT 200
        ''', [queryTerms]);

        for (final row in rows) {
          final id = row['rowid'] as int;
          final score = (row['score'] as num).toDouble().abs();
          ftsResults[id] = score;
        }
      }
    } catch (_) {
      // FTS query may fail on special chars — fall through to vector-only
    }

    // Step 2: If we have vectors, scan all chunks for cosine similarity
    // (For large DBs, this could be batched. For <10K entries it's instant.)
    final vectorScores = <int, double>{};
    if (hasVector) {
      final rows = await d.query('rag_chunks', columns: ['id', 'vector']);
      for (final row in rows) {
        final blob = row['vector'] as Uint8List?;
        if (blob == null || blob.isEmpty) continue;
        final chunkVec = _decodeVector(blob);
        if (chunkVec.length != queryVector.length) continue;
        final cos = _cosine(queryVector, chunkVec);
        if (cos > 0.05) {
          vectorScores[row['id'] as int] = cos;
        }
      }
    }

    // Step 3: Merge candidates
    final allIds = {...ftsResults.keys, ...vectorScores.keys};
    if (allIds.isEmpty) return [];

    // Normalize BM25 scores
    final maxBM25 = ftsResults.values.isEmpty
        ? 1.0
        : ftsResults.values.reduce(max);
    final maxCos = vectorScores.values.isEmpty
        ? 1.0
        : vectorScores.values.reduce(max);

    // Fetch chunk metadata for all candidates
    final idList = allIds.toList();
    final placeholders = List.filled(idList.length, '?').join(',');
    final chunks = await d.rawQuery(
      'SELECT id, text, source, source_id, created_at FROM rag_chunks '
      'WHERE id IN ($placeholders)',
      idList,
    );

    final chunkMap = {for (final c in chunks) c['id'] as int: c};

    // Score and rank
    final results = <RagSearchResult>[];
    for (final id in allIds) {
      final chunk = chunkMap[id];
      if (chunk == null) continue;

      final normBM25 = ftsResults.containsKey(id)
          ? ftsResults[id]! / maxBM25
          : 0.0;
      final normCos = vectorScores.containsKey(id)
          ? vectorScores[id]! / maxCos
          : 0.0;

      double score;
      if (hasVector && vectorScores.isNotEmpty) {
        score = vectorWeight * normCos + (1 - vectorWeight) * normBM25;
      } else {
        score = normBM25;
      }

      // Recency boost (7 days)
      final created = DateTime.parse(chunk['created_at'] as String);
      final ageDays = now.difference(created).inDays;
      if (ageDays <= 7) {
        score += 0.05 * (1 - ageDays / 7);
      }

      // Source type boost
      if (sourceHint != null && chunk['source'] == sourceHint) {
        score += 0.05;
      }

      if (score < minScore) continue;

      results.add(RagSearchResult(
        id: id,
        text: chunk['text'] as String,
        source: chunk['source'] as String,
        sourceId: chunk['source_id'] as String,
        score: score,
        timestamp: created,
      ));
    }

    results.sort((a, b) => b.score.compareTo(a.score));

    // Deduplicate by sourceId (keep best chunk per source)
    final seen = <String>{};
    final deduped = <RagSearchResult>[];
    for (final r in results) {
      if (seen.add(r.sourceId) || r.source == 'chat') {
        deduped.add(r);
      }
      if (deduped.length >= topK) break;
    }

    return deduped;
  }

  /// Get total chunk count.
  Future<int> get chunkCount async {
    final d = await db;
    final result = await d.rawQuery('SELECT COUNT(*) as c FROM rag_chunks');
    return Sqflite.firstIntValue(result) ?? 0;
  }

  /// Trim rag_chunks to newest [maxEntries].
  Future<void> trimChunks({int maxEntries = 10000}) async {
    final d = await db;
    await d.rawDelete('''
      DELETE FROM rag_chunks WHERE id NOT IN (
        SELECT id FROM rag_chunks ORDER BY created_at DESC LIMIT ?
      )
    ''', [maxEntries]);
  }

  // ════════════════════════════════════════════════════════════════════
  // RAG METADATA (fingerprints)
  // ════════════════════════════════════════════════════════════════════

  Future<void> setMeta(String key, String value) async {
    final d = await db;
    await d.insert('rag_meta', {'key': key, 'value': value},
        conflictAlgorithm: ConflictAlgorithm.replace);
  }

  Future<String?> getMeta(String key) async {
    final d = await db;
    final rows = await d.query('rag_meta', where: 'key = ?', whereArgs: [key]);
    return rows.isEmpty ? null : rows.first['value'] as String;
  }

  /// Store a fingerprint for change detection.
  Future<void> setFingerprint(String sourceId, String fingerprint) =>
      setMeta('fp:$sourceId', fingerprint);

  /// Get stored fingerprint. Returns null if not indexed.
  Future<String?> getFingerprint(String sourceId) =>
      getMeta('fp:$sourceId');

  // ════════════════════════════════════════════════════════════════════
  // KNOWLEDGE GRAPH — lightweight triple store
  // ════════════════════════════════════════════════════════════════════

  /// Add a (subject, predicate, object) triple.
  /// Skips if the same triple already exists.
  Future<int?> addTriple({
    required String subject,
    required String predicate,
    required String object,
    String? source,
    double confidence = 1.0,
  }) async {
    final d = await db;

    // Deduplicate
    final existing = await d.query('knowledge',
        where: 'subject = ? AND predicate = ? AND object = ?',
        whereArgs: [subject.toLowerCase(), predicate.toLowerCase(), object.toLowerCase()],
        limit: 1);
    if (existing.isNotEmpty) return null;

    return d.insert('knowledge', {
      'subject': subject.toLowerCase(),
      'predicate': predicate.toLowerCase(),
      'object': object.toLowerCase(),
      'source': source,
      'confidence': confidence,
      'created_at': DateTime.now().toIso8601String(),
    });
  }

  /// Query the graph: find all triples where subject or object matches.
  Future<List<KGTriple>> queryGraph(String entity, {int limit = 20}) async {
    final d = await db;
    final term = entity.toLowerCase();
    final rows = await d.query(
      'knowledge',
      where: 'subject = ? OR object = ?',
      whereArgs: [term, term],
      orderBy: 'confidence DESC, created_at DESC',
      limit: limit,
    );

    return rows.map((r) => KGTriple(
      subject: r['subject'] as String,
      predicate: r['predicate'] as String,
      object: r['object'] as String,
      source: r['source'] as String?,
      confidence: (r['confidence'] as num).toDouble(),
    )).toList();
  }

  /// Find entities related to a query via graph traversal (1-hop).
  /// Returns context strings like "user likes coffee", "user works at Acme".
  Future<String> graphContext(String query, {int limit = 5}) async {
    final terms = query
        .toLowerCase()
        .split(RegExp(r'\s+'))
        .where((w) => w.length > 2)
        .toList();

    if (terms.isEmpty) return '';

    final d = await db;
    final results = <KGTriple>[];

    for (final term in terms) {
      final rows = await d.query(
        'knowledge',
        where: 'subject LIKE ? OR object LIKE ?',
        whereArgs: ['%$term%', '%$term%'],
        orderBy: 'confidence DESC',
        limit: limit,
      );

      for (final r in rows) {
        results.add(KGTriple(
          subject: r['subject'] as String,
          predicate: r['predicate'] as String,
          object: r['object'] as String,
          source: r['source'] as String?,
          confidence: (r['confidence'] as num).toDouble(),
        ));
      }
    }

    if (results.isEmpty) return '';

    // Deduplicate
    final seen = <String>{};
    final unique = results.where((t) {
      final key = '${t.subject}|${t.predicate}|${t.object}';
      return seen.add(key);
    }).take(limit).toList();

    return unique
        .map((t) => '${t.subject} ${t.predicate} ${t.object}')
        .join('. ');
  }

  /// Extract and store knowledge triples from a chat turn.
  /// Uses simple heuristic patterns — no LLM call required.
  Future<void> extractKnowledge(String query, String response) async {
    final triples = _extractTriples(query, response);
    for (final t in triples) {
      await addTriple(
        subject: t.subject,
        predicate: t.predicate,
        object: t.object,
        source: 'chat',
      );
    }
  }

  /// Simple heuristic triple extraction from conversation.
  /// Patterns like "I am X", "I like X", "my name is X", etc.
  List<KGTriple> _extractTriples(String query, String response) {
    final triples = <KGTriple>[];
    final lower = query.toLowerCase().trim();

    // "I am X" / "I'm X"
    final iAmMatch = RegExp(r"i(?:'m| am)\s+(?:a |an )?(.+?)(?:\.|,|$)", caseSensitive: false)
        .firstMatch(lower);
    if (iAmMatch != null) {
      final obj = iAmMatch.group(1)?.trim();
      if (obj != null && obj.length > 1 && obj.length < 50) {
        triples.add(KGTriple(subject: 'user', predicate: 'is', object: obj));
      }
    }

    // "My name is X"
    final nameMatch = RegExp(r"my name is\s+(.+?)(?:\.|,|$)", caseSensitive: false)
        .firstMatch(lower);
    if (nameMatch != null) {
      final name = nameMatch.group(1)?.trim();
      if (name != null && name.length > 1 && name.length < 30) {
        triples.add(KGTriple(subject: 'user', predicate: 'name_is', object: name));
      }
    }

    // "I like/love/enjoy/prefer X"
    final likeMatch = RegExp(r"i\s+(?:like|love|enjoy|prefer)\s+(.+?)(?:\.|,|$)", caseSensitive: false)
        .firstMatch(lower);
    if (likeMatch != null) {
      final obj = likeMatch.group(1)?.trim();
      if (obj != null && obj.length > 1 && obj.length < 50) {
        triples.add(KGTriple(subject: 'user', predicate: 'likes', object: obj));
      }
    }

    // "I work at/for X" / "I'm working at X"
    final workMatch = RegExp(r"i(?:'m|\s+am)?\s+work(?:ing)?\s+(?:at|for)\s+(.+?)(?:\.|,|$)", caseSensitive: false)
        .firstMatch(lower);
    if (workMatch != null) {
      final place = workMatch.group(1)?.trim();
      if (place != null && place.length > 1 && place.length < 50) {
        triples.add(KGTriple(subject: 'user', predicate: 'works_at', object: place));
      }
    }

    // "I live in X"
    final liveMatch = RegExp(r"i\s+live\s+in\s+(.+?)(?:\.|,|$)", caseSensitive: false)
        .firstMatch(lower);
    if (liveMatch != null) {
      final place = liveMatch.group(1)?.trim();
      if (place != null && place.length > 1 && place.length < 50) {
        triples.add(KGTriple(subject: 'user', predicate: 'lives_in', object: place));
      }
    }

    // "I have X"
    final haveMatch = RegExp(r"i\s+have\s+(?:a |an )?(.+?)(?:\.|,|$)", caseSensitive: false)
        .firstMatch(lower);
    if (haveMatch != null) {
      final obj = haveMatch.group(1)?.trim();
      if (obj != null && obj.length > 1 && obj.length < 50 &&
          !obj.contains('question') && !obj.contains('problem')) {
        triples.add(KGTriple(subject: 'user', predicate: 'has', object: obj));
      }
    }

    // "I need/want X"
    final needMatch = RegExp(r"i\s+(?:need|want)\s+(?:to )?\s*(.+?)(?:\.|,|$)", caseSensitive: false)
        .firstMatch(lower);
    if (needMatch != null) {
      final obj = needMatch.group(1)?.trim();
      if (obj != null && obj.length > 2 && obj.length < 50) {
        triples.add(KGTriple(subject: 'user', predicate: 'wants', object: obj));
      }
    }

    return triples;
  }

  // ════════════════════════════════════════════════════════════════════
  // ASSET TRACKING
  // ════════════════════════════════════════════════════════════════════

  /// Upsert an indexed asset. Returns true if it was new/changed.
  Future<bool> trackAsset({
    required String path,
    required String type,
    required String fingerprint,
    Map<String, dynamic>? metadata,
  }) async {
    final d = await db;
    final existing = await d.query('assets',
        where: 'path = ?', whereArgs: [path], limit: 1);

    if (existing.isNotEmpty) {
      final old = existing.first;
      if (old['fingerprint'] == fingerprint) return false; // Unchanged
      await d.update('assets', {
        'fingerprint': fingerprint,
        'last_indexed': DateTime.now().toIso8601String(),
        'metadata': metadata != null ? jsonEncode(metadata) : old['metadata'],
      }, where: 'path = ?', whereArgs: [path]);
      return true;
    }

    await d.insert('assets', {
      'path': path,
      'type': type,
      'fingerprint': fingerprint,
      'last_indexed': DateTime.now().toIso8601String(),
      'metadata': metadata != null ? jsonEncode(metadata) : null,
    });
    return true;
  }

  /// Check if an asset fingerprint has changed.
  Future<bool?> assetChanged(String path, String fingerprint) async {
    final d = await db;
    final rows = await d.query('assets',
        where: 'path = ?', whereArgs: [path], limit: 1);
    if (rows.isEmpty) return null; // Not tracked
    return rows.first['fingerprint'] != fingerprint;
  }

  // ════════════════════════════════════════════════════════════════════
  // MIGRATION — import existing JSON files on first run
  // ════════════════════════════════════════════════════════════════════

  Future<void> _migrateFromJson() async {
    final d = await db;
    final dir = await getApplicationDocumentsDirectory();

    // Migrate episodic_memory.json → chat_turns
    final memFile = File('${dir.path}/episodic_memory.json');
    if (await memFile.exists()) {
      try {
        final count = Sqflite.firstIntValue(
            await d.rawQuery('SELECT COUNT(*) FROM chat_turns'));
        if (count == 0) {
          final raw = await memFile.readAsString();
          final list = jsonDecode(raw) as List;
          final batch = d.batch();
          for (final item in list) {
            final map = item as Map<String, dynamic>;
            batch.insert('chat_turns', {
              'query': map['query'] ?? '',
              'response': map['response'] ?? '',
              'intent': map['intent'] ?? 'chat',
              'model_used': map['model'] ?? 'free',
              'steps': map['steps'] != null ? jsonEncode(map['steps']) : null,
              'created_at': map['timestamp'] ?? DateTime.now().toIso8601String(),
            });
          }
          await batch.commit(noResult: true);
          debugPrint('[OculaDB] Migrated ${list.length} chat turns from JSON');
        }
        // Rename old file (keep as backup)
        await memFile.rename('${dir.path}/episodic_memory.json.bak');
      } catch (e) {
        debugPrint('[OculaDB] Chat migration failed (non-fatal): $e');
      }
    }

    // Migrate rag_index_v2.json → rag_chunks + rag_meta
    final ragFile = File('${dir.path}/rag_index_v2.json');
    if (await ragFile.exists()) {
      try {
        final count = Sqflite.firstIntValue(
            await d.rawQuery('SELECT COUNT(*) FROM rag_chunks'));
        if (count == 0) {
          final raw = await ragFile.readAsString();
          final data = jsonDecode(raw);

          List entries;
          Map<String, dynamic>? fingerprints;
          if (data is Map) {
            entries = data['entries'] as List? ?? [];
            fingerprints = (data['fingerprints'] as Map<String, dynamic>?)?.cast<String, String>();
          } else {
            entries = data as List;
          }

          final batch = d.batch();
          for (final item in entries) {
            final map = item as Map<String, dynamic>;
            final vector = (map['vector'] as List?)
                ?.cast<num>()
                .map((n) => n.toDouble())
                .toList() ?? [];

            batch.insert('rag_chunks', {
              'text': map['text'] ?? '',
              'vector': _encodeVector(vector),
              'source': map['source'] ?? '',
              'source_id': map['sourceId'] ?? '',
              'chunk_idx': map['chunkIdx'] ?? 0,
              'created_at': map['timestamp'] ?? DateTime.now().toIso8601String(),
            });
          }

          // Migrate fingerprints
          if (fingerprints != null) {
            for (final entry in fingerprints.entries) {
              batch.insert('rag_meta', {
                'key': 'fp:${entry.key}',
                'value': entry.value.toString(),
              }, conflictAlgorithm: ConflictAlgorithm.replace);
            }
          }

          await batch.commit(noResult: true);
          debugPrint('[OculaDB] Migrated ${entries.length} RAG chunks + '
              '${fingerprints?.length ?? 0} fingerprints from JSON');
        }
        // Rename old files
        await ragFile.rename('${dir.path}/rag_index_v2.json.bak');
        final fpFile = File('${dir.path}/rag_fingerprints.json');
        if (await fpFile.exists()) {
          await fpFile.rename('${dir.path}/rag_fingerprints.json.bak');
        }
      } catch (e) {
        debugPrint('[OculaDB] RAG migration failed (non-fatal): $e');
      }
    }
  }

  // ════════════════════════════════════════════════════════════════════
  // VECTOR ENCODING — Float32 list ↔ BLOB
  // ════════════════════════════════════════════════════════════════════

  Uint8List _encodeVector(List<double> vector) {
    if (vector.isEmpty) return Uint8List(0);
    final floats = Float32List.fromList(vector.map((v) => v.toDouble()).toList());
    return floats.buffer.asUint8List();
  }

  List<double> _decodeVector(Uint8List blob) {
    if (blob.isEmpty) return [];
    final floats = blob.buffer.asFloat32List();
    return floats.map((f) => f.toDouble()).toList();
  }

  double _cosine(List<double> a, List<double> b) {
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

  /// Get DB stats for diagnostics.
  Future<Map<String, int>> stats() async {
    final d = await db;
    final chat = Sqflite.firstIntValue(
        await d.rawQuery('SELECT COUNT(*) FROM chat_turns')) ?? 0;
    final chunks = Sqflite.firstIntValue(
        await d.rawQuery('SELECT COUNT(*) FROM rag_chunks')) ?? 0;
    final kg = Sqflite.firstIntValue(
        await d.rawQuery('SELECT COUNT(*) FROM knowledge')) ?? 0;
    final assets = Sqflite.firstIntValue(
        await d.rawQuery('SELECT COUNT(*) FROM assets')) ?? 0;
    return {
      'chat_turns': chat,
      'rag_chunks': chunks,
      'knowledge_triples': kg,
      'tracked_assets': assets,
    };
  }

  /// Close the database.
  Future<void> close() async {
    await _db?.close();
    _db = null;
  }
}

/// A single RAG search result.
class RagSearchResult {
  final int id;
  final String text;
  final String source;
  final String sourceId;
  final double score;
  final DateTime timestamp;

  RagSearchResult({
    required this.id,
    required this.text,
    required this.source,
    required this.sourceId,
    required this.score,
    required this.timestamp,
  });
}

/// A knowledge graph triple.
class KGTriple {
  final String subject;
  final String predicate;
  final String object;
  final String? source;
  final double confidence;

  KGTriple({
    required this.subject,
    required this.predicate,
    required this.object,
    this.source,
    this.confidence = 1.0,
  });
}
