import 'dart:convert';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

import 'ai_manager.dart';
import 'rag_engine.dart';
import 'network_permission.dart';
import 'local_data.dart';
import 'app_language.dart';

/// Lightweight agent orchestrator inspired by LangGraph patterns.
/// No backend. No MongoDB. Just a local state machine on the phone.
///
/// Pipeline: Intent → Retrieve → Route Model → Generate → Log

// ──────────────────────────────────────────
// State
// ──────────────────────────────────────────

enum StepStatus { pending, running, completed, failed }

class AgentState {
  final String query;
  final bool hasImage;
  String? imagePath;
  QueryIntent intent;
  AITier modelUsed;
  String ragContext;
  String response;
  StepStatus status;
  List<String> stepsCompleted;
  DateTime timestamp;

  AgentState({
    required this.query,
    this.hasImage = false,
    this.imagePath,
    this.intent = QueryIntent.chat,
    this.modelUsed = AITier.free,
    this.ragContext = '',
    this.response = '',
    this.status = StepStatus.pending,
    List<String>? stepsCompleted,
    DateTime? timestamp,
  })  : stepsCompleted = stepsCompleted ?? [],
        timestamp = timestamp ?? DateTime.now();
}

// ──────────────────────────────────────────
// Nodes — each does ONE thing
// ──────────────────────────────────────────

typedef NodeFn = Future<AgentState> Function(AgentState state);

// ──────────────────────────────────────────
// Orchestrator
// ──────────────────────────────────────────

class Orchestrator {
  final AIManager _ai;
  final RAGEngine _rag;
  final EpisodicMemory _memory;
  final NetworkPermission _network;
  final LocalData _localData;

  /// Callback the UI registers to show an "Allow internet?" dialog.
  /// Returns true if user grants permission, false if denied.
  Future<bool> Function()? onAskInternet;

  Orchestrator({
    AIManager? ai,
    RAGEngine? rag,
    EpisodicMemory? memory,
    NetworkPermission? network,
    LocalData? localData,
    this.onAskInternet,
  })  : _ai = ai ?? AIManager(),
        _rag = rag ?? RAGEngine(),
        _memory = memory ?? EpisodicMemory(),
        _network = network ?? NetworkPermission(),
        _localData = localData ?? LocalData();

  /// Run the full pipeline for a user query.
  /// Returns the final response string.
  Future<String> run(String query, {bool hasImage = false, String? imagePath}) async {
    var state = AgentState(query: query, hasImage: hasImage, imagePath: imagePath);
    state.status = StepStatus.running;

    // STEP 1: Detect intent
    state = await _detectIntent(state);

    // STEP 2: Retrieve context via RAG
    state = await _retrieve(state);

    // STEP 3: Check episodic memory for recent relevant conversations
    state = await _recallMemory(state);

    // STEP 4: If web intent, check internet permission
    if (state.intent == QueryIntent.web) {
      state = await _webSearch(state);
    }

    // STEP 5: Route to the right model
    state = await _routeModel(state);

    // STEP 6: Generate response
    state = await _generate(state);

    // STEP 7: Log to episodic memory
    await _memory.log(state);

    // Revoke temp internet grant after query completes
    _network.revokeTemp();

    state.status = StepStatus.completed;
    return state.response;
  }

  /// Node 1: Detect what the user wants.
  Future<AgentState> _detectIntent(AgentState state) async {
    final lower = state.query.toLowerCase();

    if (lower.contains('search') || lower.contains('google') || lower.contains('look up')) {
      state.intent = QueryIntent.web;
    } else if (lower.contains('email') || lower.contains('inbox') || lower.contains('mail')) {
      state.intent = QueryIntent.email;
    } else if (lower.contains('photo') || lower.contains('picture') || lower.contains('screenshot')) {
      state.intent = QueryIntent.photo;
    } else if (lower.contains('file') || lower.contains('document') || lower.contains('pdf')) {
      state.intent = QueryIntent.file;
    } else if (lower.contains('contact') || lower.contains('phone number') || lower.contains('call')) {
      state.intent = QueryIntent.contact;
    } else if (lower.contains('schedule') || lower.contains('calendar') || lower.contains('meeting')) {
      state.intent = QueryIntent.calendar;
    } else {
      state.intent = QueryIntent.chat;
    }

    state.stepsCompleted.add('detect_intent');
    return state;
  }

  /// Node 2: RAG retrieval from local vector store.
  Future<AgentState> _retrieve(AgentState state) async {
    final context = await _rag.getContext(state.query);
    state.ragContext = context;
    state.stepsCompleted.add('retrieve');
    return state;
  }

  /// Node 3: Check episodic memory for recent conversations about this topic.
  Future<AgentState> _recallMemory(AgentState state) async {
    final recentMemories = await _memory.recall(state.query);
    if (recentMemories.isNotEmpty) {
      state.ragContext += '\n\n[Recent conversations]\n$recentMemories';
    }
    state.stepsCompleted.add('recall_memory');
    return state;
  }

  /// Node 4: Web search — only runs when intent is web AND user grants access.
  Future<AgentState> _webSearch(AgentState state) async {
    await _network.load();

    bool allowed = _network.isAllowed;

    // If set to "ask every time", prompt the user via the UI callback
    if (!allowed && _network.needsPrompt && onAskInternet != null) {
      allowed = await onAskInternet!();
      if (allowed) _network.grantTemp();
    }

    if (allowed) {
      final webResult = await _localData.webSearch(state.query);
      if (webResult.isNotEmpty) {
        state.ragContext += '\n\n[Web search results]\n$webResult';
      }
      state.stepsCompleted.add('web_search');
    } else {
      // Denied — tell the LLM it can't access the web
      state.ragContext += '\n\n[Note: User denied internet access for this query. '
          'Answer using only on-device knowledge.]';
      state.stepsCompleted.add('web_search_denied');
    }

    return state;
  }

  /// Node 5: Conditional routing — pick the right model based on state.
  ///
  /// 1. Hardware gate: low-RAM devices → Sensor (free) only.
  /// 2. Reasoning intent → Thinker (pro / Qwen3-VL-2B).
  /// 3. Spatial intent   → Specialist (plus / Moondream 2).
  /// 4. Default          → Sensor (free / SmolVLM2).
  Future<AgentState> _routeModel(AgentState state) async {
    // Hardware gate
    final ram = await _ai.deviceRamMB;
    if (ram < 4000) {
      state.modelUsed = AITier.free;
      await _ai.switchEngine(state.modelUsed);
      state.stepsCompleted.add('route_model');
      return state;
    }

    final lower = state.query.toLowerCase();

    // Thinker: reasoning, analysis, documents
    final isReasoning = lower.contains('why') ||
        lower.contains('how') ||
        lower.contains('explain') ||
        lower.contains('analyze') ||
        lower.contains('compare') ||
        lower.contains('summarize') ||
        state.intent == QueryIntent.file;

    // Specialist: spatial, counting, pointing
    final isSpatial = state.hasImage ||
        lower.contains('where') ||
        lower.contains('count') ||
        lower.contains('find') ||
        lower.contains('point') ||
        lower.contains('total') ||
        lower.contains('receipt') ||
        state.intent == QueryIntent.photo;

    if (isReasoning) {
      state.modelUsed = AITier.pro;
    } else if (isSpatial) {
      state.modelUsed = AITier.plus;
    } else {
      state.modelUsed = AITier.free;
    }

    await _ai.switchEngine(state.modelUsed);
    state.stepsCompleted.add('route_model');
    return state;
  }

  /// Node 6: Generate response from the LLM.
  Future<AgentState> _generate(AgentState state) async {
    final langPrefix = AppLanguage().promptPrefix;

    final prompt = state.ragContext.isNotEmpty
        ? '${langPrefix}You are Ocula, a private AI assistant that lives on the user\'s phone. '
          'Everything you know comes from their device. Be concise.\n\n'
          'Context:\n${state.ragContext}\n\n'
          'User: ${state.query}'
        : '${langPrefix}You are Ocula, a private AI assistant. Be concise.\n\n'
          'User: ${state.query}';

    state.response = await _ai.ask(prompt);
    state.stepsCompleted.add('generate');
    return state;
  }
}

// ──────────────────────────────────────────
// Episodic Memory — remembers conversations
// ──────────────────────────────────────────

/// Lightweight episodic memory stored as a local JSON file.
/// No MongoDB. No server. Just a file on the phone.
///
/// Stores the last N conversations so Ocula can say:
/// "You asked about this yesterday — here's what I found then."
class EpisodicMemory {
  static const _maxEntries = 200;
  static const _fileName = 'episodic_memory.json';

  List<Map<String, dynamic>> _entries = [];
  bool _loaded = false;

  /// Load memory from disk.
  Future<void> _ensureLoaded() async {
    if (_loaded) return;
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/$_fileName');
      if (await file.exists()) {
        final raw = await file.readAsString();
        final list = jsonDecode(raw) as List;
        _entries = list.cast<Map<String, dynamic>>();
      }
    } catch (_) {
      _entries = [];
    }
    _loaded = true;
  }

  /// Save memory to disk.
  Future<void> _save() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/$_fileName');
      await file.writeAsString(jsonEncode(_entries));
    } catch (_) {
      // Silent fail — memory is not critical
    }
  }

  /// Log a completed interaction.
  Future<void> log(AgentState state) async {
    await _ensureLoaded();

    _entries.add({
      'query': state.query,
      'intent': state.intent.name,
      'model': state.modelUsed.name,
      'response': state.response.length > 500
          ? state.response.substring(0, 500)
          : state.response,
      'steps': state.stepsCompleted,
      'timestamp': state.timestamp.toIso8601String(),
    });

    // Keep only the last N entries
    if (_entries.length > _maxEntries) {
      _entries = _entries.sublist(_entries.length - _maxEntries);
    }

    await _save();
  }

  /// Recall recent conversations relevant to a query.
  /// Simple keyword matching — fast, no embedding needed.
  Future<String> recall(String query, {int limit = 3}) async {
    await _ensureLoaded();
    if (_entries.isEmpty) return '';

    final keywords = query.toLowerCase().split(RegExp(r'\s+'))
        .where((w) => w.length > 3)
        .toList();

    if (keywords.isEmpty) return '';

    // Score each entry by keyword overlap
    final scored = _entries.map((entry) {
      final text = '${entry['query']} ${entry['response']}'.toLowerCase();
      final hits = keywords.where((k) => text.contains(k)).length;
      return MapEntry(entry, hits);
    }).where((e) => e.value > 0).toList();

    scored.sort((a, b) => b.value.compareTo(a.value));

    return scored.take(limit).map((e) {
      final entry = e.key;
      return 'On ${entry['timestamp']}: User asked "${entry['query']}" → '
          '${entry['response']}';
    }).join('\n');
  }

  /// Get total entries count.
  Future<int> get count async {
    await _ensureLoaded();
    return _entries.length;
  }

  /// Clear all memory.
  Future<void> clear() async {
    _entries = [];
    _loaded = true;
    await _save();
  }
}
