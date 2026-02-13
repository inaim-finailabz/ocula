import 'package:flutter/foundation.dart';

import 'ai_manager.dart';
import 'rag_engine.dart';
import 'indexer.dart';
import 'network_permission.dart';
import 'local_data.dart';
import 'ocula_db.dart';

/// Lightweight agent orchestrator inspired by LangGraph patterns.
/// No backend. No MongoDB. Just a local state machine on the phone.
///
/// Pipeline: Intent → Retrieve → Route Model → Generate → Log

/// Result from an orchestrator run — response text + linked assets.
class OrchestratorResult {
  final String response;
  final List<LinkedAsset> linkedAssets;

  const OrchestratorResult(this.response, [this.linkedAssets = const []]);

  /// Empty / cancelled result.
  static const empty = OrchestratorResult('');
}

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
  List<LinkedAsset> linkedAssets;

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
    List<LinkedAsset>? linkedAssets,
  })  : stepsCompleted = stepsCompleted ?? [],
        timestamp = timestamp ?? DateTime.now(),
        linkedAssets = linkedAssets ?? [];
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

  /// True while a run() call is in flight.
  bool _isRunning = false;
  bool get isRunning => _isRunning;

  /// Set to true to abort the current pipeline mid-flight.
  bool _cancelled = false;

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

  /// Stop the current generation and cancel the pipeline.
  /// Safe to call even if nothing is running.
  Future<void> stop() async {
    _cancelled = true;
    await _ai.stopGeneration();
    debugPrint('[Orchestrator] Stop requested');
  }

  /// Run the full pipeline for a user query.
  /// Returns an [OrchestratorResult] with the response + any linked assets.
  Future<OrchestratorResult> run(String query, {bool hasImage = false, String? imagePath}) async {
    // If a previous run is still active, stop it first.
    if (_isRunning) {
      await stop();
      // Give the native side a tick to set g_should_stop
      await Future.delayed(const Duration(milliseconds: 50));
    }

    _cancelled = false;
    _isRunning = true;

    try {
      return await _runPipeline(query, hasImage: hasImage, imagePath: imagePath);
    } finally {
      _isRunning = false;
    }
  }

  /// Internal pipeline — separated so run() can manage _isRunning flag.
  Future<OrchestratorResult> _runPipeline(String query, {bool hasImage = false, String? imagePath}) async {
    var state = AgentState(query: query, hasImage: hasImage, imagePath: imagePath);
    state.status = StepStatus.running;

    // STEP 0: If a better model finished downloading in the background,
    //         switch to it NOW (between queries, never mid-generation).
    final upgraded = await _ai.applyPendingUpgrade();
    if (upgraded) {
      debugPrint('[Orchestrator] ⬆ Auto-upgraded to ${_ai.activeTier?.name}');
    }

    if (_cancelled) return OrchestratorResult.empty;

    // STEP 1: Detect intent
    state = await _detectIntent(state);
    if (_cancelled) return OrchestratorResult.empty;

    // STEP 2: Retrieve context via RAG (also collects linked assets)
    state = await _retrieve(state);
    if (_cancelled) return OrchestratorResult.empty;

    // STEP 3: Check episodic memory for recent relevant conversations
    state = await _recallMemory(state);
    if (_cancelled) return OrchestratorResult.empty;

    // STEP 4: If web intent, check internet permission
    if (state.intent == QueryIntent.web) {
      state = await _webSearch(state);
      if (_cancelled) return OrchestratorResult.empty;
    }

    // STEP 5: Route to the right model
    state = await _routeModel(state);
    if (_cancelled) return OrchestratorResult.empty;

    // STEP 6: Generate response
    state = await _generate(state);
    if (_cancelled) return OrchestratorResult.empty;

    // STEP 7: Log to episodic memory
    await _memory.log(state);

    // STEP 8: Index conversation for RAG (non-blocking)
    Indexer().indexChatTurn(state.query, state.response).catchError((_) {});

    // Revoke temp internet grant after query completes
    _network.revokeTemp();

    state.status = StepStatus.completed;
    return OrchestratorResult(state.response, state.linkedAssets);
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
  /// Passes intent-based source hint to boost relevant result types.
  /// Also collects any linked assets (files, photos, emails, contacts) from
  /// matched RAG sources for surfacing in the chat UI.
  Future<AgentState> _retrieve(AgentState state) async {
    // Map intent to a source hint for the hybrid search
    String? sourceHint;
    switch (state.intent) {
      case QueryIntent.file:
        sourceHint = 'file';
        break;
      case QueryIntent.email:
        sourceHint = 'email';
        break;
      case QueryIntent.photo:
        sourceHint = 'photo';
        break;
      case QueryIntent.calendar:
        sourceHint = 'calendar';
        break;
      case QueryIntent.contact:
        sourceHint = 'contact';
        break;
      default:
        break;
    }

    // Single search — reuse results for both context and asset linking
    final results = await _rag.search(state.query, sourceHint: sourceHint);

    if (results.isNotEmpty) {
      // Build context string from results
      state.ragContext = results.map((r) {
        final label = _sourceLabel(r.source);
        return '$label: ${r.text}';
      }).join('\n\n');

      // Collect linked assets from RAG source IDs
      try {
        final sourceIds = results.map((r) => r.sourceId).toList();
        state.linkedAssets = await OculaDB().findLinkedAssets(sourceIds);
      } catch (e) {
        if (kDebugMode) print('[Orchestrator] Asset linking skipped: $e');
      }
    }

    state.stepsCompleted.add('retrieve');
    return state;
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
  /// SAFETY: Never unloads the current model. Only switches if the target
  /// model is already downloaded and ready. If no better model is available,
  /// keeps the currently loaded model running.
  ///
  /// 1. Hardware gate: low-RAM devices → Sensor (free) only.
  /// 2. Reasoning intent → Thinker (pro / Qwen3-VL-2B).
  /// 3. Spatial intent   → Specialist (plus / Moondream 2).
  /// 4. Default          → Sensor (free / SmolVLM2).
  Future<AgentState> _routeModel(AgentState state) async {
    // Hardware gate
    final ram = await _ai.deviceRamMB;
    if (ram < 4000) {
      // Low-RAM: stay on free, don't even attempt switching
      state.modelUsed = _ai.activeTier ?? AITier.free;
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

    // Build preferred tier order with graceful fallback
    List<AITier> preferred;
    if (isReasoning) {
      preferred = [AITier.pro, AITier.plus, AITier.free];
    } else if (isSpatial) {
      preferred = [AITier.plus, AITier.pro, AITier.free];
    } else {
      preferred = [AITier.free];
    }

    // ── SAFE ROUTING: Check download status BEFORE attempting any switch ──
    // Never call switchEngine unless the model is on disk.
    // This prevents unloading the current working model for nothing.
    for (final tier in preferred) {
      // Skip tiers whose model isn't downloaded yet
      if (!await _ai.isTierDownloaded(tier)) {
        debugPrint('[Orchestrator] ${tier.name} not downloaded — skipping');
        continue;
      }

      // Already on this tier? Use it.
      if (_ai.activeTier == tier) {
        state.modelUsed = tier;
        debugPrint('[Orchestrator] Route: already on ${tier.name}');
        state.stepsCompleted.add('route_model');
        return state;
      }

      // Model is downloaded — safe to switch
      try {
        await _ai.switchEngine(tier);
        state.modelUsed = tier;
        debugPrint('[Orchestrator] Route: ${tier.name} (intent=${state.intent.name})');
        state.stepsCompleted.add('route_model');
        return state;
      } catch (e) {
        debugPrint('[Orchestrator] ${tier.name} switch failed: $e — keeping current');
        continue;
      }
    }

    // All preferred tiers failed — keep whatever is currently loaded
    state.modelUsed = _ai.activeTier ?? AITier.free;
    debugPrint('[Orchestrator] Keeping current: ${state.modelUsed.name}');
    state.stepsCompleted.add('route_model');
    return state;
  }

  /// Node 6: Generate response from the LLM.
  Future<AgentState> _generate(AgentState state) async {
    // Build context string — ai_manager.ask() handles the system prompt + ChatML template
    final context = state.ragContext;

    state.response = await _ai.ask(
      state.query,
      context: context,
      hasImage: state.hasImage,
      imagePath: state.imagePath,
      intent: state.intent,
    );
    state.stepsCompleted.add('generate');
    return state;
  }
}

// ──────────────────────────────────────────
// Episodic Memory — backed by OculaDB (SQLite)
// ──────────────────────────────────────────

/// Episodic memory backed by SQLite via OculaDB.
/// No more loading entire JSON into memory.
///
/// Auto-migrates from episodic_memory.json on first DB open.
class EpisodicMemory {
  final OculaDB _db = OculaDB();

  /// Log a completed interaction.
  Future<void> log(AgentState state) async {
    await _db.logChat(
      query: state.query,
      response: state.response,
      intent: state.intent.name,
      modelUsed: state.modelUsed.name,
      steps: state.stepsCompleted,
    );

    // Extract knowledge graph triples from conversation
    await _db.extractKnowledge(state.query, state.response);

    // Auto-trim at 2000 entries (up from 200 with JSON)
    final count = await _db.chatCount;
    if (count > 2200) {
      await _db.trimChat(maxEntries: 2000);
    }
  }

  /// Recall recent conversations relevant to a query.
  Future<String> recall(String query, {int limit = 3}) async {
    // Combine keyword recall with knowledge graph context
    final chatRecall = await _db.recallChat(query, limit: limit);
    final graphContext = await _db.graphContext(query, limit: 3);

    final parts = <String>[];
    if (chatRecall.isNotEmpty) parts.add(chatRecall);
    if (graphContext.isNotEmpty) {
      parts.add('[Knowledge graph] $graphContext');
    }
    return parts.join('\n');
  }

  /// Get total entries count.
  Future<int> get count async => _db.chatCount;

  /// Clear all memory.
  Future<void> clear() async => _db.clearChat();
}
