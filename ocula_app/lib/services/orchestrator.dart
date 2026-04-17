import 'dart:async';
import 'package:flutter/foundation.dart';

import 'ai_manager.dart';
import 'rag_engine.dart';
import 'rag_config.dart';
import 'indexer.dart';
import 'network_permission.dart';
import 'local_data.dart';
import 'ocula_db.dart';

/// Lightweight agent orchestrator inspired by LangGraph patterns.
/// No backend. No MongoDB. Just a local state machine on the phone.
///
/// Pipeline: Intent → Retrieve → Route Model → Generate → Log

// ──────────────────────────────────────────
// Capability permission system
// Inspired by agent-web-client/src/agent/permissions.ts
// ──────────────────────────────────────────

/// Capabilities the orchestrator may request at runtime.
enum OculaCapability { webSearch, contactsRead, calendarRead, emailRead, photoRead, fileRead }

enum CapabilityBehavior { allow, deny, ask }

/// In-memory permission context for a single orchestrator instance.
/// Persistent user preferences live in NetworkPermission (SharedPreferences).
/// This layer handles per-query temporary grants and tier-based overrides.
class CapabilityPermission {
  final Map<OculaCapability, CapabilityBehavior> _rules;

  CapabilityPermission._(this._rules);

  factory CapabilityPermission.defaults() => CapabilityPermission._({
    OculaCapability.webSearch: CapabilityBehavior.ask,
    OculaCapability.contactsRead: CapabilityBehavior.allow,
    OculaCapability.calendarRead: CapabilityBehavior.allow,
    OculaCapability.emailRead: CapabilityBehavior.allow,
    OculaCapability.photoRead: CapabilityBehavior.allow,
    OculaCapability.fileRead: CapabilityBehavior.allow,
  });

  CapabilityBehavior resolve(OculaCapability cap) =>
      _rules[cap] ?? CapabilityBehavior.ask;

  /// Grant a capability for the current query only.
  void grantTemp(OculaCapability cap) => _rules[cap] = CapabilityBehavior.allow;

  /// Revert a temporary grant back to "ask" for the next query.
  void revokeTemp(OculaCapability cap) => _rules[cap] = CapabilityBehavior.ask;

  /// Permanently block a capability (e.g. tier restriction).
  void block(OculaCapability cap) => _rules[cap] = CapabilityBehavior.deny;
}

// ──────────────────────────────────────────
// Agent step events
// Inspired by agent-web-client/src/index.ts ServerEvent types
// ──────────────────────────────────────────

enum AgentStepType {
  detectingIntent,
  retrieving,
  webSearching,
  requestingPermission,
  routingModel,
  generating,
  complete,
  error,
}

class AgentStep {
  final AgentStepType type;
  final String? detail;
  AgentStep(this.type, [this.detail]);

  String get label {
    switch (type) {
      case AgentStepType.detectingIntent:      return 'Analyzing...';
      case AgentStepType.retrieving:           return detail ?? 'Searching...';
      case AgentStepType.webSearching:         return 'Searching the web...';
      case AgentStepType.requestingPermission: return 'Waiting for permission...';
      case AgentStepType.routingModel:         return 'Selecting model...';
      case AgentStepType.generating:           return 'Generating response...';
      case AgentStepType.complete:             return 'Done';
      case AgentStepType.error:                return detail ?? 'Error';
    }
  }
}

/// Result from an orchestrator run — response text + linked assets.
class OrchestratorResult {
  final String response;
  final List<LinkedAsset> linkedAssets;

  const OrchestratorResult(this.response, [this.linkedAssets = const []]);

  /// Empty / cancelled result.
  static const empty = OrchestratorResult('');
}

enum RetrievalScope { all, docs, images, contacts, email, calendar, location }

// ──────────────────────────────────────────
// State
// ──────────────────────────────────────────

enum StepStatus { pending, running, completed, failed }

class AgentState {
  final String query;
  final bool hasImage;
  String? imagePath;
  AITier? forcedTier;
  RetrievalScope retrievalScope;
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
    this.forcedTier,
    this.retrievalScope = RetrievalScope.all,
    this.intent = QueryIntent.chat,
    this.modelUsed = AITier.free,
    this.ragContext = '',
    this.response = '',
    this.status = StepStatus.pending,
    List<String>? stepsCompleted,
    DateTime? timestamp,
    List<LinkedAsset>? linkedAssets,
  }) : stepsCompleted = stepsCompleted ?? [],
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

  /// Per-run capability permission context.
  /// Web search starts as "ask"; local data capabilities default to "allow".
  final CapabilityPermission _permissions = CapabilityPermission.defaults();

  /// Broadcast stream of pipeline step events for the UI to display progress.
  final StreamController<AgentStep> _stepController =
      StreamController<AgentStep>.broadcast();
  Stream<AgentStep> get stepStream => _stepController.stream;

  /// Called when a capability requires user confirmation.
  /// Return true to grant, false to deny for this query.
  Future<bool> Function(OculaCapability capability)? onAskCapability;

  /// Called when the web is allowed but device connectivity is off.
  /// Return true if user agreed to open Settings.
  Future<bool> Function()? onConnectivityNeeded;

  Orchestrator({
    AIManager? ai,
    RAGEngine? rag,
    EpisodicMemory? memory,
    NetworkPermission? network,
    LocalData? localData,
  }) : _ai = ai ?? AIManager(),
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
  Future<OrchestratorResult> run(
    String query, {
    bool hasImage = false,
    String? imagePath,
    RetrievalScope retrievalScope = RetrievalScope.all,
    AITier? forcedTier,
    String? sessionId,
  }) async {
    // If a previous run is still active, stop it first.
    if (_isRunning) {
      await stop();
      // Give the native side a tick to set g_should_stop
      await Future.delayed(const Duration(milliseconds: 50));
    }

    _cancelled = false;
    _isRunning = true;

    try {
      return await _runPipeline(
        query,
        hasImage: hasImage,
        imagePath: imagePath,
        retrievalScope: retrievalScope,
        forcedTier: forcedTier,
        sessionId: sessionId,
      );
    } finally {
      _isRunning = false;
    }
  }

  /// Internal pipeline — separated so run() can manage _isRunning flag.
  Future<OrchestratorResult> _runPipeline(
    String query, {
    bool hasImage = false,
    String? imagePath,
    RetrievalScope retrievalScope = RetrievalScope.all,
    AITier? forcedTier,
    String? sessionId,
  }) async {
    var state = AgentState(
      query: query,
      hasImage: hasImage,
      imagePath: imagePath,
      forcedTier: forcedTier,
      retrievalScope: retrievalScope,
    );
    state.status = StepStatus.running;

    // STEP 0: If a better model finished downloading in the background,
    //         switch to it NOW (between queries, never mid-generation).
    final upgraded = await _ai.applyPendingUpgrade();
    if (upgraded) {
      debugPrint('[Orchestrator] ⬆ Auto-upgraded to ${_ai.activeTier?.name}');
    }

    if (_cancelled) return OrchestratorResult.empty;

    // STEP 1: Detect intent
    _stepController.add(AgentStep(AgentStepType.detectingIntent));
    state = await _detectIntent(state);
    if (_cancelled) return OrchestratorResult.empty;

    // SHORT-CIRCUIT: Greetings bypass RAG/LLM entirely.
    if (state.intent == QueryIntent.chat && _isGreeting(state.query)) {
      state.response =
          'Hi! I\'m Ocula, your private AI assistant. '
          'How can I help you today?';
      state.status = StepStatus.completed;
      state.stepsCompleted.add('greeting_shortcut');
      return OrchestratorResult(state.response);
    }

    // STEPS 2+3: RAG retrieval and episodic memory in parallel
    _stepController.add(AgentStep(AgentStepType.retrieving,
        _retrievalLabel(state.intent, state.retrievalScope)));
    // These are independent DB queries — running them concurrently saves 200-500ms.
    final results = await Future.wait([
      _retrieve(
        AgentState(
          query: query,
          hasImage: hasImage,
          imagePath: imagePath,
          retrievalScope: retrievalScope,
        )..intent = state.intent,
      ),
      _recallMemory(
        AgentState(
          query: query,
          hasImage: hasImage,
          imagePath: imagePath,
          retrievalScope: retrievalScope,
        )..intent = state.intent,
        sessionId: sessionId,
      ),
    ]);
    if (_cancelled) return OrchestratorResult.empty;

    // Merge results back into state
    final retrieveState = results[0];
    final memoryState = results[1];
    state.ragContext = retrieveState.ragContext;
    state.linkedAssets = retrieveState.linkedAssets;
    state.stepsCompleted.addAll(retrieveState.stepsCompleted);
    if (memoryState.ragContext.isNotEmpty) {
      state.ragContext += memoryState.ragContext;
    }
    state.stepsCompleted.addAll(memoryState.stepsCompleted);

    // STEP 4: If web intent, check internet permission
    if (state.intent == QueryIntent.web) {
      _stepController.add(AgentStep(AgentStepType.webSearching));
      state = await _webSearch(state);
      if (_cancelled) return OrchestratorResult.empty;
    }

    // STEP 5: Route to the right model
    _stepController.add(AgentStep(AgentStepType.routingModel));
    state = await _routeModel(state);
    if (_cancelled) return OrchestratorResult.empty;

    // STEP 6: Generate response
    _stepController.add(AgentStep(AgentStepType.generating));
    state = await _generate(state);
    if (_cancelled) return OrchestratorResult.empty;

    // STEP 7: Log to episodic memory
    await _memory.log(state, sessionId: sessionId);

    // STEP 8: Index conversation for RAG (non-blocking)
    Indexer().indexChatTurn(state.query, state.response).catchError((_) {});

    // Revoke temp grants after query completes
    _network.revokeTemp();
    _permissions.revokeTemp(OculaCapability.webSearch);

    state.status = StepStatus.completed;
    _stepController.add(AgentStep(AgentStepType.complete));
    return OrchestratorResult(state.response, state.linkedAssets);
  }

  /// Dispose the step stream. Call when the orchestrator is no longer needed.
  void dispose() {
    _stepController.close();
  }

  /// Human-readable retrieval label for the step stream.
  String _retrievalLabel(QueryIntent intent, RetrievalScope scope) {
    switch (scope) {
      case RetrievalScope.docs:      return 'Searching documents...';
      case RetrievalScope.images:    return 'Searching photos...';
      case RetrievalScope.contacts:  return 'Searching contacts...';
      case RetrievalScope.email:     return 'Searching emails...';
      case RetrievalScope.calendar:  return 'Searching calendar...';
      case RetrievalScope.location:  return 'Searching by location...';
      case RetrievalScope.all:
        switch (intent) {
          case QueryIntent.contact:  return 'Searching contacts...';
          case QueryIntent.calendar: return 'Searching calendar...';
          case QueryIntent.email:    return 'Searching emails...';
          case QueryIntent.file:     return 'Searching documents...';
          case QueryIntent.photo:    return 'Searching photos...';
          default:                   return 'Searching...';
        }
    }
  }

  /// Check if a query is a simple greeting (hello, hi, hey, etc.).
  bool _isGreeting(String query) {
    final trimmed = query.trim().toLowerCase().replaceAll(
      RegExp(r'[^a-z\s]'),
      '',
    );
    const greetings = {
      'hello',
      'hi',
      'hey',
      'hiya',
      'howdy',
      'yo',
      'sup',
      'good morning',
      'good afternoon',
      'good evening',
      'hi there',
      'hello there',
      'hey there',
      'whats up',
      'hows it going',
      'how are you',
    };
    return greetings.contains(trimmed);
  }

  /// Node 1: Detect what the user wants.
  Future<AgentState> _detectIntent(AgentState state) async {
    switch (state.retrievalScope) {
      case RetrievalScope.docs:
        state.intent = QueryIntent.file;
        state.stepsCompleted.add('detect_intent');
        return state;
      case RetrievalScope.images:
      case RetrievalScope.location:
        state.intent = QueryIntent.photo;
        state.stepsCompleted.add('detect_intent');
        return state;
      case RetrievalScope.contacts:
        state.intent = QueryIntent.contact;
        state.stepsCompleted.add('detect_intent');
        return state;
      case RetrievalScope.email:
        state.intent = QueryIntent.email;
        state.stepsCompleted.add('detect_intent');
        return state;
      case RetrievalScope.calendar:
        state.intent = QueryIntent.calendar;
        state.stepsCompleted.add('detect_intent');
        return state;
      case RetrievalScope.all:
        break;
    }

    final lower = state.query.toLowerCase();

    if (lower.contains('search') ||
        lower.contains('google') ||
        lower.contains('look up') ||
        lower.contains('internet') ||
        lower.contains('online') ||
        lower.contains('browse') ||
        lower.contains('web') ||
        lower.contains('real-time') ||
        lower.contains('realtime') ||
        lower.contains('real time') ||
        lower.contains('latest') ||
        lower.contains('current price') ||
        lower.contains('live ')) {
      state.intent = QueryIntent.web;
    } else if (lower.contains('email') ||
        lower.contains('inbox') ||
        lower.contains('mail')) {
      state.intent = QueryIntent.email;
    } else if (lower.contains('photo') ||
        lower.contains('picture') ||
        lower.contains('screenshot') ||
        lower.contains('vacation') ||
        lower.contains('selfie') ||
        lower.contains('image')) {
      state.intent = QueryIntent.photo;
    } else if (lower.contains('file') ||
        lower.contains('document') ||
        lower.contains('pdf') ||
        lower.contains('license') ||
        lower.contains('receipt') ||
        lower.contains('invoice') ||
        lower.contains('contract') ||
        lower.contains('certificate')) {
      state.intent = QueryIntent.file;
    } else if (lower.contains('contact') ||
        lower.contains('phone number') ||
        lower.contains('call')) {
      state.intent = QueryIntent.contact;
    } else if (lower.contains('schedule') ||
        lower.contains('calendar') ||
        lower.contains('meeting') ||
        lower.contains('appointment') ||
        lower.contains('event')) {
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
    switch (state.retrievalScope) {
      case RetrievalScope.docs:
        sourceHint = 'file';
        break;
      case RetrievalScope.images:
        sourceHint = 'photo';
        break;
      case RetrievalScope.contacts:
        sourceHint = 'contact';
        break;
      case RetrievalScope.email:
        sourceHint = 'email';
        break;
      case RetrievalScope.calendar:
        sourceHint = 'calendar';
        break;
      case RetrievalScope.location:
        sourceHint = 'photo';
        break;
      case RetrievalScope.all:
        break;
    }
    if (sourceHint != null) {
      debugPrint(
        '[Orchestrator] Retrieval scope=${state.retrievalScope.name} forcing sourceHint=$sourceHint',
      );
    }
    switch (state.intent) {
      case QueryIntent.file:
        sourceHint ??= 'file';
        break;
      case QueryIntent.email:
        sourceHint ??= 'email';
        break;
      case QueryIntent.photo:
        sourceHint ??= 'photo';
        break;
      case QueryIntent.calendar:
        sourceHint ??= 'calendar';
        break;
      case QueryIntent.contact:
        sourceHint ??= 'contact';
        break;
      default:
        break;
    }

    // Expand query for better recall on photo/document content queries.
    // Users ask things like "my driver's license" or "vacation in Greece" —
    // the raw query may not match indexed metadata, so we add synonyms.
    final rewrite = _rewriteQueryForRetrieval(
      state.query,
      state.intent,
      state.retrievalScope,
    );
    final searchQuery = rewrite.searchQuery;
    final retrievePlan = _buildRetrievePlan(state.intent);

    // Single search — reuse results for both context and asset linking
    var results = await _rag.search(
      searchQuery,
      sourceHint: sourceHint,
      topK: retrievePlan.topK,
      minScore: retrievePlan.minScore,
    );
    debugPrint(
      '[Orchestrator] Hybrid search: ${results.length} results '
      '(sourceHint=$sourceHint, topK=${retrievePlan.topK}, '
      'minScore=${retrievePlan.minScore}, query="$searchQuery")',
    );

    // Fallback: If hybrid search found nothing but we have a specific
    // source intent, list all entries of that type. This handles "list all"
    // queries like "who are my contacts" or "what's on my calendar" where
    // the query doesn't semantically match individual records.
    if (sourceHint != null && results.length < retrievePlan.minDesiredResults) {
      final needed = retrievePlan.minDesiredResults - results.length;
      debugPrint(
        '[Orchestrator] Hybrid search thin (${results.length}) — '
        'backfilling $sourceHint (need $needed)',
      );
      final fallback = await _rag.listBySource(
        sourceHint,
        limit: retrievePlan.backfillLimit,
      );
      results = _mergeUniqueResults(results, fallback);
      debugPrint('[Orchestrator] Backfilled results: ${results.length}');
    }

    results = _rerankByMetadata(
      results,
      rewrite: rewrite,
      sourceHint: sourceHint,
    );

    if (results.isNotEmpty) {
      // Build structured context string so the model can explain:
      // what source it used, when it was created, and what it says.
      state.ragContext = results
          .asMap()
          .entries
          .map((e) => _formatRagContext(e.key + 1, e.value))
          .join('\n\n');

      final ambiguity = _retrievalAmbiguityNote(results);
      if (ambiguity != null) {
        state.ragContext += '\n\n[Ambiguity] $ambiguity';
      }

      // Collect linked assets from RAG source IDs
      try {
        final sourceIds = results.map((r) => r.sourceId).toList();
        final dbAssets = await OculaDB().findLinkedAssets(sourceIds);
        final linkedById = {for (final a in dbAssets) a.sourceId: a};

        // Synthesize chips for file/photo sources not in the DB so tappable
        // links always appear even when linkAsset was never called for them.
        for (final r in results) {
          if (linkedById.containsKey(r.sourceId)) continue;
          if (r.sourceId.startsWith('file:')) {
            final path = r.sourceId.substring('file:'.length);
            linkedById[r.sourceId] = LinkedAsset(
              sourceId: r.sourceId,
              assetType: 'file',
              assetRef: path,
              label: path.split('/').last,
            );
          } else if (r.sourceId.startsWith('photo:')) {
            final path = r.sourceId.substring('photo:'.length);
            linkedById[r.sourceId] = LinkedAsset(
              sourceId: r.sourceId,
              assetType: 'photo',
              assetRef: path,
              label: path.split('/').last,
            );
          }
        }

        // Strict intent-based chip gating.
        // Each chip type is only shown when the user explicitly queried
        // that category. Chat/web queries never show chips — only explicit
        // data queries (contact, email, file, photo, calendar) show their
        // own category's chips. This prevents contact/phone numbers from
        // bleeding into meeting notes, file lookups, and general chat.
        final allAssets = linkedById.values.toList();
        final intent = state.intent;
        state.linkedAssets = allAssets.where((a) {
          switch (a.assetType) {
            case 'contact':
            case 'phone':
              // Only show for explicit contact queries
              return intent == QueryIntent.contact;
            case 'email':
              // Show for contact AND email queries (email address in contacts)
              return intent == QueryIntent.contact ||
                  intent == QueryIntent.email;
            case 'file':
              return intent == QueryIntent.file;
            case 'photo':
            case 'video':
              return intent == QueryIntent.photo;
            case 'calendar':
              return intent == QueryIntent.calendar;
            default:
              // Unknown asset types: never show to avoid clutter
              return false;
          }
        }).toList();
      } catch (e) {
        if (kDebugMode) print('[Orchestrator] Asset linking skipped: $e');
      }
    }

    // Enrich with knowledge graph context — only for contact/email intents.
    // Calendar queries don't need entity enrichment (and adding contacts from
    // the graph pollutes meeting/schedule answers with irrelevant people).
    // File/chat/web intents never get graph enrichment.
    final socialIntents = {
      QueryIntent.contact,
      QueryIntent.email,
    };
    if (socialIntents.contains(state.intent)) {
      try {
        final graphCtx = await OculaDB().graphContext(
          state.query,
          limit: retrievePlan.graphLimit,
        );
        if (graphCtx.isNotEmpty) {
          state.ragContext += '\n\n[Linked entities] $graphCtx';
          debugPrint(
            '[Orchestrator] Graph context added: ${graphCtx.length} chars',
          );
        }
      } catch (e) {
        if (kDebugMode) {
          print('[Orchestrator] Knowledge graph query skipped: $e');
        }
      }
    }

    state.stepsCompleted.add('retrieve');
    return state;
  }

  String? _retrievalAmbiguityNote(List<RAGResult> results) {
    if (results.isEmpty) return 'No strong source matched this query.';

    final top = results.first.score;
    if (top < 0.18) {
      return 'Low-confidence retrieval. Ask a clarifying question before giving specific facts.';
    }

    if (results.length >= 2) {
      final second = results[1].score;
      final close = (top - second).abs() <= 0.03;
      final differentSource = results[0].sourceId != results[1].sourceId;
      if (close && differentSource) {
        return 'Multiple near-equal matches found. Ask which source/date/contact the user means before finalizing.';
      }
    }

    return null;
  }

  _QueryRewrite _rewriteQueryForRetrieval(
    String query,
    QueryIntent intent,
    RetrievalScope scope,
  ) {
    final expanded = _expandQuery(query, intent);
    final lower = query.toLowerCase();

    final entityTokens = <String>{};
    final monthTokens = <String>{};
    final locationTokens = <String>{};

    const months = {
      'january',
      'february',
      'march',
      'april',
      'may',
      'june',
      'july',
      'august',
      'september',
      'october',
      'november',
      'december',
      'jan',
      'feb',
      'mar',
      'apr',
      'jun',
      'jul',
      'aug',
      'sep',
      'oct',
      'nov',
      'dec',
    };
    const locationHints = {
      'at',
      'in',
      'near',
      'from',
      'to',
      'on',
      'inside',
      'outside',
    };

    final words = lower
        .replaceAll(RegExp(r'[^a-z0-9\s:@._/-]'), ' ')
        .split(RegExp(r'\s+'))
        .where((w) => w.isNotEmpty)
        .toList();

    for (int i = 0; i < words.length; i++) {
      final w = words[i];
      if (months.contains(w)) monthTokens.add(w);
      if (RegExp(r'^\d{4}$').hasMatch(w) ||
          RegExp(r'^\d{1,2}/\d{1,2}(/\d{2,4})?$').hasMatch(w)) {
        monthTokens.add(w);
      }
      if (w.length >= 4 &&
          !locationHints.contains(w) &&
          !months.contains(w) &&
          !RegExp(r'^\d+$').hasMatch(w)) {
        entityTokens.add(w);
      }
      if (locationHints.contains(w) && i + 1 < words.length) {
        final next = words[i + 1];
        if (next.length >= 3) locationTokens.add(next);
      }
    }

    final queryAugments = <String>[];
    if (intent == QueryIntent.file) {
      queryAugments.add('document file pdf contract receipt invoice');
    }
    if (intent == QueryIntent.photo) {
      queryAugments.add('photo image screenshot gallery camera');
    }
    if (intent == QueryIntent.email) {
      queryAugments.add('email sender subject attachment');
    }
    if (scope == RetrievalScope.location) {
      queryAugments.add(
        'location gps address coordinates place city country map',
      );
    }
    if (locationTokens.isNotEmpty) {
      queryAugments.add(locationTokens.join(' '));
    }
    if (monthTokens.isNotEmpty) {
      queryAugments.add(monthTokens.join(' '));
    }

    final searchQuery = queryAugments.isEmpty
        ? expanded
        : '$expanded ${queryAugments.join(' ')}';

    return _QueryRewrite(
      searchQuery: searchQuery.trim(),
      entityTokens: entityTokens,
      locationTokens: locationTokens,
      dateTokens: monthTokens,
    );
  }

  List<RAGResult> _rerankByMetadata(
    List<RAGResult> results, {
    required _QueryRewrite rewrite,
    required String? sourceHint,
  }) {
    if (results.isEmpty) return results;

    double boostFor(RAGResult r) {
      var boost = 0.0;
      final text = r.text.toLowerCase();
      final sid = r.sourceId.toLowerCase();

      if (sourceHint != null && r.source == sourceHint) {
        boost += 0.08;
      }

      for (final t in rewrite.entityTokens) {
        if (text.contains(t) || sid.contains(t)) boost += 0.015;
      }
      for (final t in rewrite.dateTokens) {
        if (text.contains(t) || sid.contains(t)) boost += 0.025;
      }
      for (final t in rewrite.locationTokens) {
        if (text.contains(t) || sid.contains(t)) boost += 0.025;
      }

      if (r.source == 'email' &&
          (text.contains('from:') || sid.startsWith('email:'))) {
        boost += 0.02;
      }
      if (r.source == 'photo' &&
          (text.contains('gps:') ||
              text.contains('taken ') ||
              text.contains('in album'))) {
        boost += 0.02;
      }
      if (r.source == 'file' &&
          (text.contains('file:') || sid.startsWith('file:'))) {
        boost += 0.02;
      }

      return boost;
    }

    final rescored = results
        .map((r) => _ScoredResult(r, r.score + boostFor(r)))
        .toList();
    rescored.sort((a, b) => b.score.compareTo(a.score));
    return rescored.map((e) => e.result).toList();
  }

  String _formatRagContext(int idx, RAGResult r) {
    final label = _sourceLabel(r.source);
    final ts = r.timestamp.toLocal().toString().substring(0, 16);
    final ref = _sourceRefLabel(r.source, r.sourceId);
    return '[SOURCE $idx]\n'
        'Type: $label\n'
        'Reference: $ref\n'
        'Date: $ts\n'
        'Content: ${r.text}';
  }

  String _sourceRefLabel(String source, String sourceId) {
    switch (source) {
      case 'file':
      case 'photo':
        final path = sourceId.contains(':')
            ? sourceId.substring(sourceId.indexOf(':') + 1)
            : sourceId;
        return path.split('/').last;
      case 'email':
      case 'contact':
      case 'calendar':
        return sourceId;
      default:
        return sourceId;
    }
  }

  _RetrievePlan _buildRetrievePlan(QueryIntent intent) {
    final cfg = RagConfig();
    switch (intent) {
      case QueryIntent.file:
      case QueryIntent.photo:
        return _RetrievePlan(
          topK: cfg.topK < 10 ? 10 : cfg.topK,
          minScore: cfg.minScore > 0.08 ? 0.08 : cfg.minScore,
          minDesiredResults: 8,
          backfillLimit: 24,
          graphLimit: 8,
        );
      case QueryIntent.contact:
      case QueryIntent.calendar:
      case QueryIntent.email:
        return _RetrievePlan(
          topK: cfg.topK < 8 ? 8 : cfg.topK,
          minScore: cfg.minScore > 0.12 ? 0.12 : cfg.minScore,
          minDesiredResults: 5,
          backfillLimit: 16,
          graphLimit: 6,
        );
      case QueryIntent.web:
      case QueryIntent.chat:
        return _RetrievePlan(
          topK: cfg.topK,
          minScore: cfg.minScore,
          minDesiredResults: 3,
          backfillLimit: 10,
          graphLimit: 5,
        );
    }
  }

  List<RAGResult> _mergeUniqueResults(
    List<RAGResult> primary,
    List<RAGResult> fallback,
  ) {
    final merged = <RAGResult>[];
    final seen = <String>{};

    void addAllUnique(List<RAGResult> items) {
      for (final r in items) {
        final key = '${r.sourceId}|${r.text}';
        if (seen.add(key)) {
          merged.add(r);
        }
      }
    }

    addAllUnique(primary);
    addAllUnique(fallback);
    return merged;
  }

  /// Expand a user query with related terms to improve RAG recall.
  /// Example: "my driver's license" → "my driver's license document ID card"
  String _expandQuery(String query, QueryIntent intent) {
    final lower = query.toLowerCase();

    // For photo/file intents, add content-type synonyms to help match
    // metadata-based labels when the user asks about content.
    final expansions = <String>[];

    // Document type expansions
    const docSynonyms = {
      'license': 'ID card document identification',
      'driver': 'driving license permit',
      'passport': 'travel document ID',
      'receipt': 'purchase invoice bill payment',
      'invoice': 'bill receipt payment',
      'contract': 'agreement document signed',
      'resume': 'CV curriculum vitae',
      'ticket': 'boarding pass travel',
      'insurance': 'policy coverage document',
      'certificate': 'diploma degree document',
    };

    // Location/activity expansions for photos
    const placeSynonyms = {
      'vacation': 'holiday trip travel',
      'beach': 'sea ocean coast shore',
      'mountain': 'hiking trail peak summit',
      'wedding': 'ceremony celebration marriage',
      'birthday': 'celebration party cake',
      'concert': 'show music event performance',
      'restaurant': 'dinner food dining meal',
      'graduation': 'ceremony diploma degree',
    };

    final synonymMap = (intent == QueryIntent.photo)
        ? placeSynonyms
        : docSynonyms;
    for (final entry in synonymMap.entries) {
      if (lower.contains(entry.key)) {
        expansions.add(entry.value);
      }
    }

    // Always check both maps for general queries
    if (intent == QueryIntent.chat) {
      for (final entry in {...docSynonyms, ...placeSynonyms}.entries) {
        if (lower.contains(entry.key)) {
          expansions.add(entry.value);
        }
      }
    }

    if (expansions.isEmpty) return query;
    return '$query ${expansions.join(' ')}';
  }

  String _sourceLabel(String source) {
    switch (source) {
      case 'contact':
        return 'CONTACT';
      case 'calendar':
        return 'CALENDAR EVENT';
      case 'photo':
        return 'PHOTO';
      case 'file':
        return 'FILE';
      case 'email':
        return 'EMAIL';
      case 'chat':
        return 'PREVIOUS CONVERSATION';
      default:
        return source.toUpperCase();
    }
  }

  /// Node 3: Check episodic memory for recent conversations about this topic.
  /// [sessionId] scopes recall to the current session only — a new session
  /// starts with a clean slate and does NOT inherit previous sessions' memory.
  Future<AgentState> _recallMemory(
    AgentState state, {
    String? sessionId,
  }) async {
    final recentMemories = await _memory.recall(
      state.query,
      sessionId: sessionId,
    );
    if (recentMemories.isNotEmpty) {
      state.ragContext += '\n\n[Recent conversations]\n$recentMemories';
    }
    state.stepsCompleted.add('recall_memory');
    return state;
  }

  /// Node 4: Web search — only runs when intent is web AND user grants access.
  ///
  /// Permission resolution order:
  /// 1. CapabilityPermission in-memory context (allow/deny overrides)
  /// 2. NetworkPermission persistent prefs (always-allow / always-deny / ask)
  /// 3. onAskCapability UI callback for "ask" behavior
  Future<AgentState> _webSearch(AgentState state) async {
    await _network.load();

    // Check in-memory capability context first (tier blocks, temp grants)
    final capBehavior = _permissions.resolve(OculaCapability.webSearch);
    bool allowed;

    if (capBehavior == CapabilityBehavior.deny) {
      allowed = false;
    } else if (capBehavior == CapabilityBehavior.allow) {
      allowed = true;
    } else {
      // "ask" — defer to persistent NetworkPermission prefs
      allowed = _network.isAllowed;

      if (!allowed && _network.needsPrompt) {
        _stepController.add(AgentStep(AgentStepType.requestingPermission));
        if (onAskCapability != null) {
          allowed = await onAskCapability!(OculaCapability.webSearch);
          if (allowed) {
            _network.grantTemp();
            _permissions.grantTemp(OculaCapability.webSearch);
          }
        }
      }
    }

    if (allowed) {
      final online = await _localData.hasInternetConnection();
      if (!online) {
        bool userWillEnable = false;
        if (onConnectivityNeeded != null) {
          userWillEnable = await onConnectivityNeeded!();
        }
        state.ragContext +=
            '\n\n[Note: Internet is currently unavailable. '
            '${userWillEnable ? 'Please enable Wi-Fi/mobile data in Settings, then try again.' : 'Turn on Wi-Fi/mobile data and try again.'}]';
        state.stepsCompleted.add('web_search');
        return state;
      }

      final webResult = await _localData.webSearch(state.query);
      if (webResult.isNotEmpty) {
        state.ragContext += '\n\n[Web search results]\n$webResult';
      } else {
        state.ragContext +=
            '\n\n[Note: Web search is enabled, but no online results were returned.]';
      }
      state.stepsCompleted.add('web_search');
    } else {
      state.ragContext +=
          '\n\n[Note: User denied internet access for this query. '
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
    // Per-request forced tier (used by feature-specific flows like
    // recording summarization that must run on a specific model).
    final forcedTier = state.forcedTier;
    if (forcedTier != null) {
      if (!await _ai.isTierDownloaded(forcedTier)) {
        throw ModelNotReadyException(forcedTier);
      }
      if (_ai.activeTier != forcedTier) {
        await _ai.switchEngine(forcedTier);
      }
      if (_ai.activeTier != forcedTier) {
        throw ModelNotReadyException(forcedTier);
      }
      state.modelUsed = forcedTier;
      debugPrint('[Orchestrator] Route: forced -> ${forcedTier.name}');
      state.stepsCompleted.add('route_model');
      return state;
    }

    // ── Manual override: if user set a specific model, try it first ──
    final overrideTier = RagConfig().modelOverrideTier;
    debugPrint(
      '[Orchestrator] Route: override=${overrideTier?.name ?? "auto"}, '
      'activeTier=${_ai.activeTier?.name ?? "none"}, hasImage=${state.hasImage}',
    );
    if (overrideTier != null) {
      final downloaded = await _ai.isTierDownloaded(overrideTier);
      debugPrint(
        '[Orchestrator] Override ${overrideTier.name} downloaded=$downloaded',
      );
      if (downloaded) {
        if (_ai.activeTier == overrideTier) {
          state.modelUsed = overrideTier;
          debugPrint(
            '[Orchestrator] Route: override → ${overrideTier.name} (already loaded)',
          );
          state.stepsCompleted.add('route_model');
          return state;
        }
        try {
          await _ai.switchEngine(overrideTier);
          if (_ai.activeTier == overrideTier) {
            state.modelUsed = overrideTier;
            debugPrint('[Orchestrator] Route: override → ${overrideTier.name}');
            state.stepsCompleted.add('route_model');
            return state;
          }
          debugPrint(
            '[Orchestrator] Override ${overrideTier.name} requested but active tier is '
            '${_ai.activeTier?.name ?? "none"} — falling back to auto',
          );
        } catch (e) {
          debugPrint(
            '[Orchestrator] Override ${overrideTier.name} failed: $e — falling back to auto',
          );
        }
      } else {
        debugPrint(
          '[Orchestrator] Override ${overrideTier.name} not downloaded — falling back to auto',
        );
      }
    }

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
    final isReasoning =
        lower.contains('why') ||
        lower.contains('how') ||
        lower.contains('explain') ||
        lower.contains('analyze') ||
        lower.contains('compare') ||
        lower.contains('summarize') ||
        state.intent == QueryIntent.file;

    // Specialist: spatial, counting, pointing
    final isSpatial =
        state.hasImage ||
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
        debugPrint(
          '[Orchestrator] Route: ${tier.name} (intent=${state.intent.name})',
        );
        state.stepsCompleted.add('route_model');
        return state;
      } catch (e) {
        debugPrint(
          '[Orchestrator] ${tier.name} switch failed: $e — keeping current',
        );
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
    debugPrint(
      '[Orchestrator] Generate: intent=${state.intent.name}, '
      'hasImage=${state.hasImage}, tier=${_ai.activeTier?.name}, '
      'contextLen=${context.length}',
    );

    state.response = await _ai.ask(
      state.query,
      context: context,
      hasImage: state.hasImage,
      imagePath: state.imagePath,
      intent: state.intent,
    );
    debugPrint('[Orchestrator] Response: ${state.response.length} chars');
    state.stepsCompleted.add('generate');
    return state;
  }
}

class _RetrievePlan {
  final int topK;
  final double minScore;
  final int minDesiredResults;
  final int backfillLimit;
  final int graphLimit;

  const _RetrievePlan({
    required this.topK,
    required this.minScore,
    required this.minDesiredResults,
    required this.backfillLimit,
    required this.graphLimit,
  });
}

class _QueryRewrite {
  final String searchQuery;
  final Set<String> entityTokens;
  final Set<String> locationTokens;
  final Set<String> dateTokens;

  const _QueryRewrite({
    required this.searchQuery,
    required this.entityTokens,
    required this.locationTokens,
    required this.dateTokens,
  });
}

class _ScoredResult {
  final RAGResult result;
  final double score;

  const _ScoredResult(this.result, this.score);
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
  Future<void> log(AgentState state, {String? sessionId}) async {
    await _db.logChat(
      query: state.query,
      response: state.response,
      intent: state.intent.name,
      modelUsed: state.modelUsed.name,
      steps: state.stepsCompleted,
      sessionId: sessionId,
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
  /// [sessionId] scopes recall to the current session only.
  Future<String> recall(String query, {int limit = 3, String? sessionId}) async {
    // Combine keyword recall with knowledge graph context
    final chatRecall = await _db.recallChat(
      query,
      limit: limit,
      sessionId: sessionId,
    );
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
