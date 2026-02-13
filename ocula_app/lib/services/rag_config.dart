import 'package:shared_preferences/shared_preferences.dart';
import 'ai_manager.dart';

/// Persisted RAG tuning parameters.
///
/// Singleton backed by SharedPreferences. Call [load] once at startup,
/// then read properties synchronously. Setters persist immediately.
class RagConfig {
  static final RagConfig _instance = RagConfig._();
  factory RagConfig() => _instance;
  RagConfig._();

  SharedPreferences? _prefs;
  bool _loaded = false;

  // ── Defaults ──
  static const double defaultVectorWeight = 0.55;
  static const int defaultTopK = 5;
  static const double defaultMinScore = 0.15;
  static const int defaultContextBudgetChars = 1200;
  static const int defaultMaxResponseTokens = 384;
  static const int defaultChunkSize = 800;
  /// 'auto' means intent-based routing. Otherwise 'free', 'plus', 'pro'.
  static const String defaultModelOverride = 'auto';

  // ── Keys ──
  static const _kVectorWeight = 'rag_vector_weight';
  static const _kTopK = 'rag_top_k';
  static const _kMinScore = 'rag_min_score';
  static const _kContextBudget = 'rag_context_budget';
  static const _kMaxTokens = 'rag_max_tokens';
  static const _kChunkSize = 'rag_chunk_size';
  static const _kModelOverride = 'rag_model_override';

  /// Load from disk. Safe to call multiple times.
  Future<void> load() async {
    if (_loaded) return;
    _prefs = await SharedPreferences.getInstance();
    _loaded = true;
  }

  // ── Getters ──

  double get vectorWeight =>
      _prefs?.getDouble(_kVectorWeight) ?? defaultVectorWeight;

  int get topK => _prefs?.getInt(_kTopK) ?? defaultTopK;

  double get minScore =>
      _prefs?.getDouble(_kMinScore) ?? defaultMinScore;

  int get contextBudgetChars =>
      _prefs?.getInt(_kContextBudget) ?? defaultContextBudgetChars;

  int get maxResponseTokens =>
      _prefs?.getInt(_kMaxTokens) ?? defaultMaxResponseTokens;

  int get chunkSize => _prefs?.getInt(_kChunkSize) ?? defaultChunkSize;

  /// Model override: 'auto', 'free', 'plus', or 'pro'.
  String get modelOverride =>
      _prefs?.getString(_kModelOverride) ?? defaultModelOverride;

  /// Resolved tier from the override. Null = auto (intent-based routing).
  AITier? get modelOverrideTier {
    switch (modelOverride) {
      case 'free': return AITier.free;
      case 'plus': return AITier.plus;
      case 'pro': return AITier.pro;
      default: return null; // 'auto'
    }
  }

  // ── Setters (persist immediately) ──

  Future<void> setVectorWeight(double v) async {
    await _ensureLoaded();
    await _prefs!.setDouble(_kVectorWeight, v);
  }

  Future<void> setTopK(int v) async {
    await _ensureLoaded();
    await _prefs!.setInt(_kTopK, v);
  }

  Future<void> setMinScore(double v) async {
    await _ensureLoaded();
    await _prefs!.setDouble(_kMinScore, v);
  }

  Future<void> setContextBudgetChars(int v) async {
    await _ensureLoaded();
    await _prefs!.setInt(_kContextBudget, v);
  }

  Future<void> setMaxResponseTokens(int v) async {
    await _ensureLoaded();
    await _prefs!.setInt(_kMaxTokens, v);
  }

  Future<void> setChunkSize(int v) async {
    await _ensureLoaded();
    await _prefs!.setInt(_kChunkSize, v);
  }

  Future<void> setModelOverride(String v) async {
    await _ensureLoaded();
    await _prefs!.setString(_kModelOverride, v);
  }

  /// Reset all parameters to defaults.
  Future<void> resetDefaults() async {
    await _ensureLoaded();
    await _prefs!.remove(_kVectorWeight);
    await _prefs!.remove(_kTopK);
    await _prefs!.remove(_kMinScore);
    await _prefs!.remove(_kContextBudget);
    await _prefs!.remove(_kMaxTokens);
    await _prefs!.remove(_kChunkSize);
    await _prefs!.remove(_kModelOverride);
  }

  Future<void> _ensureLoaded() async {
    if (!_loaded) await load();
  }
}
