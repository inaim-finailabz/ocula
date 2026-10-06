import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Lightweight on-device RL-style preference learning.
///
/// Tracks what the user engages with (thumbs up/down, re-asks, follows
/// suggestions, opens linked assets) and builds a preference profile
/// that adjusts the assistant's behavior — tone, suggestion frequency,
/// activity types, meal preferences, and model routing.
///
/// No backend. No cloud. All learning happens on-device via a simple
/// bandit-style reward signal: +1 for positive signal, -1 for negative.
/// Older scores shrink slightly with each new signal, so recent behavior
/// matters more.
class UserProfile {
  static final UserProfile _instance = UserProfile._();
  factory UserProfile() => _instance;
  UserProfile._();

  static const _kVersion = 'user_profile_v1';

  /// Preference categories the RL system tracks.
  /// Each category has a score from -10 to +10.
  final Map<String, double> _scores = {};
  final Map<String, int> _counts = {}; // how many signals we've seen

  Future<void>? _loading;

  /// Load saved profile once. Every mutator awaits this first so a signal
  /// recorded before load can't overwrite the stored profile with an
  /// empty one.
  Future<void> load() => _loading ??= _load();

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    final json = prefs.getString(_kVersion);
    if (json != null) {
      try {
        final data = jsonDecode(json) as Map<String, dynamic>;
        final scores = data['scores'] as Map<String, dynamic>?;
        final counts = data['counts'] as Map<String, dynamic>?;
        if (scores != null) {
          _scores.clear();
          scores.forEach((k, v) => _scores[k] = (v as num).toDouble());
        }
        if (counts != null) {
          _counts.clear();
          counts.forEach((k, v) => _counts[k] = (v as num).toInt());
        }
      } catch (e) {
        debugPrint('[UserProfile] Load error: $e');
      }
    }
  }

  /// Persist the current profile.
  Future<void> save() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _kVersion,
      jsonEncode({
        'scores': _scores.map((k, v) => MapEntry(k, v)),
        'counts': _counts.map((k, v) => MapEntry(k, v)),
        'updated_at': DateTime.now().toIso8601String(),
      }),
    );
  }

  /// Record a positive signal (+1) for a preference category.
  /// Categories: 'briefing_detailed', 'meal_suggestions', 'gym_reminders',
  /// 'activity_suggestions', 'proactive_actions', 'photo_search',
  /// 'voice_input', 'follow_ups', 'uplifting_messages'.
  Future<void> reward(String category, {double amount = 1.0}) async {
    await load();
    _applySignal(category, amount);
    await save();
  }

  /// Record a negative signal (-1) for a preference category.
  Future<void> penalize(String category, {double amount = 1.0}) async {
    await load();
    _applySignal(category, -amount);
    await save();
  }

  void _applySignal(String category, double delta) {
    final current = _scores[category] ?? 0.0;
    final count = _counts[category] ?? 0;

    // Shrink the existing score a little so newer signals dominate
    final decayed = current * _decayFactor(count);
    _scores[category] = (decayed + delta).clamp(-10.0, 10.0);
    _counts[category] = count + 1;

    debugPrint(
      '[UserProfile] $category: ${_scores[category]!.toStringAsFixed(2)} '
      '(${_counts[category]} signals)',
    );
  }

  /// Decay factor: 1.0 for first signal, approaches 0.95 as count grows.
  /// This means recent signals have slightly more weight.
  double _decayFactor(int count) {
    if (count == 0) return 1.0;
    return 1.0 - (count * 0.001).clamp(0.0, 0.05);
  }

  /// Get the score for a category (-10 to +10). 0 = neutral.
  double score(String category) => _scores[category] ?? 0.0;

  /// True if the user has a positive preference for this category (> 1.0).
  bool prefers(String category) => score(category) > 1.0;

  /// True if the user has a negative preference for this category (< -1.0).
  bool dislikes(String category) => score(category) < -1.0;

  /// Get a confidence-weighted score from 0.0 to 1.0.
  /// Low signal count → 0.5 (neutral). More signals → closer to actual score.
  double confidence(String category) {
    final s = score(category);
    final c = _counts[category] ?? 0;
    if (c == 0) return 0.5;
    // Map [-10, 10] → [0, 1], with confidence scaling by signal count
    final normalized = (s + 10) / 20;
    final confWeight = (c / 10).clamp(0.0, 1.0);
    return 0.5 + (normalized - 0.5) * confWeight;
  }

  /// Build a context string for the LLM system prompt.
  /// Tells the model what the user likes/dislikes so it can personalize.
  String toPromptContext() {
    if (_scores.isEmpty) return '';

    final parts = <String>[];

    // Only include categories with enough signal (count >= 2)
    final significant = _scores.entries
        .where((e) => (_counts[e.key] ?? 0) >= 2)
        .toList();

    for (final entry in significant) {
      final cat = entry.key;
      final score = entry.value;
      final count = _counts[cat] ?? 0;

      if (score > 2.0) {
        parts.add(
          'User likes $cat (score: ${score.toStringAsFixed(1)}, '
          'based on $count interactions).',
        );
      } else if (score < -2.0) {
        parts.add(
          'User dislikes $cat (score: ${score.toStringAsFixed(1)}, '
          'based on $count interactions).',
        );
      }
    }

    if (parts.isEmpty) return '';
    return '\n\n[User preferences]\n${parts.join('\n')}';
  }

  /// Get a summary of the profile for the settings screen.
  Map<String, dynamic> get summary => {
    'scores': Map<String, double>.from(_scores),
    'counts': Map<String, int>.from(_counts),
    'total_signals': _counts.values.fold(0, (a, b) => a + b),
  };

  /// Detect preference signals from user behavior automatically.
  ///
  /// Called by the orchestrator after each interaction:
  /// - User asks a follow-up about the same topic → reward('follow_ups')
  /// - User opens a linked asset → reward('photo_search') or reward('file_search')
  /// - User says "thanks" or "good" → reward('briefing_detailed')
  /// - User dismisses/ignores a briefing → penalize('briefing_detailed')
  /// - User asks about food/meal → reward('meal_suggestions')
  /// - User asks about gym/exercise → reward('gym_reminders')
  /// - User uses voice input → reward('voice_input')
  Future<void> recordInteraction({
    required String userQuery,
    required String assistantResponse,
    bool? userOpenedAsset,
    bool? isVoiceInput,
    bool? isFollowUp,
  }) async {
    await load();
    final lower = userQuery.toLowerCase();
    bool has(RegExp re) => re.hasMatch(lower);

    if (isVoiceInput == true) _applySignal('voice_input', 1);
    if (isFollowUp == true) _applySignal('follow_ups', 1);
    if (userOpenedAsset == true && has(_photoWords)) {
      _applySignal('photo_search', 1);
    }
    if (has(_mealWords)) _applySignal('meal_suggestions', 1);
    if (has(_activityWords)) _applySignal('gym_reminders', 1);
    if (has(_positiveWords)) _applySignal('briefing_detailed', 1);
    if (has(_negativeWords)) _applySignal('briefing_detailed', -1);

    // One write per interaction.
    await save();
  }

  // Whole-word matching: "great" must not count as "eat", "brunch" as "run".
  static final _photoWords = RegExp(r'\b(photos?|images?|pictures?|pics?)\b');
  static final _mealWords = RegExp(
    r'\b(eat|eating|food|meals?|breakfast|lunch|dinner|hungry|recipes?)\b',
  );
  static final _activityWords = RegExp(
    r'\b(gym|exercise|workout|running|run|walk|walking|activity|fitness)\b',
  );
  static final _positiveWords = RegExp(
    r'\b(thanks|thank you|great|helpful|love it|perfect)\b',
  );
  static final _negativeWords = RegExp(
    r"\b(too long|annoying|too much|stop (the|these) (briefings?|messages))\b",
  );

  /// Reset all preferences (for the "clear profile" button in settings).
  Future<void> reset() async {
    await load();
    _scores.clear();
    _counts.clear();
    await save();
  }
}
