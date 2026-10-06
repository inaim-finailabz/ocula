import 'ocula_db.dart';

/// Fast, deterministic lookups for contacts, documents and photos.
///
/// The on-device model is slow and unreliable at restating records, so plain
/// "find / show / what's X's number" requests are answered straight from the
/// index: names, file names and photo labels are matched in SQLite, and the
/// reply is a short templated line with the records rendered as cards.
/// Anything that needs reasoning still goes through the model.
class DirectLookup {
  DirectLookup._();

  // Words that describe the request rather than the thing being looked for.
  static const _stopWords = {
    'a', 'about', 'all', 'an', 'and', 'any', 'are', 'can', 'could', 'do',
    'does', 'for', 'from', 'get', 'give', 'have', 'i', 'in', 'is', 'it',
    'look', 'me', 'my', 'of', 'on', 'open', 'please', 'pull', 'search',
    'show', 'find', 'the', 'their', 'to', 'up', 'what', 'whats', 'where',
    'wheres', 'which', 'who', 'whos', 'with', 'you', 'your', 'list', 'need',
    'want', 'see', 'some', 'there', 'that', 'this', 'one', 'ones',
    // type words
    'contact', 'contacts', 'number', 'numbers', 'phone', 'mobile', 'cell',
    'email', 'emails', 'address', 'call', 'text', 'details', 'info',
    'photo', 'photos', 'picture', 'pictures', 'pic', 'pics', 'image',
    'images', 'document', 'documents', 'doc', 'docs', 'file', 'files', 'pdf',
  };

  /// Search terms left after removing filler/type words and possessives.
  /// "What's Kate's phone number?" → ["kate"].
  static List<String> terms(String query) {
    final cleaned = query
        .toLowerCase()
        .replaceAll(RegExp(r"[’']s\b"), '')
        .replaceAll(RegExp(r"[^a-z0-9@._\-\s]"), ' ');
    final seen = <String>{};
    return [
      for (final t in cleaned.split(RegExp(r'\s+')))
        if (t.length >= 2 && !_stopWords.contains(t) && seen.add(t)) t,
    ];
  }

  static final _lookupLead = RegExp(
    r"^\s*(find|show|get|open|search|look\s*up|pull\s*up|give|list|call|text|"
    r"email|where(?:'s| is| are)|what(?:'s| is| are)|who(?:'s| is))\b",
  );

  /// True for retrieval requests the cards answer fully on their own.
  /// Questions that need reasoning ("when did…", "summarise…", "why…")
  /// return false and go to the model.
  static bool isLookup(String query) {
    final lower = query.toLowerCase().trim();
    if (RegExp(
      r'\b(summar|explain|why|how many|compare|when did|translate|write|draft)',
    ).hasMatch(lower)) {
      return false;
    }
    if (_lookupLead.hasMatch(lower)) return true;
    // Bare names / short noun phrases: "kate bell", "passport scan".
    return terms(lower).isNotEmpty && lower.split(RegExp(r'\s+')).length <= 4;
  }

  /// Short spoken/displayed reply for a direct hit. The cards carry the data.
  static String answer(
    String source,
    List<LinkedAsset> hits,
    List<String> terms,
  ) {
    final what = terms.isEmpty ? '' : ' matching “${terms.join(' ')}”';
    final n = hits.length;
    switch (source) {
      case 'contact':
        if (n == 1) {
          final h = hits.first;
          final phone = RegExp(
            r'^Phone number:\s*(.+)$',
            multiLine: true,
          ).firstMatch(h.snippet ?? '')?.group(1);
          final name = h.label ?? 'this contact';
          return phone == null ? 'Here’s $name.' : 'Here’s $name — $phone.';
        }
        return 'I found $n contacts$what.';
      case 'file':
        return n == 1
            ? 'Here’s the document${what.isEmpty ? '' : what}.'
            : 'I found $n documents$what.';
      case 'photo':
        return n == 1 ? 'Here’s the photo$what.' : 'Here are $n photos$what.';
      default:
        return 'I found $n results$what.';
    }
  }

  /// Ranks rows by how many terms they contain; ties keep DB order.
  static List<RagSearchResult> rank(
    List<RagSearchResult> rows,
    List<String> terms, {
    int limit = 8,
  }) {
    int hits(RagSearchResult r) {
      final hay = '${r.text}\n${r.sourceId}'.toLowerCase();
      return terms.where(hay.contains).length;
    }

    final seen = <String>{};
    final unique = [
      for (final r in rows)
        if (seen.add(r.sourceId)) r,
    ];
    final scored = [for (final r in unique) (r, hits(r))]
      ..removeWhere((e) => e.$2 == 0);
    // Require every term when several were given, if any row satisfies that.
    final full = scored.where((e) => e.$2 == terms.length).toList();
    final pool = full.isNotEmpty ? full : scored;
    pool.sort((a, b) => b.$2.compareTo(a.$2));
    return [for (final e in pool.take(limit)) e.$1];
  }
}
