// Pure text-processing utilities shared across services.
// Kept free of Flutter/platform imports so they can be unit-tested
// with `flutter test` without native channels.

/// Strip leaked prompt/context from model output.
///
/// Some small on-device models echo the system prompt or RAG context
/// verbatim instead of actually answering. This post-processor detects
/// and removes the echoed blocks.
///
/// Returns the original [text] if stripping would empty the string.
String stripLeakedContext(String text) {
  var cleaned = text;

  // If the model starts by echoing the Data:/Phone data: context block,
  // try to find the actual answer after it.
  if (cleaned.startsWith('Data:') || cleaned.startsWith('Phone data:')) {
    // Look for the question marker — actual answer is after it
    final qMatch = RegExp(r'\n(?:Q|Question): .+\n').firstMatch(cleaned);
    if (qMatch != null) {
      cleaned = cleaned.substring(qMatch.end).trim();
    } else {
      // No question marker — strip the echoed context up to the first double newline
      final breakIdx = cleaned.indexOf('\n\n');
      if (breakIdx > 0 && breakIdx < cleaned.length - 10) {
        cleaned = cleaned.substring(breakIdx + 2).trim();
      }
    }
  }

  // Strip echoed section headers from RAG context
  cleaned = cleaned.replaceAll(
    RegExp(
      r'^\[(?:Linked entities|Recent conversations|Note|Knowledge)\].*\n?',
      multiLine: true,
    ),
    '',
  );

  // Strip repeated context labels (CONTACT:, CALENDAR EVENT:, etc.)
  if (cleaned.startsWith('CONTACT:') ||
      cleaned.startsWith('CALENDAR EVENT:') ||
      cleaned.startsWith('PREVIOUS CONVERSATION:') ||
      cleaned.startsWith('FILE:')) {
    final breakIdx = cleaned.indexOf('\n\n');
    if (breakIdx > 0 && breakIdx < cleaned.length - 10) {
      cleaned = cleaned.substring(breakIdx + 2).trim();
    }
  }

  return cleaned.isEmpty ? text : cleaned;
}
