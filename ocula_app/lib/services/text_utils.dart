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

/// Removes model reasoning/thinking traces so the UI only shows the final answer.
///
/// Handles:
/// - Closed `<think>…</think>` blocks (and `<thinking>`, `<reasoning>` variants)
/// - Fenced ` ```thinking … ``` ` blocks
/// - Unclosed open tags (generation stopped mid-thought)
/// - ChatML control tokens (`<|im_start|>`, `<|im_end|>`)
String stripReasoningArtifacts(String text) {
  var t = text
      .replaceAll('<|im_end|>', '')
      .replaceAll('<|im_start|>', '')
      .trimLeft();

  // Remove closed reasoning blocks.
  t = t.replaceAll(
    RegExp(
      r'<\s*(think|thinking|reasoning)\b[^>]*>.*?<\s*/\s*\1\s*>',
      caseSensitive: false,
      dotAll: true,
    ),
    '',
  );

  // Remove fenced reasoning blocks some models emit.
  t = t.replaceAll(
    RegExp(
      r'```\s*(thinking|reasoning)\b[\s\S]*?```',
      caseSensitive: false,
      dotAll: true,
    ),
    '',
  );

  // Hide any unfinished reasoning block (generation ran out of tokens mid-thought).
  final openTag = RegExp(
    r'<\s*(think|thinking|reasoning)\b[^>]*>',
    caseSensitive: false,
  ).firstMatch(t);
  if (openTag != null) {
    t = t.substring(0, openTag.start).trimRight();
  }

  return t.trim();
}
