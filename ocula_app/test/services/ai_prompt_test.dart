// Tests for model prompt construction and answer post-processing.
// All functions under test are pure (no native bridge) — safe for CI.

import 'package:flutter_test/flutter_test.dart';
import 'package:ocula_app/services/text_utils.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Helpers — mirrors ai_manager.dart prompt construction exactly.
// Keep in sync with ask() / askStream() when the template changes.
// ─────────────────────────────────────────────────────────────────────────────

/// Builds the full ChatML prompt as ai_manager.dart does.
/// /no_think goes in the USER turn (Qwen3 spec), NOT the system message.
String buildChatMLPrompt(String systemMsg, String userMsg) {
  // Append /no_think to user message exactly as ai_manager does
  final userMsgWithFlag = '$userMsg /no_think';
  return '<|im_start|>system\n$systemMsg<|im_end|>\n'
      '<|im_start|>user\n$userMsgWithFlag<|im_end|>\n'
      '<|im_start|>assistant\n<think>\n\n</think>\n';
}

void main() {
  // ── stripReasoningArtifacts — closed think blocks ─────────────────────────

  group('stripReasoningArtifacts — closed think blocks', () {
    test('strips <think>…</think> leaving the actual answer', () {
      const raw = '<think>\nI should answer concisely.\n</think>\nHello! How can I help?';
      expect(stripReasoningArtifacts(raw), 'Hello! How can I help?');
    });

    test('strips multi-line think block', () {
      const raw =
          '<think>\n'
          'The user asks about the weather.\n'
          'I have no data so I should say so.\n'
          '</think>\n'
          'I don\'t have weather data on your device.';
      expect(stripReasoningArtifacts(raw), 'I don\'t have weather data on your device.');
    });

    test('strips <thinking>…</thinking> variant', () {
      const raw = '<thinking>Some reasoning here.</thinking>The answer is 42.';
      expect(stripReasoningArtifacts(raw), 'The answer is 42.');
    });

    test('strips <reasoning>…</reasoning> variant', () {
      const raw = '<reasoning>Step-by-step work.</reasoning>Result: done.';
      expect(stripReasoningArtifacts(raw), 'Result: done.');
    });

    test('strips multiple think blocks', () {
      const raw =
          '<think>First thought.</think>Part one. '
          '<think>Second thought.</think>Part two.';
      expect(stripReasoningArtifacts(raw), 'Part one. Part two.');
    });

    test('preserves answer when no think block present', () {
      const raw = 'Your meeting is at 3pm today.';
      expect(stripReasoningArtifacts(raw), raw);
    });

    test('returns empty string when entire response is a closed think block', () {
      const raw = '<think>This is all thinking.</think>';
      expect(stripReasoningArtifacts(raw), '');
    });
  });

  // ── stripReasoningArtifacts — unclosed think blocks ───────────────────────

  group('stripReasoningArtifacts — unclosed think blocks', () {
    test('strips from open <think> to end when no closing tag', () {
      const raw = 'Preamble.\n<think>\nReasoning that never ends...';
      expect(stripReasoningArtifacts(raw), 'Preamble.');
    });

    test('returns empty when response starts with unclosed <think>', () {
      const raw = '<think>\nHalf-formed reasoning without a closing tag.';
      expect(stripReasoningArtifacts(raw), '');
    });

    test('strips trailing unclosed <think> after a real answer line', () {
      const raw = 'You have 3 events today.\n<think>Let me also mention';
      expect(stripReasoningArtifacts(raw), 'You have 3 events today.');
    });
  });

  // ── stripReasoningArtifacts — fenced blocks ───────────────────────────────

  group('stripReasoningArtifacts — fenced blocks', () {
    test('strips ```thinking fenced block', () {
      final raw = '```thinking\nSome internal analysis\n```\nActual answer.';
      expect(stripReasoningArtifacts(raw), 'Actual answer.');
    });

    test('strips ```reasoning fenced block', () {
      final raw = '```reasoning\nStep 1: ...\n```\nFinal answer.';
      expect(stripReasoningArtifacts(raw), 'Final answer.');
    });
  });

  // ── stripReasoningArtifacts — ChatML control tokens ──────────────────────

  group('stripReasoningArtifacts — ChatML control tokens', () {
    test('strips <|im_end|> token that leaks into response', () {
      const raw = 'Good answer.<|im_end|>';
      expect(stripReasoningArtifacts(raw), 'Good answer.');
    });

    test('strips <|im_start|> token', () {
      const raw = '<|im_start|>assistant\nHello!';
      final result = stripReasoningArtifacts(raw);
      expect(result.contains('<|im_start|>'), isFalse);
    });

    test('strips both control tokens when present', () {
      const raw = '<|im_start|>assistant\nHello!<|im_end|>';
      final result = stripReasoningArtifacts(raw);
      expect(result.contains('<|im_start|>'), isFalse);
      expect(result.contains('<|im_end|>'), isFalse);
      expect(result, contains('Hello!'));
    });

    test('strips multi-turn bleed-through (model generated past <|im_end|>)', () {
      // Happens when native bridge does not stop at <|im_end|>.
      // After the fix in llama_cpp_bridge.mm, the native layer strips it —
      // but the Dart layer must also handle any remnant.
      const raw =
          'Here is the answer.<|im_end|>\n'
          '<|im_start|>user\nWhat else?<|im_end|>\n'
          '<|im_start|>assistant\n<think>\nMore thinking.</think>\nExtra.';
      final result = stripReasoningArtifacts(raw);
      expect(result.contains('<|im_end|>'), isFalse);
      expect(result.contains('<|im_start|>'), isFalse);
      expect(result.contains('<think>'), isFalse);
      expect(result, contains('Here is the answer.'));
    });
  });

  // ── stripReasoningArtifacts — real Qwen3 output patterns ─────────────────

  group('stripReasoningArtifacts — real Qwen3 output patterns', () {
    test('prefill did not suppress thinking — strips full block before answer', () {
      const raw =
          '<think>\n'
          'The user said hello. I should respond in a friendly manner.\n'
          'Keep it short.\n'
          '</think>\n'
          'Hi there! How can I help you today?';
      expect(stripReasoningArtifacts(raw), 'Hi there! How can I help you today?');
    });

    test('generation cut off mid-think — returns empty rather than raw reasoning', () {
      const raw =
          '<think>\n'
          'The user wants calendar info. Let me check...\n'
          'Actually I should';
      expect(stripReasoningArtifacts(raw), '');
    });

    test('clean Qwen3 output after prefill suppression — passthrough', () {
      const raw = 'You have a team standup at 10am in Conference Room A.';
      expect(stripReasoningArtifacts(raw), raw);
    });

    test('think block with whitespace-padded tags', () {
      const raw = '< think >\nSome thinking.\n< /think >\nAnswer here.';
      expect(stripReasoningArtifacts(raw), 'Answer here.');
    });

    test('second think block generated after <|im_end|> bleed-through is stripped', () {
      // This is the exact bug pattern: model generates a good answer, then
      // <|im_end|> is not caught by native EOG check, model continues and
      // generates a second <think> block. Both layers must strip it.
      const raw =
          'Your next meeting is at 2pm.<|im_end|>\n'
          '<|im_start|>assistant\n'
          '<think>\n'
          'Let me reconsider...\n'
          '</think>\n'
          'Actually, you have two meetings.';
      final result = stripReasoningArtifacts(raw);
      // The <|im_end|> and <|im_start|> are stripped first
      expect(result.contains('<|im_end|>'), isFalse);
      expect(result.contains('<think>'), isFalse);
      // First real answer should be present
      expect(result, contains('Your next meeting is at 2pm.'));
    });
  });

  // ── Prompt format contract ────────────────────────────────────────────────
  // These tests encode the correct Qwen3 prompt structure.
  // /no_think MUST be in the USER turn, NOT the system message.
  // The empty <think></think> prefill in the assistant turn provides
  // a second layer of suppression.

  group('Prompt format contract — /no_think in USER turn (Qwen3 spec)', () {
    const systemMsg = 'You are Ocula, an AI assistant.';
    const userMsg = 'What meetings do I have today?';

    test('/no_think is NOT in the system message', () {
      final prompt = buildChatMLPrompt(systemMsg, userMsg);
      final systemStart = prompt.indexOf('<|im_start|>system\n');
      final systemEnd = prompt.indexOf('<|im_end|>', systemStart);
      final systemTurn = prompt.substring(systemStart, systemEnd);
      expect(systemTurn.contains('/no_think'), isFalse,
          reason: '/no_think must be in the user turn for Qwen3 to honour it');
    });

    test('/no_think IS in the user turn', () {
      final prompt = buildChatMLPrompt(systemMsg, userMsg);
      final userStart = prompt.indexOf('<|im_start|>user\n');
      final userEnd = prompt.indexOf('<|im_end|>', userStart);
      final userTurn = prompt.substring(userStart, userEnd);
      expect(userTurn, contains('/no_think'));
    });

    test('/no_think appears after the user query text', () {
      final prompt = buildChatMLPrompt(systemMsg, userMsg);
      final queryIdx = prompt.indexOf(userMsg);
      final noThinkIdx = prompt.indexOf('/no_think');
      expect(noThinkIdx, greaterThan(queryIdx));
    });

    test('prompt uses ChatML structure', () {
      final prompt = buildChatMLPrompt(systemMsg, userMsg);
      expect(prompt, contains('<|im_start|>system\n'));
      expect(prompt, contains('<|im_start|>user\n'));
      expect(prompt, contains('<|im_start|>assistant\n'));
      expect(prompt, contains('<|im_end|>'));
    });

    test('prompt ends with empty think prefill', () {
      final prompt = buildChatMLPrompt(systemMsg, userMsg);
      expect(prompt, endsWith('<think>\n\n</think>\n'));
    });

    test('/no_think is before the assistant turn', () {
      final prompt = buildChatMLPrompt(systemMsg, userMsg);
      final noThinkIdx = prompt.indexOf('/no_think');
      final assistantIdx = prompt.indexOf('<|im_start|>assistant');
      expect(noThinkIdx, lessThan(assistantIdx));
    });

    test('user query text is inside the user turn', () {
      final prompt = buildChatMLPrompt(systemMsg, userMsg);
      final userTurnStart = prompt.indexOf('<|im_start|>user\n');
      final userTurnEnd = prompt.indexOf('<|im_end|>', userTurnStart);
      final userTurn = prompt.substring(userTurnStart, userTurnEnd);
      expect(userTurn, contains(userMsg));
    });

    test('system message does not contain user query or /no_think', () {
      final prompt = buildChatMLPrompt(systemMsg, 'Tell me about my contacts.');
      final systemStart = prompt.indexOf('<|im_start|>system\n');
      final systemEnd = prompt.indexOf('<|im_end|>', systemStart);
      final system = prompt.substring(systemStart, systemEnd);
      expect(system.contains('/no_think'), isFalse);
      expect(system.contains('Tell me about my contacts.'), isFalse);
    });
  });

  // ── Stop sequence stripping ───────────────────────────────────────────────
  // These tests verify the Dart-layer defence against <|im_end|> leaking in
  // (in case the native bridge doesn't strip it before returning).

  group('Stop sequence stripping via stripReasoningArtifacts', () {
    test('strips trailing <|im_end|> from clean response', () {
      const raw = 'You have no events today.<|im_end|>';
      expect(stripReasoningArtifacts(raw), 'You have no events today.');
    });

    test('strips mid-response <|im_end|> and everything after', () {
      // If native didn't stop at <|im_end|>, extra content may follow.
      // The Dart layer strips all control tokens, leaving first-answer text.
      const raw = 'Meeting at 3pm.<|im_end|>\n<|im_start|>user\nFollowup?';
      final result = stripReasoningArtifacts(raw);
      expect(result.contains('<|im_end|>'), isFalse);
      expect(result.contains('<|im_start|>'), isFalse);
    });

    test('clean response without stop token passes through unchanged', () {
      const raw = 'Your contact John has phone 555-1234.';
      expect(stripReasoningArtifacts(raw), raw);
    });
  });

  // ── Double-suppression contract ───────────────────────────────────────────
  // Verifies that BOTH suppression layers are present simultaneously.

  group('Double-suppression contract (/no_think + think prefill)', () {
    test('prompt has both /no_think in user turn AND think prefill', () {
      const sysMsg = 'You are Ocula.';
      const usrMsg = 'Hello';
      final prompt = buildChatMLPrompt(sysMsg, usrMsg);
      // /no_think in user turn
      final userStart = prompt.indexOf('<|im_start|>user\n');
      final userEnd = prompt.indexOf('<|im_end|>', userStart);
      expect(prompt.substring(userStart, userEnd), contains('/no_think'));
      // think prefill in assistant turn
      expect(prompt, contains('<|im_start|>assistant\n<think>\n\n</think>\n'));
    });

    test('even if model ignores /no_think, stripReasoningArtifacts catches output', () {
      // Simulate: model ignored both suppressors and emitted a think block
      const modelOutput =
          '<think>\nLet me reason about this carefully.\n</think>\n'
          'You have 2 events today.';
      expect(stripReasoningArtifacts(modelOutput), 'You have 2 events today.');
    });

    test('even if model emits unclosed think (token budget hit mid-thought)', () {
      const modelOutput = '<think>\nStarting to reason about';
      expect(stripReasoningArtifacts(modelOutput), '');
    });
  });
}
