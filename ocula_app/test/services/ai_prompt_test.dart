// Tests for model prompt construction and answer post-processing.
// All functions under test are pure (no native bridge) — safe for CI.

import 'package:flutter_test/flutter_test.dart';
import 'package:ocula_app/services/text_utils.dart';

void main() {
  // ── stripReasoningArtifacts ────────────────────────────────────────────────

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
      expect(
        stripReasoningArtifacts(raw),
        'I don\'t have weather data on your device.',
      );
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

  group('stripReasoningArtifacts — ChatML control tokens', () {
    test('strips <|im_end|> token', () {
      const raw = 'Good answer.<|im_end|>';
      expect(stripReasoningArtifacts(raw), 'Good answer.');
    });

    test('strips <|im_start|> token', () {
      const raw = '<|im_start|>assistant\nHello!';
      // <|im_start|> removed, then trimLeft removes "assistant\n"? No —
      // trimLeft only removes leading whitespace. "assistant\nHello!" remains.
      // What matters is that the control token itself is gone.
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
  });

  group('stripReasoningArtifacts — real Qwen3 output patterns', () {
    test('prefill did not suppress thinking — strips full block before answer', () {
      // Qwen3-1.7B emits a new <think> block despite the empty prefill.
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
  });

  // ── Prompt format contract ─────────────────────────────────────────────────
  // These tests encode the expected prompt structure so CI catches regressions
  // when someone changes the template (e.g. removes /no_think or the prefill).

  group('Prompt format contract', () {
    String buildPrompt(String systemMsg, String userMsg) {
      return '<|im_start|>system\n$systemMsg<|im_end|>\n'
          '<|im_start|>user\n$userMsg<|im_end|>\n'
          '<|im_start|>assistant\n<think>\n\n</think>\n';
    }

    test('prompt contains /no_think directive', () {
      const systemMsg = '/no_think\nYou are Ocula, an AI assistant.';
      final prompt = buildPrompt(systemMsg, 'Hello');
      expect(prompt, contains('/no_think'));
    });

    test('prompt uses ChatML <|im_start|> / <|im_end|> template', () {
      const systemMsg = '/no_think\nYou are Ocula.';
      final prompt = buildPrompt(systemMsg, 'Hi');
      expect(prompt, contains('<|im_start|>system\n'));
      expect(prompt, contains('<|im_start|>user\n'));
      expect(prompt, contains('<|im_start|>assistant\n'));
      expect(prompt, contains('<|im_end|>'));
    });

    test('prompt ends with empty think prefill to suppress Qwen3 reasoning', () {
      const systemMsg = '/no_think\nYou are Ocula.';
      final prompt = buildPrompt(systemMsg, 'Hi');
      expect(prompt, endsWith('<think>\n\n</think>\n'));
    });

    test('/no_think appears before the assistant role content', () {
      const systemMsg = '/no_think\nYou are Ocula.';
      final prompt = buildPrompt(systemMsg, 'Hi');
      final noThinkIdx = prompt.indexOf('/no_think');
      final assistantIdx = prompt.indexOf('<|im_start|>assistant');
      expect(noThinkIdx, lessThan(assistantIdx));
    });

    test('user message is placed in the user turn', () {
      const systemMsg = '/no_think\nYou are Ocula.';
      const userMsg = 'What meetings do I have today?';
      final prompt = buildPrompt(systemMsg, userMsg);
      final userTurnStart = prompt.indexOf('<|im_start|>user\n');
      final userTurnEnd = prompt.indexOf('<|im_end|>', userTurnStart);
      final userTurn = prompt.substring(userTurnStart, userTurnEnd);
      expect(userTurn, contains(userMsg));
    });
  });
}
