import 'package:flutter_test/flutter_test.dart';
import 'package:ocula_app/services/text_utils.dart';

void main() {
  group('stripLeakedContext', () {
    // ── Data: / Phone data: prefix ──

    test('strips echoed Data: block with Question marker', () {
      const input =
          'Data: CONTACT: John Doe — phone: 555-1234\n'
          'Question: Who is John?\n'
          'John Doe is one of your contacts.';
      final result = stripLeakedContext(input);
      expect(result, equals('John Doe is one of your contacts.'));
    });

    test('strips echoed Phone data: block with Q marker', () {
      const input =
          'Phone data: CALENDAR EVENT: Meeting at 3pm\n'
          'Q: What\'s on my calendar?\n'
          'You have a meeting at 3pm today.';
      final result = stripLeakedContext(input);
      expect(result, equals('You have a meeting at 3pm today.'));
    });

    test('strips Data: block via double-newline when no Q marker', () {
      const input =
          'Data: CONTACT: Anna Haro — email: anna@example.com\n\n'
          'Anna Haro is in your contacts with email anna@example.com.';
      final result = stripLeakedContext(input);
      expect(result, equals('Anna Haro is in your contacts with email anna@example.com.'));
    });

    test('does NOT strip Data: prefix if remaining text too short', () {
      // breakIdx would be at 10, remaining after break < 10 chars
      const input = 'Data: abc\n\nshort';
      final result = stripLeakedContext(input);
      // Should not strip because remaining after \n\n is < 10 chars
      expect(result, equals(input));
    });

    // ── Section headers ──

    test('strips [Linked entities] header line', () {
      const input = '[Linked entities] John → knows → Jane\nJohn knows Jane.';
      final result = stripLeakedContext(input);
      expect(result, equals('John knows Jane.'));
    });

    test('strips [Recent conversations] header line', () {
      const input = '[Recent conversations] User asked about weather\nThe weather is nice.';
      final result = stripLeakedContext(input);
      expect(result, equals('The weather is nice.'));
    });

    test('strips multiple section headers', () {
      const input =
          '[Linked entities] foo\n'
          '[Knowledge] bar\n'
          'The actual answer is here.';
      final result = stripLeakedContext(input);
      expect(result, equals('The actual answer is here.'));
    });

    test('strips [Note] header line', () {
      const input = '[Note] Some internal note\nReal answer.';
      final result = stripLeakedContext(input);
      expect(result, equals('Real answer.'));
    });

    // ── Context labels ──

    test('strips CONTACT: prefix with double newline', () {
      const input =
          'CONTACT: John Doe — phone: 555-1234, email: john@example.com\n\n'
          'John Doe is one of your saved contacts.';
      final result = stripLeakedContext(input);
      expect(result, equals('John Doe is one of your saved contacts.'));
    });

    test('strips CALENDAR EVENT: prefix with double newline', () {
      const input =
          'CALENDAR EVENT: Team standup | 2026-02-13 10:00 | Conference Room A\n\n'
          'You have Team standup at 10am in Conference Room A.';
      final result = stripLeakedContext(input);
      expect(result, equals('You have Team standup at 10am in Conference Room A.'));
    });

    test('strips PREVIOUS CONVERSATION: prefix', () {
      const input =
          'PREVIOUS CONVERSATION: User asked about cats\n\n'
          'Based on our earlier conversation, you were curious about cats.';
      final result = stripLeakedContext(input);
      expect(result, equals('Based on our earlier conversation, you were curious about cats.'));
    });

    test('strips FILE: prefix', () {
      const input =
          'FILE: report.pdf — Q4 financial results summary\n\n'
          'The Q4 report shows positive growth.';
      final result = stripLeakedContext(input);
      expect(result, equals('The Q4 report shows positive growth.'));
    });

    // ── Passthrough cases ──

    test('returns clean text unchanged', () {
      const input = 'This is a perfectly normal response about the weather.';
      final result = stripLeakedContext(input);
      expect(result, equals(input));
    });

    test('returns original text if stripping would leave empty string', () {
      const input = 'Data: only echoed content here';
      final result = stripLeakedContext(input);
      // No double-newline break and no Q/Question marker — can't strip
      expect(result, equals(input));
    });

    test('handles empty input', () {
      final result = stripLeakedContext('');
      expect(result, equals(''));
    });

    // ── Compound scenarios ──

    test('strips Data: block AND section headers together', () {
      const input =
          'Data: CONTACT: Jane Smith\n\n'
          '[Linked entities] Jane → colleague → Bob\n'
          'Jane Smith is your colleague who works with Bob.';
      final result = stripLeakedContext(input);
      expect(result, equals('Jane Smith is your colleague who works with Bob.'));
    });

    test('real-world model echo from simulator', () {
      // Simulates the actual garbage output seen on the simulator
      const input =
          'Data: PREVIOUS CONVERSATION: User: i hope you\'re having a wonderful day!\n'
          'Q: my calendar point meant for today\n'
          'You have 3 events today: Team standup at 10am, Lunch with Sarah at 12pm, '
          'and Code review at 3pm.';
      final result = stripLeakedContext(input);
      expect(
        result,
        equals(
          'You have 3 events today: Team standup at 10am, Lunch with Sarah at 12pm, '
          'and Code review at 3pm.',
        ),
      );
    });
  });
}
