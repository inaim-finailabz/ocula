import 'package:flutter_test/flutter_test.dart';
import 'package:ocula_app/services/speech_service.dart';

void main() {
  group('SpeechService.toSpeakable', () {
    test('strips markdown, bullets and citations', () {
      final out = SpeechService.toSpeakable(
        'Where I found it: **Contacts** [1]\n'
        '- Sarah Haddad — +44 7700 900123\n'
        '- Meeting on 10/06/2026 at 3pm',
      );
      expect(out,
          'Where I found it: Contacts. Sarah Haddad, +44 7700 900123. '
          'Meeting on 10/06/2026 at 3pm');
    });

    test('drops URLs, paths and emoji but keeps link labels', () {
      final out = SpeechService.toSpeakable(
        'See [the report](https://x.com/a) 📄 at /var/mobile/Docs/a.pdf '
        'or https://example.com/page.',
      );
      expect(out, 'See the report at or');
    });
  });
}
