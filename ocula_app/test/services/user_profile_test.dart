import 'package:flutter_test/flutter_test.dart';
import 'package:ocula_app/services/user_profile.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  setUp(() async {
    SharedPreferences.setMockInitialValues({});
    await UserProfile().reset();
  });

  test('whole-word matching ignores substrings', () async {
    final p = UserProfile();
    // "great" contains "eat", "brunch" contains "run" — neither should count.
    await p.recordInteraction(userQuery: 'brunch was great', assistantResponse: '');
    expect(p.score('meal_suggestions'), 0);
    expect(p.score('gym_reminders'), 0);
    expect(p.score('briefing_detailed'), greaterThan(0));
  });

  test("don't forget is not a negative signal", () async {
    final p = UserProfile();
    await p.recordInteraction(userQuery: "don't forget to call mum", assistantResponse: '');
    expect(p.score('briefing_detailed'), 0);
  });

  test('profile persists and reaches the prompt after 2 signals', () async {
    final p = UserProfile();
    for (var i = 0; i < 3; i++) {
      await p.recordInteraction(userQuery: 'what should I eat for dinner', assistantResponse: '');
    }
    expect(p.toPromptContext(), contains('meal_suggestions'));
    final prefs = await SharedPreferences.getInstance();
    expect(prefs.getString('user_profile_v1'), contains('meal_suggestions'));
  });
}
