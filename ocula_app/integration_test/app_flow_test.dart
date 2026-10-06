// End-to-end flows on a simulator/device with real on-device data.
//
// Run:  flutter drive --driver=test_driver/integration_test.dart \
//         --target=integration_test/app_flow_test.dart -d <simulator-id>
//
// Expects onboarding to be complete and the simulator seeded with a contact
// named "Sarah Haddad" and at least one photo.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:ocula_app/main.dart' as app;

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  /// Pumps frames until [finder] matches (the orb animates forever, so
  /// pumpAndSettle would never return).
  Future<void> waitFor(
    WidgetTester tester,
    Finder finder, {
    Duration timeout = const Duration(seconds: 90),
  }) async {
    final end = DateTime.now().add(timeout);
    while (DateTime.now().isBefore(end)) {
      await tester.pump(const Duration(milliseconds: 250));
      if (finder.evaluate().isNotEmpty) return;
    }
    throw TestFailure('Timed out waiting for $finder');
  }

  Future<void> shot(WidgetTester tester, String name) async {
    await tester.pump(const Duration(milliseconds: 600));
    await binding.takeScreenshot(name);
  }

  Future<void> ask(WidgetTester tester, String text) async {
    await tester.enterText(find.byType(TextField), text);
    await tester.pump();
    // The previous reply may still be read aloud (Stop shows instead).
    await waitFor(
      tester,
      find.byTooltip('Send'),
      timeout: const Duration(seconds: 60),
    );
    await tester.tap(find.byTooltip('Send'));
    await tester.pump();
  }

  testWidgets('home, contact lookup, contact sheet, photos, briefing', (
    tester,
  ) async {
    app.main();
    await waitFor(tester, find.text('Ask Ocula…'));
    // Give the indexer time to pick up contacts/photos on first run.
    for (var i = 0; i < 40; i++) {
      await tester.pump(const Duration(milliseconds: 250));
    }
    await shot(tester, '01_home');

    // Contact lookup is answered directly from the index with a card.
    await tester.tap(find.text('Find a contact'));
    await tester.pump();
    await ask(tester, 'Find Sarah');
    await waitFor(tester, find.textContaining('Sarah Haddad'));
    await shot(tester, '02_contact_card');

    // Tapping the card opens the detail sheet with call / message / email.
    await tester.tap(find.text('Sarah Haddad').last);
    await waitFor(tester, find.text('Message'));
    expect(find.text('Call'), findsWidgets);
    expect(find.text('+44 7700 900123'), findsWidgets);
    await shot(tester, '03_contact_sheet');
    await tester.tapAt(const Offset(20, 80)); // dismiss sheet
    for (var i = 0; i < 6; i++) {
      await tester.pump(const Duration(milliseconds: 100));
    }

    // Photos render as thumbnails.
    await ask(tester, 'show my photos');
    await waitFor(
      tester,
      find.byKey(const ValueKey('photo-strip')),
      timeout: const Duration(seconds: 120),
    );
    await shot(tester, '04_photos');
  });
}
