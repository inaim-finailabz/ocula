import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ocula_app/services/ocula_db.dart';
import 'package:ocula_app/widgets/result_cards.dart';

void main() {
  group('ContactInfo.fromAsset', () {
    test('parses indexed contact fields', () {
      final c = ContactInfo.fromAsset(LinkedAsset(
        sourceId: 'contact:tel:+15550101',
        assetType: 'contact',
        assetRef: 'tel:+15550101',
        label: 'Kate Bell',
        snippet: 'Name: Kate Bell\nPhone number: (555) 564-8583\n'
            'Email address: kate-bell@mac.com\nWorks at: Creative Consulting',
      ));
      expect(c.name, 'Kate Bell');
      expect(c.phone, '(555) 564-8583');
      expect(c.email, 'kate-bell@mac.com');
      expect(c.organization, 'Creative Consulting');
      expect(c.initials, 'KB');
    });

    test('falls back to the asset ref without a snippet', () {
      final c = ContactInfo.fromAsset(LinkedAsset(
        sourceId: 'contact:email:a@b.co',
        assetType: 'contact',
        assetRef: 'email:a@b.co',
        label: 'Anna',
      ));
      expect(c.name, 'Anna');
      expect(c.email, 'a@b.co');
      expect(c.phone, isNull);
    });
  });

  group('ResultCards layout', () {
    Widget host(List<LinkedAsset> assets) => MaterialApp(
          theme: ThemeData.dark(useMaterial3: true),
          home: Scaffold(body: ResultCards(assets: assets)),
        );

    LinkedAsset contact(String name, String phone) => LinkedAsset(
          sourceId: 'contact:tel:$phone',
          assetType: 'contact',
          assetRef: 'tel:$phone',
          label: name,
          snippet: 'Name: $name\nPhone number: $phone',
        );

    LinkedAsset photo(int i) => LinkedAsset(
          sourceId: 'photo:/missing/$i.jpg',
          assetType: 'photo',
          assetRef: '/missing/$i.jpg',
        );

    testWidgets('single contact shows inline call/message/email',
        (tester) async {
      await tester.pumpWidget(host([contact('Sarah Haddad', '+44 7700 900123')]));
      expect(find.text('Sarah Haddad'), findsOneWidget);
      expect(find.text('Call'), findsOneWidget);
      expect(find.text('Message'), findsOneWidget);
      expect(find.text('Email'), findsOneWidget);
    });

    testWidgets('several contacts do not show inline actions', (tester) async {
      await tester.pumpWidget(host([
        contact('Sarah Haddad', '+44 7700 900123'),
        contact('Sam Lee', '+44 7700 900456'),
      ]));
      expect(find.text('Message'), findsNothing);
    });

    testWidgets('photos show at most three thumbnails with a +N count',
        (tester) async {
      await tester.pumpWidget(host([for (var i = 0; i < 5; i++) photo(i)]));
      await tester.pump();
      expect(find.byType(AspectRatio), findsNWidgets(3));
      expect(find.text('+2'), findsOneWidget);
    });
  });
}
