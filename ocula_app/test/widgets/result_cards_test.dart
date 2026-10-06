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
}
