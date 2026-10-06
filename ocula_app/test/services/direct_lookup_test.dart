import 'package:flutter_test/flutter_test.dart';
import 'package:ocula_app/services/direct_lookup.dart';
import 'package:ocula_app/services/ocula_db.dart';

RagSearchResult row(String sourceId, String text) => RagSearchResult(
  id: 0,
  text: text,
  source: 'contact',
  sourceId: sourceId,
  score: 1,
  timestamp: DateTime(2026),
);

void main() {
  group('terms', () {
    test('strips filler, type words and possessives', () {
      expect(DirectLookup.terms("What's Kate's phone number?"), ['kate']);
      expect(DirectLookup.terms('find my passport pdf'), ['passport']);
      expect(DirectLookup.terms('show photos of the beach'), ['beach']);
    });
  });

  group('isLookup', () {
    test('plain retrieval requests', () {
      expect(DirectLookup.isLookup('find kate'), isTrue);
      expect(DirectLookup.isLookup("what's John's number"), isTrue);
      expect(DirectLookup.isLookup('kate bell'), isTrue);
      expect(DirectLookup.isLookup('show me beach photos'), isTrue);
    });
    test('reasoning requests go to the model', () {
      expect(DirectLookup.isLookup('summarise the lease contract'), isFalse);
      expect(DirectLookup.isLookup('when did I last meet Kate at the office'), isFalse);
      expect(DirectLookup.isLookup('how many photos did I take in Rome last summer'), isFalse);
    });
  });

  group('rank', () {
    test('prefers rows matching every term and dedupes by source', () {
      final rows = [
        row('contact:tel:1', 'Name: Kate Smith'),
        row('contact:tel:2', 'Name: Kate Bell'),
        row('contact:tel:2', 'Name: Kate Bell'),
        row('contact:tel:3', 'Name: John Bell'),
      ];
      final ranked = DirectLookup.rank(rows, ['kate', 'bell']);
      expect(ranked.map((r) => r.sourceId), ['contact:tel:2']);
    });
  });

  group('answer', () {
    test('single contact includes the number', () {
      final a = LinkedAsset(
        sourceId: 'contact:tel:1',
        assetType: 'contact',
        assetRef: 'tel:1',
        label: 'Kate Bell',
        snippet: 'Name: Kate Bell\nPhone number: (555) 564-8583',
      );
      expect(DirectLookup.answer('contact', [a], ['kate']), 'Here’s Kate Bell — (555) 564-8583.');
    });
  });
}
