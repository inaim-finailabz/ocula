import 'dart:io';
import 'package:flutter/material.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

/// Handles PDF generation and sharing (Pro tier feature).
class ExportService {
  /// Generate a PDF from AI analysis content and open the system share sheet.
  ///
  /// [origin] is required on iPad for the share-sheet popover anchor.
  Future<void> exportAndShare(String aiContent, {String title = 'AI Research Report', Rect? origin}) async {
    final pdf = pw.Document();

    pdf.addPage(
      pw.Page(
        build: (context) => pw.Column(
          crossAxisAlignment: pw.CrossAxisAlignment.start,
          children: [
            pw.Text(title, style: pw.TextStyle(fontSize: 24, fontWeight: pw.FontWeight.bold)),
            pw.SizedBox(height: 16),
            pw.Text(aiContent, style: const pw.TextStyle(fontSize: 12)),
          ],
        ),
      ),
    );

    final tempDir = await getTemporaryDirectory();
    final file = File('${tempDir.path}/report.pdf');
    await file.writeAsBytes(await pdf.save());

    // Trigger system share sheet (Email, WhatsApp, Slack, etc.)
    await Share.shareXFiles(
      [XFile(file.path)],
      text: title,
      sharePositionOrigin: origin ?? const Rect.fromLTWH(0, 0, 100, 100),
    );
  }
}
