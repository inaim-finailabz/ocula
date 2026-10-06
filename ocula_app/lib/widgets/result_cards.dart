import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:photo_manager/photo_manager.dart';
import 'package:share_plus/share_plus.dart';
import 'package:url_launcher/url_launcher.dart';

import '../services/ocula_db.dart';

/// Structured, tappable results under an assistant reply.
///
/// Built straight from the retrieved records (not from the model's text) so
/// contacts, photos, documents and events always show their real data:
/// contacts open a detail sheet with call / message / email, photos show
/// thumbnails and open full-screen, documents show an excerpt and open a
/// preview, events deep-link into the Calendar app.
class ResultCards extends StatefulWidget {
  final List<LinkedAsset> assets;
  const ResultCards({super.key, required this.assets});

  @override
  State<ResultCards> createState() => _ResultCardsState();
}

class _ResultCardsState extends State<ResultCards> {
  static const _collapsedCount = 3;
  final Set<String> _expandedGroups = {};

  @override
  Widget build(BuildContext context) {
    final photos = <LinkedAsset>[];
    final groups = <String, List<LinkedAsset>>{};
    for (final a in widget.assets) {
      if (a.assetType == 'photo' || a.assetType == 'video') {
        photos.add(a);
      } else {
        groups.putIfAbsent(a.assetType, () => []).add(a);
      }
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      mainAxisSize: MainAxisSize.min,
      children: [
        if (photos.isNotEmpty) _PhotoStrip(photos: photos),
        for (final entry in groups.entries)
          ..._buildGroup(entry.key, entry.value),
      ],
    );
  }

  List<Widget> _buildGroup(String type, List<LinkedAsset> items) {
    final expanded = _expandedGroups.contains(type);
    final visible = expanded ? items : items.take(_collapsedCount).toList();
    final hidden = items.length - visible.length;
    final colors = Theme.of(context).colorScheme;
    return [
      for (final a in visible) _ResultTile(asset: a),
      if (hidden > 0)
        Align(
          alignment: Alignment.centerLeft,
          child: TextButton(
            style: TextButton.styleFrom(
              visualDensity: VisualDensity.compact,
              padding: const EdgeInsets.symmetric(horizontal: 4),
            ),
            onPressed: () => setState(() => _expandedGroups.add(type)),
            child: Text(
              'Show $hidden more',
              style: TextStyle(fontSize: 12, color: colors.primary),
            ),
          ),
        ),
    ];
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Parsed record fields
// ─────────────────────────────────────────────────────────────────────────

/// Contact fields parsed from the indexed chunk
/// ("Name: …\nPhone number: …\nEmail address: …\nWorks at: …") with the
/// asset ref (`tel:…`, `email:…`, `name:…`) as fallback.
class ContactInfo {
  final String name;
  final String? phone;
  final String? email;
  final String? organization;

  const ContactInfo({
    required this.name,
    this.phone,
    this.email,
    this.organization,
  });

  factory ContactInfo.fromAsset(LinkedAsset a) {
    String? field(String key) {
      final m = RegExp(
        '^$key:\\s*(.+)\$',
        multiLine: true,
      ).firstMatch(a.snippet ?? '');
      final v = m?.group(1)?.trim();
      return (v == null || v.isEmpty) ? null : v;
    }

    final ref = a.assetRef;
    return ContactInfo(
      name: field('Name') ?? a.label ?? ref,
      phone:
          field('Phone number') ??
          (ref.startsWith('tel:') ? ref.substring(4) : null),
      email:
          field('Email address') ??
          (ref.startsWith('email:') ? ref.substring(6) : null),
      organization: field('Works at'),
    );
  }

  String get initials {
    final parts = name.trim().split(RegExp(r'\s+')).where((p) => p.isNotEmpty);
    return parts.take(2).map((p) => p[0].toUpperCase()).join();
  }
}

String _excerpt(String? snippet, {int max = 160}) {
  final t = (snippet ?? '').replaceAll(RegExp(r'\s+'), ' ').trim();
  return t.length <= max ? t : '${t.substring(0, max)}…';
}

String _fileName(LinkedAsset a) =>
    (a.label?.isNotEmpty ?? false) ? a.label! : a.assetRef.split('/').last;

String _filePath(LinkedAsset a) => a.assetRef.startsWith('file://')
    ? Uri.parse(a.assetRef).toFilePath()
    : a.assetRef;

IconData _fileIcon(String name) {
  final ext = name.contains('.') ? name.split('.').last.toLowerCase() : '';
  switch (ext) {
    case 'pdf':
      return Icons.picture_as_pdf_outlined;
    case 'doc':
    case 'docx':
    case 'txt':
    case 'md':
    case 'rtf':
      return Icons.description_outlined;
    case 'xls':
    case 'xlsx':
    case 'csv':
      return Icons.table_chart_outlined;
    case 'ppt':
    case 'pptx':
      return Icons.slideshow_outlined;
    default:
      return Icons.insert_drive_file_outlined;
  }
}

Future<void> _launch(BuildContext context, Uri uri) async {
  bool ok = false;
  try {
    ok = await launchUrl(uri, mode: LaunchMode.externalApplication);
  } catch (_) {}
  if (!ok && context.mounted) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          'Could not open ${uri.scheme == 'tel' ? 'phone' : uri.toString()}',
        ),
      ),
    );
  }
}

Uri? _calendarUri(LinkedAsset a) {
  final millis = int.tryParse(a.assetRef.replaceFirst('cal:', ''));
  if (millis == null) return null;
  if (Platform.isIOS || Platform.isMacOS) {
    // calshow: takes seconds since 2001-01-01 UTC.
    const macEpochOffsetSeconds = 978307200;
    return Uri.parse(
      'calshow:${(millis / 1000).round() - macEpochOffsetSeconds}',
    );
  }
  if (Platform.isAndroid)
    return Uri.parse('content://com.android.calendar/time/$millis');
  return null;
}

// ─────────────────────────────────────────────────────────────────────────
// Tiles
// ─────────────────────────────────────────────────────────────────────────

class _ResultTile extends StatelessWidget {
  final LinkedAsset asset;
  const _ResultTile({required this.asset});

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    late final Widget leading;
    late final String title;
    String? subtitle;
    late final VoidCallback onTap;
    Widget? trailing;

    switch (asset.assetType) {
      case 'contact':
      case 'phone':
        final c = ContactInfo.fromAsset(asset);
        leading = CircleAvatar(
          radius: 16,
          backgroundColor: colors.primaryContainer,
          child: Text(
            c.initials,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: colors.onPrimaryContainer,
            ),
          ),
        );
        title = c.name;
        subtitle = [
          c.phone,
          c.organization ?? c.email,
        ].whereType<String>().join(' · ');
        onTap = () => showContactSheet(context, c);
        if (c.phone != null) {
          trailing = IconButton(
            tooltip: 'Call',
            visualDensity: VisualDensity.compact,
            icon: Icon(Icons.call_outlined, size: 18, color: colors.primary),
            onPressed: () => _launch(
              context,
              Uri(
                scheme: 'tel',
                path: c.phone!.replaceAll(RegExp(r'[^0-9+]'), ''),
              ),
            ),
          );
        }
      case 'file':
        final name = _fileName(asset);
        leading = _IconBox(icon: _fileIcon(name));
        title = name;
        subtitle = _excerpt(asset.snippet, max: 90);
        onTap = () => showDocumentSheet(context, asset);
      case 'calendar':
        leading = const _IconBox(icon: Icons.event_outlined);
        final lines = (asset.snippet ?? '').split('\n');
        // Indexed labels look like "Title — 2026-10-05 09:00".
        title = (asset.label ?? '').split(' — ').first;
        subtitle = asset.label?.contains(' — ') == true
            ? asset.label!.split(' — ').last
            : lines.where((l) => l.trim().isNotEmpty).join(' · ');
        onTap = () {
          final uri = _calendarUri(asset);
          if (uri != null) _launch(context, uri);
        };
      case 'email':
        final addr = asset.assetRef
            .replaceFirst('mailto:', '')
            .replaceFirst('email:', '');
        leading = const _IconBox(icon: Icons.mail_outline);
        title = asset.label ?? addr;
        subtitle = asset.label == null ? null : addr;
        onTap = () => _launch(context, Uri(scheme: 'mailto', path: addr));
      default:
        leading = const _IconBox(icon: Icons.link);
        title = asset.label ?? asset.assetRef;
        onTap = () {
          final uri = Uri.tryParse(asset.assetRef);
          if (uri != null)
            _launch(
              context,
              uri.hasScheme ? uri : Uri.parse('https://${asset.assetRef}'),
            );
        };
    }

    return Padding(
      padding: const EdgeInsets.only(top: 6),
      child: Material(
        color: colors.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(12),
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(8, 6, 4, 6),
            child: Row(
              children: [
                leading,
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        title,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                          color: colors.onSurface,
                        ),
                      ),
                      if (subtitle != null && subtitle.isNotEmpty)
                        Text(
                          subtitle,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            fontSize: 12,
                            height: 1.3,
                            color: colors.onSurfaceVariant,
                          ),
                        ),
                    ],
                  ),
                ),
                trailing ??
                    Icon(
                      Icons.chevron_right,
                      size: 18,
                      color: colors.onSurfaceVariant,
                    ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _IconBox extends StatelessWidget {
  final IconData icon;
  const _IconBox({required this.icon});

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Container(
      width: 32,
      height: 32,
      decoration: BoxDecoration(
        color: colors.primaryContainer.withAlpha(140),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Icon(icon, size: 18, color: colors.primary),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Contact detail sheet
// ─────────────────────────────────────────────────────────────────────────

Future<void> showContactSheet(BuildContext context, ContactInfo c) {
  return showModalBottomSheet(
    context: context,
    showDragHandle: true,
    builder: (ctx) {
      final colors = Theme.of(ctx).colorScheme;
      final digits = c.phone?.replaceAll(RegExp(r'[^0-9+]'), '');
      Widget action(IconData icon, String label, VoidCallback? onTap) =>
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4),
              child: FilledButton.tonal(
                onPressed: onTap,
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 10),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(icon, size: 20),
                    const SizedBox(height: 2),
                    Text(label, style: const TextStyle(fontSize: 12)),
                  ],
                ),
              ),
            ),
          );
      Widget row(IconData icon, String label, String value) => ListTile(
        dense: true,
        leading: Icon(icon, color: colors.primary),
        title: Text(
          label,
          style: TextStyle(fontSize: 12, color: colors.onSurfaceVariant),
        ),
        subtitle: Text(
          value,
          style: TextStyle(fontSize: 15, color: colors.onSurface),
        ),
        trailing: IconButton(
          tooltip: 'Copy',
          icon: const Icon(Icons.copy, size: 18),
          onPressed: () {
            Clipboard.setData(ClipboardData(text: value));
            ScaffoldMessenger.of(
              ctx,
            ).showSnackBar(SnackBar(content: Text('$label copied')));
          },
        ),
      );

      return SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              CircleAvatar(
                radius: 32,
                backgroundColor: colors.primaryContainer,
                child: Text(
                  c.initials,
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w600,
                    color: colors.onPrimaryContainer,
                  ),
                ),
              ),
              const SizedBox(height: 8),
              Text(
                c.name,
                style: Theme.of(ctx).textTheme.titleLarge,
                textAlign: TextAlign.center,
              ),
              if (c.organization != null)
                Text(
                  c.organization!,
                  style: TextStyle(color: colors.onSurfaceVariant),
                ),
              const SizedBox(height: 16),
              Row(
                children: [
                  action(
                    Icons.call,
                    'Call',
                    digits == null || digits.isEmpty
                        ? null
                        : () => _launch(ctx, Uri(scheme: 'tel', path: digits)),
                  ),
                  action(
                    Icons.message,
                    'Message',
                    digits == null || digits.isEmpty
                        ? null
                        : () => _launch(ctx, Uri(scheme: 'sms', path: digits)),
                  ),
                  action(
                    Icons.mail,
                    'Email',
                    c.email == null
                        ? null
                        : () => _launch(
                            ctx,
                            Uri(scheme: 'mailto', path: c.email),
                          ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              if (c.phone != null) row(Icons.phone_outlined, 'Phone', c.phone!),
              if (c.email != null) row(Icons.mail_outline, 'Email', c.email!),
            ],
          ),
        ),
      );
    },
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Document preview sheet
// ─────────────────────────────────────────────────────────────────────────

Future<void> showDocumentSheet(BuildContext context, LinkedAsset a) {
  final name = _fileName(a);
  final path = _filePath(a);
  return showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    showDragHandle: true,
    builder: (ctx) {
      final colors = Theme.of(ctx).colorScheme;
      return DraggableScrollableSheet(
        expand: false,
        initialChildSize: 0.6,
        maxChildSize: 0.92,
        builder: (ctx, scroll) => Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(
                children: [
                  _IconBox(icon: _fileIcon(name)),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      name,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: Theme.of(ctx).textTheme.titleMedium,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                'Matching excerpt',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: colors.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 4),
              Expanded(
                child: Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: colors.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: SingleChildScrollView(
                    controller: scroll,
                    child: SelectableText(
                      (a.snippet ?? '').trim().isEmpty
                          ? 'No preview text available.'
                          : a.snippet!.trim(),
                      style: const TextStyle(fontSize: 14, height: 1.45),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              FilledButton.icon(
                icon: const Icon(Icons.ios_share),
                label: const Text('Open / Share'),
                onPressed: () async {
                  if (await File(path).exists()) {
                    await Share.shareXFiles([XFile(path)], subject: name);
                  } else if (ctx.mounted) {
                    ScaffoldMessenger.of(ctx).showSnackBar(
                      SnackBar(content: Text('File not found: $name')),
                    );
                  }
                },
              ),
            ],
          ),
        ),
      );
    },
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Photos
// ─────────────────────────────────────────────────────────────────────────

/// Loads a photo's bytes: the cached file if it still exists, otherwise the
/// photo library asset (sourceId `photo:<assetId>`), since iOS may purge the
/// cache copy the indexer saw.
Future<Uint8List?> _photoBytes(LinkedAsset a, {required bool full}) async {
  final file = File(_filePath(a));
  if (await file.exists()) return file.readAsBytes();
  final id = a.sourceId.startsWith('photo:') ? a.sourceId.substring(6) : null;
  if (id == null || id.startsWith('/')) return null;
  try {
    final entity = await AssetEntity.fromId(id);
    if (entity == null) return null;
    return full
        ? await entity.originBytes
        : await entity.thumbnailDataWithSize(const ThumbnailSize(240, 240));
  } catch (_) {
    return null;
  }
}

class _PhotoStrip extends StatelessWidget {
  final List<LinkedAsset> photos;
  const _PhotoStrip({required this.photos});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 6),
      child: SizedBox(
        key: const ValueKey('photo-strip'),
        height: 92,
        child: ListView.separated(
          scrollDirection: Axis.horizontal,
          itemCount: photos.length,
          separatorBuilder: (_, _) => const SizedBox(width: 6),
          itemBuilder: (ctx, i) => GestureDetector(
            onTap: () => Navigator.of(ctx).push(
              MaterialPageRoute(
                fullscreenDialog: true,
                builder: (_) => _PhotoViewer(photos: photos, initialIndex: i),
              ),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: SizedBox(
                width: 92,
                height: 92,
                child: _PhotoThumb(asset: photos[i]),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _PhotoThumb extends StatelessWidget {
  final LinkedAsset asset;
  final bool full;
  const _PhotoThumb({required this.asset, this.full = false});

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return FutureBuilder<Uint8List?>(
      future: _photoBytes(asset, full: full),
      builder: (ctx, snap) {
        final bytes = snap.data;
        if (bytes != null) {
          return Image.memory(
            bytes,
            fit: full ? BoxFit.contain : BoxFit.cover,
            cacheWidth: full ? null : 240,
            gaplessPlayback: true,
            errorBuilder: (_, _, _) => _placeholder(colors),
          );
        }
        return _placeholder(
          colors,
          loading: snap.connectionState != ConnectionState.done,
        );
      },
    );
  }

  Widget _placeholder(ColorScheme colors, {bool loading = false}) => Container(
    color: colors.surfaceContainerHighest,
    alignment: Alignment.center,
    child: loading
        ? const SizedBox(
            width: 16,
            height: 16,
            child: CircularProgressIndicator(strokeWidth: 2),
          )
        : Icon(Icons.broken_image_outlined, color: colors.onSurfaceVariant),
  );
}

class _PhotoViewer extends StatefulWidget {
  final List<LinkedAsset> photos;
  final int initialIndex;
  const _PhotoViewer({required this.photos, required this.initialIndex});

  @override
  State<_PhotoViewer> createState() => _PhotoViewerState();
}

class _PhotoViewerState extends State<_PhotoViewer> {
  late final PageController _pages = PageController(
    initialPage: widget.initialIndex,
  );
  late int _index = widget.initialIndex;

  @override
  void dispose() {
    _pages.dispose();
    super.dispose();
  }

  Future<void> _share() async {
    final a = widget.photos[_index];
    final bytes = await _photoBytes(a, full: true);
    if (bytes == null) return;
    await Share.shareXFiles([
      XFile.fromData(bytes, mimeType: 'image/jpeg', name: 'photo.jpg'),
    ]);
  }

  @override
  Widget build(BuildContext context) {
    final a = widget.photos[_index];
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Text(
          widget.photos.length > 1
              ? '${_index + 1} of ${widget.photos.length}'
              : '',
          style: const TextStyle(fontSize: 15),
        ),
        actions: [
          IconButton(
            tooltip: 'Share',
            icon: const Icon(Icons.ios_share),
            onPressed: _share,
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: PageView.builder(
              controller: _pages,
              itemCount: widget.photos.length,
              onPageChanged: (i) => setState(() => _index = i),
              itemBuilder: (_, i) => InteractiveViewer(
                child: Center(
                  child: _PhotoThumb(asset: widget.photos[i], full: true),
                ),
              ),
            ),
          ),
          if ((a.label ?? '').isNotEmpty)
            SafeArea(
              top: false,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Text(
                  a.label!,
                  maxLines: 3,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(color: Colors.white70, fontSize: 13),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
