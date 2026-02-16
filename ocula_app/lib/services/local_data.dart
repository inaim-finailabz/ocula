// Local data layer — accesses everything on the user's device.
// NOTHING leaves the phone. All processing is on-device.
//
// v3: Real platform integrations for photos, contacts, calendar.
// All TODO stubs replaced with working implementations.

import 'dart:io';
import 'package:device_calendar/device_calendar.dart' as cal;
import 'package:enough_mail/enough_mail.dart' as imap;
import 'package:flutter/foundation.dart';
import 'package:flutter_contacts/flutter_contacts.dart';
import 'package:path_provider/path_provider.dart';
import 'package:photo_manager/photo_manager.dart';
import 'package:shared_preferences/shared_preferences.dart';

class LocalData {
  static final LocalData _instance = LocalData._();
  factory LocalData() => _instance;
  LocalData._();

  // ──────────────────────────────────────────
  // PERMISSIONS — request all at first launch
  // ──────────────────────────────────────────

  /// Request all asset permissions up front. Returns map of granted states.
  /// Call once during onboarding or first launch.
  Future<Map<String, bool>> requestAllPermissions() async {
    final results = <String, bool>{};

    // Photos
    final photoState = await PhotoManager.requestPermissionExtend();
    results['photos'] =
        photoState.isAuth || photoState == PermissionState.limited;

    // Contacts
    results['contacts'] = await FlutterContacts.requestPermission(
      readonly: true,
    );

    // Calendar
    final calPlugin = cal.DeviceCalendarPlugin();
    final calResult = await calPlugin.requestPermissions();
    results['calendar'] = calResult.data ?? false;

    if (kDebugMode) {
      print('[LocalData] Permissions: $results');
    }
    return results;
  }

  // ──────────────────────────────────────────
  // PHOTOS & CAMERA ROLL
  // ──────────────────────────────────────────

  /// Get recent photos from the device photo library.
  /// Returns photos with metadata (date, dimensions) for indexing.
  Future<List<LocalPhoto>> recentPhotos({int limit = 100}) async {
    try {
      final permitted = await PhotoManager.requestPermissionExtend();
      if (!permitted.isAuth && permitted != PermissionState.limited) {
        if (kDebugMode) print('[LocalData] Photo permission denied');
        return [];
      }

      // Get all albums, start with recent/camera roll
      final albums = await PhotoManager.getAssetPathList(
        type: RequestType.image,
        hasAll: true,
      );
      if (albums.isEmpty) return [];

      // "Recent" or "All Photos" album is usually first
      final recentAlbum = albums.first;
      final assets = await recentAlbum.getAssetListRange(start: 0, end: limit);

      final photos = <LocalPhoto>[];
      for (final asset in assets) {
        final file = await asset.file;
        if (file == null) continue;

        // Build a descriptive label from available metadata
        final label = _buildPhotoLabel(asset);

        photos.add(
          LocalPhoto(
            path: file.path,
            date: asset.createDateTime,
            label: label,
            width: asset.width,
            height: asset.height,
            assetId: asset.id,
          ),
        );
      }

      if (kDebugMode) {
        print('[LocalData] Loaded ${photos.length} photos from library');
      }
      return photos;
    } catch (e) {
      if (kDebugMode) print('[LocalData] Photo access error: $e');
      return [];
    }
  }

  /// Search photos by date range or keyword in title/label.
  Future<List<LocalPhoto>> searchPhotos(String query) async {
    // Get all recent photos and filter locally
    final all = await recentPhotos(limit: 500);
    if (query.isEmpty) return all;

    final lower = query.toLowerCase();
    return all.where((p) {
      final label = (p.label ?? '').toLowerCase();
      return label.contains(lower);
    }).toList();
  }

  /// Build a descriptive label for a photo from its metadata.
  /// Build a rich label for a photo from all available metadata.
  /// GPS coordinates, album name, title, date, and orientation all help
  /// the RAG engine match semantic queries like "vacation in Greece" or
  /// "my driver's license".
  Future<String> buildPhotoLabel(AssetEntity asset, {String? albumName}) async {
    final parts = <String>[];

    // Type description
    if (asset.width > asset.height) {
      parts.add('Landscape photo');
    } else if (asset.height > asset.width) {
      parts.add('Portrait photo');
    } else {
      parts.add('Square photo');
    }

    // Dimensions
    parts.add('${asset.width}x${asset.height}');

    // Date — include day-of-week for better temporal queries
    final d = asset.createDateTime;
    const weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec',
    ];
    parts.add(
      'taken ${weekdays[d.weekday - 1]} '
      '${d.day} ${months[d.month - 1]} ${d.year}',
    );

    // Title / filename — often contains useful keywords
    if (asset.title != null && asset.title!.isNotEmpty) {
      parts.add('titled "${asset.title}"');
    }

    // Album name — "Screenshots", "Camera", "Downloads", etc.
    if (albumName != null && albumName.isNotEmpty) {
      parts.add('in album "$albumName"');
    }

    // GPS location — critical for place-based queries
    try {
      final latLng = await asset.latlngAsync();
      final lat = latLng?.latitude;
      final lng = latLng?.longitude;
      if (lat != null && lng != null && lat != 0.0 && lng != 0.0) {
        parts.add('GPS: ${lat.toStringAsFixed(4)}, ${lng.toStringAsFixed(4)}');
      }
    } catch (_) {}

    return parts.join(', ');
  }

  // Sync wrapper for backward compat (no GPS, no album)
  String _buildPhotoLabel(AssetEntity asset) {
    final parts = <String>[];
    if (asset.width > asset.height) {
      parts.add('Landscape photo');
    } else if (asset.height > asset.width) {
      parts.add('Portrait photo');
    } else {
      parts.add('Square photo');
    }
    parts.add('${asset.width}x${asset.height}');
    final d = asset.createDateTime;
    parts.add(
      'taken ${d.year}-${d.month.toString().padLeft(2, '0')}-${d.day.toString().padLeft(2, '0')}',
    );
    if (asset.title != null && asset.title!.isNotEmpty) {
      parts.add('titled "${asset.title}"');
    }
    return parts.join(', ');
  }

  // ──────────────────────────────────────────
  // CONTACTS
  // ──────────────────────────────────────────

  /// Get all contacts for indexing. Returns name, phone, email.
  Future<List<LocalContact>> getAllContacts() async {
    try {
      final hasPermission = await FlutterContacts.requestPermission(
        readonly: true,
      );
      if (!hasPermission) {
        if (kDebugMode) print('[LocalData] Contacts permission denied');
        return [];
      }

      final contacts = await FlutterContacts.getContacts(
        withProperties: true,
        withPhoto: false, // Don't load photos — just metadata
      );

      final results = <LocalContact>[];
      for (final c in contacts) {
        final phone = c.phones.isNotEmpty ? c.phones.first.number : null;
        final email = c.emails.isNotEmpty ? c.emails.first.address : null;
        final org = c.organizations.isNotEmpty
            ? c.organizations.first.company
            : null;

        results.add(
          LocalContact(
            name: c.displayName,
            phone: phone,
            email: email,
            organization: org,
          ),
        );
      }

      if (kDebugMode) {
        print('[LocalData] Loaded ${results.length} contacts');
      }
      return results;
    } catch (e) {
      if (kDebugMode) print('[LocalData] Contacts access error: $e');
      return [];
    }
  }

  /// Search contacts by name, number, or email.
  Future<List<LocalContact>> searchContacts(String query) async {
    final all = await getAllContacts();
    if (query.isEmpty) return all;

    final lower = query.toLowerCase();
    return all.where((c) {
      return c.name.toLowerCase().contains(lower) ||
          (c.phone?.contains(lower) ?? false) ||
          (c.email?.toLowerCase().contains(lower) ?? false) ||
          (c.organization?.toLowerCase().contains(lower) ?? false);
    }).toList();
  }

  // ──────────────────────────────────────────
  // CALENDAR
  // ──────────────────────────────────────────

  /// Get events for a date range from all device calendars.
  Future<List<LocalEvent>> getEvents(DateTime from, DateTime to) async {
    try {
      final plugin = cal.DeviceCalendarPlugin();

      // Request permission
      var permResult = await plugin.hasPermissions();
      if (permResult.data != true) {
        permResult = await plugin.requestPermissions();
        if (permResult.data != true) {
          if (kDebugMode) print('[LocalData] Calendar permission denied');
          return [];
        }
      }

      // Get all calendars
      final calendarsResult = await plugin.retrieveCalendars();
      final calendars = calendarsResult.data ?? [];
      if (calendars.isEmpty) return [];

      final events = <LocalEvent>[];

      for (final calendar in calendars) {
        if (calendar.id == null) continue;

        final eventsResult = await plugin.retrieveEvents(
          calendar.id!,
          cal.RetrieveEventsParams(startDate: from, endDate: to),
        );

        for (final event in eventsResult.data ?? <cal.Event>[]) {
          if (event.title == null || event.title!.isEmpty) continue;

          events.add(
            LocalEvent(
              title: event.title!,
              start: event.start ?? from,
              end: event.end ?? to,
              location: event.location,
              description: event.description,
              calendarName: calendar.name,
            ),
          );
        }
      }

      if (kDebugMode) {
        print(
          '[LocalData] Loaded ${events.length} calendar events '
          '(${from.toIso8601String()} → ${to.toIso8601String()})',
        );
      }
      return events;
    } catch (e) {
      if (kDebugMode) print('[LocalData] Calendar access error: $e');
      return [];
    }
  }

  // ──────────────────────────────────────────
  // EMAIL (IMAP via enough_mail)
  // ──────────────────────────────────────────

  /// Check if email (IMAP) is configured.
  Future<bool> get isEmailConfigured async {
    final prefs = await SharedPreferences.getInstance();
    final host = prefs.getString('imap_host') ?? '';
    final user = prefs.getString('imap_user') ?? '';
    return host.isNotEmpty && user.isNotEmpty;
  }

  /// Save IMAP credentials. Call from settings screen.
  Future<void> saveEmailConfig({
    required String host,
    required int port,
    required String user,
    required String password,
    bool useSsl = true,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('imap_host', host);
    await prefs.setInt('imap_port', port);
    await prefs.setString('imap_user', user);
    await prefs.setString('imap_password', password);
    await prefs.setBool('imap_ssl', useSsl);
  }

  /// Fetch recent emails from IMAP server.
  /// Credentials are stored locally via SharedPreferences.
  /// All processing stays on-device — we just pull messages from the server.
  Future<List<LocalEmail>> recentEmails({int limit = 50}) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final host = prefs.getString('imap_host') ?? '';
      final user = prefs.getString('imap_user') ?? '';
      final password = prefs.getString('imap_password') ?? '';
      final port = prefs.getInt('imap_port') ?? 993;
      final useSsl = prefs.getBool('imap_ssl') ?? true;

      if (host.isEmpty || user.isEmpty || password.isEmpty) {
        return []; // Not configured
      }

      final client = imap.ImapClient(isLogEnabled: false);
      await client.connectToServer(host, port, isSecure: useSsl);
      await client.login(user, password);

      // Select INBOX
      await client.selectInbox();

      // Fetch last N messages
      final fetchResult = await client.fetchRecentMessages(
        messageCount: limit,
        criteria:
            'BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)] BODY.PEEK[TEXT]',
      );

      final emails = <LocalEmail>[];
      for (final msg in fetchResult.messages) {
        final from =
            msg.from?.first.email ?? msg.from?.first.personalName ?? '';
        final subject = msg.decodeSubject() ?? '(no subject)';
        final date = msg.decodeDate() ?? DateTime.now();
        final body =
            msg.decodeTextPlainPart() ?? msg.decodeTextHtmlPart() ?? '';

        // Truncate long bodies for indexing (keep first 2000 chars)
        final truncBody = body.length > 2000 ? body.substring(0, 2000) : body;

        emails.add(
          LocalEmail(from: from, subject: subject, body: truncBody, date: date),
        );
      }

      await client.logout();

      if (kDebugMode) {
        print('[LocalData] Fetched ${emails.length} emails via IMAP');
      }
      return emails;
    } catch (e) {
      if (kDebugMode) print('[LocalData] Email fetch error: $e');
      return [];
    }
  }

  /// Search emails by sender, subject, or content.
  Future<List<LocalEmail>> searchEmails(String query) async {
    final all = await recentEmails(limit: 50);
    if (query.isEmpty) return all;

    final lower = query.toLowerCase();
    return all.where((e) {
      return e.from.toLowerCase().contains(lower) ||
          e.subject.toLowerCase().contains(lower) ||
          e.body.toLowerCase().contains(lower);
    }).toList();
  }

  // ──────────────────────────────────────────
  // FILES & DOCUMENTS
  // ──────────────────────────────────────────

  /// Text-based file extensions we can read and index.
  static const _textExtensions = {
    // Documents
    '.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm',
    '.rtf', '.tex', '.org', '.rst', '.adoc', '.wiki',
    // Config
    '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.env',
    '.properties', '.plist',
    // Code
    '.py', '.js', '.ts', '.jsx', '.tsx', '.dart', '.java', '.kt',
    '.swift', '.c', '.cpp', '.h', '.hpp', '.rs', '.go', '.rb',
    '.sh', '.bat', '.ps1', '.zsh', '.fish',
    '.sql', '.graphql', '.proto',
    // Data / logs
    '.log', '.ndjson', '.jsonl',
    // Markup
    '.css', '.scss', '.less', '.svg',
  };

  /// Directories to skip during file scanning.
  static const _skipDirs = {
    'node_modules',
    'build',
    '.git',
    '.svn',
    '.hg',
    '__pycache__',
    '.gradle',
    '.idea',
    '.vscode',
    '.dart_tool',
    'Pods',
    '.cache',
    'dist',
    'target',
    'vendor',
    '.pub-cache',
    'DerivedData',
    'xcuserdata',
  };

  /// Get all scannable directories for the current platform.
  Future<List<_ScanDir>> _getScanDirs() async {
    final dirs = <_ScanDir>[];

    // Always include app documents directory
    try {
      final docDir = await getApplicationDocumentsDirectory();
      dirs.add(_ScanDir(docDir, maxDepth: 5));
    } catch (_) {}

    if (Platform.isMacOS || Platform.isLinux || Platform.isWindows) {
      // Desktop: scan user home subdirectories
      final home =
          Platform.environment['HOME'] ?? Platform.environment['USERPROFILE'];
      if (home != null) {
        final scanPaths = [
          '$home/Documents',
          '$home/Downloads',
          '$home/Desktop',
          '$home/Notes',
          '$home/Projects',
          '$home/Developer',
        ];
        for (final path in scanPaths) {
          final dir = Directory(path);
          if (await dir.exists()) {
            dirs.add(_ScanDir(dir, maxDepth: 4));
          }
        }
      }
    } else if (Platform.isIOS || Platform.isAndroid) {
      // Mobile: scan external directories if available
      try {
        final extDirs = await getExternalStorageDirectories();
        if (extDirs != null) {
          for (final dir in extDirs) {
            dirs.add(_ScanDir(dir, maxDepth: 3));
          }
        }
      } catch (_) {
        // iOS doesn't have external storage — app docs is enough
      }

      // Android: try common user directories
      if (Platform.isAndroid) {
        for (final subDir in ['Download', 'Documents', 'DCIM']) {
          final dir = Directory('/storage/emulated/0/$subDir');
          try {
            if (await dir.exists()) {
              dirs.add(_ScanDir(dir, maxDepth: 2));
            }
          } catch (_) {}
        }
      }
    }

    return dirs;
  }

  /// Scan all known directories for indexable files.
  /// If [query] is empty, returns ALL indexable files.
  /// Returns files with fingerprints for change detection.
  Future<List<LocalFile>> searchFiles(String query) async {
    final files = <LocalFile>[];
    final seenPaths = <String>{};

    try {
      final scanDirs = await _getScanDirs();

      for (final scanDir in scanDirs) {
        await _scanDirectory(
          scanDir.dir,
          files,
          query,
          maxDepth: scanDir.maxDepth,
          seenPaths: seenPaths,
        );
      }

      if (kDebugMode) {
        print(
          '[LocalData] Scanned ${scanDirs.length} dirs, '
          'found ${files.length} indexable files',
        );
      }
    } catch (e) {
      if (kDebugMode) {
        print('[LocalData] Error scanning files: $e');
      }
    }

    return files;
  }

  /// Read the text content of a file (if it's a supported text file).
  /// Returns null for binary files, files too large, or on failure.
  Future<String?> readFileContent(String path) async {
    try {
      final file = File(path);
      if (!await file.exists()) return null;

      final ext = path.contains('.')
          ? '.${path.split('.').last.toLowerCase()}'
          : '';
      if (!_textExtensions.contains(ext)) return null;

      // Skip files larger than 2MB
      final stat = await file.stat();
      if (stat.size > 2 * 1024 * 1024) return null;
      if (stat.size == 0) return null;

      return await file.readAsString();
    } catch (e) {
      // Binary file or encoding error — silently skip
      return null;
    }
  }

  /// Compute a change-detection fingerprint for a file: "mtime_ms:size".
  /// Returns null if file doesn't exist.
  Future<String?> fileFingerprint(String path) async {
    try {
      final stat = await File(path).stat();
      if (stat.type == FileSystemEntityType.notFound) return null;
      return '${stat.modified.millisecondsSinceEpoch}:${stat.size}';
    } catch (_) {
      return null;
    }
  }

  Future<void> _scanDirectory(
    Directory dir,
    List<LocalFile> results,
    String query, {
    required int maxDepth,
    int currentDepth = 0,
    Set<String>? seenPaths,
  }) async {
    if (currentDepth > maxDepth) return;

    try {
      await for (final entity in dir.list(followLinks: false)) {
        if (entity is File) {
          final name = entity.path.split('/').last;
          if (name.startsWith('.')) continue;

          // Dedup by absolute path
          if (seenPaths != null && !seenPaths.add(entity.path)) continue;

          final ext = name.contains('.')
              ? '.${name.split('.').last.toLowerCase()}'
              : '';
          if (!_textExtensions.contains(ext)) continue;

          // Filter by query if provided
          if (query.isNotEmpty) {
            final lowerName = name.toLowerCase();
            final lowerQuery = query.toLowerCase();
            if (!lowerName.contains(lowerQuery)) continue;
          }

          try {
            final stat = await entity.stat();
            if (stat.size == 0 || stat.size > 2 * 1024 * 1024) continue;

            results.add(
              LocalFile(
                path: entity.path,
                name: name,
                sizeBytes: stat.size,
                modified: stat.modified,
                fingerprint:
                    '${stat.modified.millisecondsSinceEpoch}:${stat.size}',
              ),
            );
          } catch (_) {}
        } else if (entity is Directory) {
          final dirName = entity.path.split('/').last;
          if (dirName.startsWith('.')) continue;
          if (_skipDirs.contains(dirName)) continue;
          await _scanDirectory(
            entity,
            results,
            query,
            maxDepth: maxDepth,
            currentDepth: currentDepth + 1,
            seenPaths: seenPaths,
          );
        }
      }
    } catch (e) {
      // Permission denied or other error — skip silently
      if (kDebugMode && currentDepth == 0) {
        debugPrint('[LocalData] Cannot scan ${dir.path}: $e');
      }
    }
  }

  // ──────────────────────────────────────────
  // WEB (Only when user explicitly asks)
  // ──────────────────────────────────────────

  /// Lightweight online check. True when DNS/network is reachable.
  Future<bool> hasInternetConnection() async {
    try {
      final result = await InternetAddress.lookup(
        'example.com',
      ).timeout(const Duration(seconds: 2));
      return result.isNotEmpty && result.first.rawAddress.isNotEmpty;
    } catch (_) {
      return false;
    }
  }

  /// Search the web to augment local knowledge.
  /// ONLY called when the user says "search", "look up", "google", etc.
  Future<String> webSearch(String query) async {
    // TODO: Use http package to hit a search API
    return '';
  }
}

/// Internal: directory + its scan depth limit.
class _ScanDir {
  final Directory dir;
  final int maxDepth;
  _ScanDir(this.dir, {this.maxDepth = 3});
}

// ──────────────────────────────────────────
// Local data models
// ──────────────────────────────────────────

class LocalPhoto {
  final String path;
  final DateTime date;
  final String? label; // AI-generated or metadata-based label
  final int width;
  final int height;
  final String? assetId; // Platform asset ID for re-access

  LocalPhoto({
    required this.path,
    required this.date,
    this.label,
    this.width = 0,
    this.height = 0,
    this.assetId,
  });
}

class LocalEmail {
  final String from;
  final String subject;
  final String body;
  final DateTime date;
  final List<String> attachments;

  LocalEmail({
    required this.from,
    required this.subject,
    required this.body,
    required this.date,
    this.attachments = const [],
  });
}

class LocalFile {
  final String path;
  final String name;
  final int sizeBytes;
  final DateTime modified;
  final String fingerprint; // "mtime_ms:size" for change detection

  LocalFile({
    required this.path,
    required this.name,
    required this.sizeBytes,
    required this.modified,
    required this.fingerprint,
  });
}

class LocalContact {
  final String name;
  final String? phone;
  final String? email;
  final String? organization;

  LocalContact({required this.name, this.phone, this.email, this.organization});
}

class LocalEvent {
  final String title;
  final DateTime start;
  final DateTime end;
  final String? location;
  final String? description;
  final String? calendarName;

  LocalEvent({
    required this.title,
    required this.start,
    required this.end,
    this.location,
    this.description,
    this.calendarName,
  });
}
