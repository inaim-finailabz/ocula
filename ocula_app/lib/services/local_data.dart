// Local data layer — accesses everything on the user's device.
// NOTHING leaves the phone. All processing is on-device.
//
// v2: Broader scanning, file fingerprints for change detection,
// more file types, platform-aware directory discovery.

import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

class LocalData {
  static final LocalData _instance = LocalData._();
  factory LocalData() => _instance;
  LocalData._();

  // ──────────────────────────────────────────
  // PHOTOS & CAMERA ROLL
  // ──────────────────────────────────────────

  /// Search photos by date, location, or content (after indexing).
  Future<List<LocalPhoto>> searchPhotos(String query) async {
    // TODO: Use photo_manager to access camera roll and do label-based search
    return [];
  }

  /// Get recent photos for quick access.
  Future<List<LocalPhoto>> recentPhotos({int limit = 20}) async {
    // TODO: photo_manager integration
    return [];
  }

  // ──────────────────────────────────────────
  // EMAIL (Local IMAP cache)
  // ──────────────────────────────────────────

  /// Search local email cache by sender, subject, or content.
  Future<List<LocalEmail>> searchEmails(String query) async {
    // TODO: enough_mail IMAP integration
    return [];
  }

  /// Get recent emails.
  Future<List<LocalEmail>> recentEmails({int limit = 10}) async {
    // TODO: enough_mail IMAP integration
    return [];
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
    'node_modules', 'build', '.git', '.svn', '.hg', '__pycache__',
    '.gradle', '.idea', '.vscode', '.dart_tool', 'Pods',
    '.cache', 'dist', 'target', 'vendor', '.pub-cache',
    'DerivedData', 'xcuserdata',
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
      final home = Platform.environment['HOME'] ??
          Platform.environment['USERPROFILE'];
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
          scanDir.dir, files, query,
          maxDepth: scanDir.maxDepth,
          seenPaths: seenPaths,
        );
      }

      if (kDebugMode) {
        print('[LocalData] Scanned ${scanDirs.length} dirs, '
            'found ${files.length} indexable files');
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

            results.add(LocalFile(
              path: entity.path,
              name: name,
              sizeBytes: stat.size,
              modified: stat.modified,
              fingerprint: '${stat.modified.millisecondsSinceEpoch}:${stat.size}',
            ));
          } catch (_) {}
        } else if (entity is Directory) {
          final dirName = entity.path.split('/').last;
          if (dirName.startsWith('.')) continue;
          if (_skipDirs.contains(dirName)) continue;
          await _scanDirectory(entity, results, query,
              maxDepth: maxDepth,
              currentDepth: currentDepth + 1,
              seenPaths: seenPaths);
        }
      }
    } catch (e) {
      // Permission denied or other error — skip silently
      if (kDebugMode && currentDepth == 0) {
        print('[LocalData] Cannot scan ${dir.path}: $e');
      }
    }
  }

  // ──────────────────────────────────────────
  // CONTACTS
  // ──────────────────────────────────────────

  /// Search contacts by name, number, or email.
  Future<List<LocalContact>> searchContacts(String query) async {
    // TODO: flutter_contacts package integration
    return [];
  }

  // ──────────────────────────────────────────
  // CALENDAR
  // ──────────────────────────────────────────

  /// Get events for a date range.
  Future<List<LocalEvent>> getEvents(DateTime from, DateTime to) async {
    // TODO: device_calendar package integration
    return [];
  }

  // ──────────────────────────────────────────
  // WEB (Only when user explicitly asks)
  // ──────────────────────────────────────────

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
  final String? label; // AI-generated label from SmolVLM indexing

  LocalPhoto({required this.path, required this.date, this.label});
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

  LocalContact({required this.name, this.phone, this.email});
}

class LocalEvent {
  final String title;
  final DateTime start;
  final DateTime end;
  final String? location;

  LocalEvent({
    required this.title,
    required this.start,
    required this.end,
    this.location,
  });
}
