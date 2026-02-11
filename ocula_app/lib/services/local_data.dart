// Local data layer — accesses everything on the user's device.
// NOTHING leaves the phone. All processing is on-device.
//
// Uses platform Method Channels to access native iOS/Android APIs.
// Each data source requires user permission on first access.

class LocalData {
  static final LocalData _instance = LocalData._();
  factory LocalData() => _instance;
  LocalData._();

  // ──────────────────────────────────────────
  // PHOTOS & CAMERA ROLL
  // ──────────────────────────────────────────

  /// Search photos by date, location, or content (after indexing).
  Future<List<LocalPhoto>> searchPhotos(String query) async {
    // TODO: Use photo_manager package to access camera roll
    // Index photos with SmolVLM on first run (background task)
    // Store labels in local SQLite for fast search
    return [];
  }

  /// Get recent photos for quick access.
  Future<List<LocalPhoto>> recentPhotos({int limit = 20}) async {
    return [];
  }

  // ──────────────────────────────────────────
  // EMAIL (Local IMAP cache)
  // ──────────────────────────────────────────

  /// Search local email cache by sender, subject, or content.
  Future<List<LocalEmail>> searchEmails(String query) async {
    // TODO: Use enough_mail package for IMAP access
    // Cache emails locally in SQLite
    // User grants email access once, then it stays local
    return [];
  }

  /// Get recent emails.
  Future<List<LocalEmail>> recentEmails({int limit = 10}) async {
    return [];
  }

  // ──────────────────────────────────────────
  // FILES & DOCUMENTS
  // ──────────────────────────────────────────

  /// Search files on device (PDFs, docs, downloads).
  Future<List<LocalFile>> searchFiles(String query) async {
    // TODO: Use file_picker + path_provider to scan local storage
    // Index PDF text content with Qwen for semantic search
    return [];
  }

  // ──────────────────────────────────────────
  // CONTACTS
  // ──────────────────────────────────────────

  /// Search contacts by name, number, or email.
  Future<List<LocalContact>> searchContacts(String query) async {
    // TODO: Use flutter_contacts package
    return [];
  }

  // ──────────────────────────────────────────
  // CALENDAR
  // ──────────────────────────────────────────

  /// Get events for a date range.
  Future<List<LocalEvent>> getEvents(DateTime from, DateTime to) async {
    // TODO: Use device_calendar package
    return [];
  }

  // ──────────────────────────────────────────
  // WEB (Only when user explicitly asks)
  // ──────────────────────────────────────────

  /// Search the web to augment local knowledge.
  /// ONLY called when the user says "search", "look up", "google", etc.
  Future<String> webSearch(String query) async {
    // TODO: Use http package to hit a search API
    // This is the ONLY function that touches the internet
    return '';
  }
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

  LocalFile({
    required this.path,
    required this.name,
    required this.sizeBytes,
    required this.modified,
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
