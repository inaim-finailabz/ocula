import 'dart:io';
import 'package:device_calendar/device_calendar.dart' as cal;
import 'package:flutter/foundation.dart';
import 'package:timezone/timezone.dart' as tz;
import 'package:url_launcher/url_launcher.dart';
import 'local_data.dart';
import 'notification_service.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

enum ActionType { createEvent, setReminder, callContact, textContact, draftEmail }

/// Holds everything needed to execute a single proactive action.
/// Fields are mutable so the confirmation card can let the user edit them.
class ActionRequest {
  final ActionType type;
  final String rawQuery;

  String? title;
  DateTime? when;
  Duration? duration;
  String? contactName;
  String? contactPhone;
  String? contactEmail;
  String? body;

  ActionRequest({
    required this.type,
    required this.rawQuery,
    this.title,
    this.when,
    this.duration,
    this.contactName,
    this.contactPhone,
    this.contactEmail,
    this.body,
  });

  ActionRequest copyWith({
    String? title,
    DateTime? when,
    Duration? duration,
    String? contactName,
    String? contactPhone,
    String? contactEmail,
    String? body,
  }) => ActionRequest(
    type: type,
    rawQuery: rawQuery,
    title: title ?? this.title,
    when: when ?? this.when,
    duration: duration ?? this.duration,
    contactName: contactName ?? this.contactName,
    contactPhone: contactPhone ?? this.contactPhone,
    contactEmail: contactEmail ?? this.contactEmail,
    body: body ?? this.body,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ActionService
// ─────────────────────────────────────────────────────────────────────────────

class ActionService {
  // ── Detection ──────────────────────────────────────────────────────────────

  /// Returns a parsed `ActionRequest` if the query is an action command,
  /// or null if it should go through the normal RAG pipeline.
  static ActionRequest? detect(String query) {
    final lower = query.toLowerCase().trim();

    // Call — check before text/email since "call" is unambiguous
    if (_has(lower, ['call ', 'ring ', 'dial '])) {
      return ActionRequest(
        type: ActionType.callContact,
        rawQuery: query,
        contactName: _contactName(lower, ['call', 'ring', 'dial']),
      );
    }

    // Text / SMS
    if (_has(lower, ['text ', 'sms ', 'whatsapp ']) ||
        lower.startsWith('message ') ||
        lower.startsWith('send a message to ') ||
        lower.startsWith('send message to ')) {
      final req = ActionRequest(type: ActionType.textContact, rawQuery: query);
      req.contactName = _contactName(lower, ['text', 'sms', 'whatsapp', 'send a message to', 'send message to', 'message']);
      req.body = _bodyAfterContact(lower, req.contactName);
      return req;
    }

    // Email draft
    if (_has(lower, ['email ', 'send an email to ', 'send email to ',
        'draft an email', 'write an email to ', 'compose an email to '])) {
      final req = ActionRequest(type: ActionType.draftEmail, rawQuery: query);
      req.contactName = _contactName(lower, [
        'email', 'send an email to', 'send email to',
        'write an email to', 'compose an email to',
      ]);
      req.title = _subjectLine(lower);
      req.body = _bodyAfterContact(lower, req.contactName);
      return req;
    }

    // Reminder
    if (_has(lower, [
      'remind me', 'set a reminder', 'set reminder', 'set an alarm',
      "don't let me forget", 'remember to ', 'alert me at ', 'notify me when ',
    ])) {
      return ActionRequest(
        type: ActionType.setReminder,
        rawQuery: query,
        title: _reminderTitle(lower),
        when: _parseDateTime(lower),
      );
    }

    // Calendar event
    if (_has(lower, [
      'schedule ', 'set up a meeting', 'add to calendar', 'add to my calendar',
      'create an event', 'create event', 'book a ', 'book an ',
      'plan a ', 'plan an ', 'put on my calendar', 'calendar event',
      'add a meeting', 'add an appointment',
    ])) {
      return ActionRequest(
        type: ActionType.createEvent,
        rawQuery: query,
        title: _eventTitle(lower),
        when: _parseDateTime(lower),
        duration: _parseDuration(lower),
      );
    }

    return null;
  }

  static bool _has(String lower, List<String> tokens) =>
      tokens.any((t) => lower.contains(t));

  // ── Contact resolution ─────────────────────────────────────────────────────

  /// Fills in `contactPhone` and `contactEmail` by searching local contacts.
  static Future<void> resolveContact(ActionRequest req) async {
    if (req.contactName == null) return;
    try {
      final hits = await LocalData().searchContacts(req.contactName!);
      if (hits.isNotEmpty) {
        req.contactPhone ??= hits.first.phone;
        req.contactEmail ??= hits.first.email;
      }
    } catch (e) {
      debugPrint('[ActionService] Contact lookup: $e');
    }
  }

  // ── Execution ──────────────────────────────────────────────────────────────

  static Future<String> execute(ActionRequest req) async {
    switch (req.type) {
      case ActionType.createEvent:   return _createEvent(req);
      case ActionType.setReminder:   return _scheduleReminder(req);
      case ActionType.callContact:   return _call(req);
      case ActionType.textContact:   return _text(req);
      case ActionType.draftEmail:    return _email(req);
    }
  }

  // Create calendar event via device_calendar
  static Future<String> _createEvent(ActionRequest req) async {
    if (!Platform.isAndroid && !Platform.isIOS) {
      return 'Event noted — open Calendar to add it on this platform.';
    }
    final plugin = cal.DeviceCalendarPlugin();
    var perm = await plugin.hasPermissions();
    if (perm.data != true) perm = await plugin.requestPermissions();
    if (perm.data != true) return 'Calendar access denied. Enable in Settings > Privacy.';

    final cals = (await plugin.retrieveCalendars()).data ?? [];
    final writable = cals.where((c) => c.isReadOnly != true).toList();
    final target = writable.isNotEmpty ? writable.first : (cals.isNotEmpty ? cals.first : null);
    if (target?.id == null) return 'No writable calendar found on this device.';

    final start = req.when ?? DateTime.now().add(const Duration(hours: 1));
    final end = start.add(req.duration ?? const Duration(hours: 1));
    final loc = tz.local;

    final event = cal.Event(
      target!.id!,
      title: req.title ?? 'New Event',
      start: tz.TZDateTime.from(start, loc),
      end: tz.TZDateTime.from(end, loc),
    );
    final result = await plugin.createOrUpdateEvent(event);
    if (result?.isSuccess == true) {
      return 'Added to your calendar: "${req.title ?? 'New Event'}" on ${_fmtDateTime(start)}.';
    }
    final err = result == null ? 'unknown error' : (result.errors.firstOrNull?.errorMessage ?? 'unknown error');
    return 'Could not create event: $err';
  }

  // Schedule a local notification reminder
  static Future<String> _scheduleReminder(ActionRequest req) async {
    final when = req.when;
    if (when == null) {
      return "I couldn't work out when to remind you. Try: \"remind me to X at 3pm tomorrow\".";
    }
    if (when.isBefore(DateTime.now())) {
      return "That time has already passed. Did you mean tomorrow?";
    }
    final title = req.title ?? 'Reminder';
    final ok = await NotificationService().scheduleReminder(title: title, when: when);
    if (ok) return 'Reminder set: "$title" — ${_fmtDateTime(when)}.';
    return 'Could not schedule the reminder — check notification permissions in Settings.';
  }

  // Open phone dialler
  static Future<String> _call(ActionRequest req) async {
    final phone = req.contactPhone;
    if (phone == null || phone.isEmpty) {
      return "I couldn't find a phone number for ${req.contactName ?? 'that contact'}. Open Contacts to call manually.";
    }
    final digits = phone.replaceAll(RegExp(r'[^0-9+]'), '');
    final uri = Uri(scheme: 'tel', path: digits);
    if (await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      return 'Opening Phone to call ${req.contactName ?? phone}…';
    }
    return 'Could not open the Phone app.';
  }

  // Open Messages with pre-filled body
  static Future<String> _text(ActionRequest req) async {
    final phone = req.contactPhone;
    if (phone == null || phone.isEmpty) {
      return "I couldn't find a phone number for ${req.contactName ?? 'that contact'}. Open Messages manually.";
    }
    final digits = phone.replaceAll(RegExp(r'[^0-9+]'), '');
    final body = req.body ?? '';
    final uri = Uri(
      scheme: 'sms',
      path: digits,
      queryParameters: body.isNotEmpty ? {'body': body} : null,
    );
    if (await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      return 'Opening Messages for ${req.contactName ?? phone}…';
    }
    return 'Could not open the Messages app.';
  }

  // Open Mail with pre-filled fields
  static Future<String> _email(ActionRequest req) async {
    final emailAddr = req.contactEmail;
    if (emailAddr == null || emailAddr.isEmpty) {
      return "I couldn't find an email for ${req.contactName ?? 'that contact'}. Open Mail manually.";
    }
    final params = <String, String>{};
    if (req.title != null) params['subject'] = req.title!;
    if (req.body != null) params['body'] = req.body!;
    final uri = Uri(scheme: 'mailto', path: emailAddr, queryParameters: params.isNotEmpty ? params : null);
    if (await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      return 'Opened Mail — draft ready for ${req.contactName ?? emailAddr}.';
    }
    return 'Could not open the Mail app.';
  }

  // ── Name / text extraction helpers ─────────────────────────────────────────

  static String? _contactName(String lower, List<String> keywords) {
    final sorted = List.of(keywords)..sort((a, b) => b.length.compareTo(a.length));
    for (final kw in sorted) {
      final idx = lower.indexOf(kw);
      if (idx < 0) continue;
      var rest = lower.substring(idx + kw.length).trim();
      // Strip article
      if (rest.startsWith('a ') || rest.startsWith('an ')) {
        rest = rest.replaceFirst(RegExp(r'^a(?:n)? '), '');
      }
      // Stop before body-of-message markers
      for (final stop in [
        ' that ', ' to say ', ' about ', ' saying ', ' telling ',
        ' regarding ', ' to tell ', ' and say ', ': ',
      ]) {
        final si = rest.indexOf(stop);
        if (si > 0) rest = rest.substring(0, si).trim();
      }
      // Strip trailing time fragments ("at 3pm", "tomorrow", etc.)
      rest = rest.replaceAll(RegExp(r'\s+(at|in|on|tomorrow|tonight|today|next|this)\s.*$'), '').trim();
      final words = rest.split(RegExp(r'\s+')).where((w) => w.isNotEmpty).toList();
      if (words.isNotEmpty && words.first.length > 1) {
        final name = words.take(3).map((w) => w[0].toUpperCase() + w.substring(1)).join(' ');
        return name;
      }
    }
    return null;
  }

  static String? _bodyAfterContact(String lower, String? contactName) {
    if (contactName == null) return null;
    final ni = lower.indexOf(contactName.toLowerCase());
    if (ni < 0) return null;
    var rest = lower.substring(ni + contactName.length).trim();
    for (final prefix in ['that ', 'to say ', 'saying ', ': ']) {
      if (rest.startsWith(prefix)) { rest = rest.substring(prefix.length).trim(); break; }
    }
    if (rest.length < 3) return null;
    return rest[0].toUpperCase() + rest.substring(1);
  }

  static String? _subjectLine(String lower) {
    final m = RegExp(r'(?:about|re:|subject:|regarding) (.+?)(?:\s+(?:to|from|for)|$)').firstMatch(lower);
    if (m?.group(1) != null) {
      final s = m!.group(1)!.trim();
      return s.isNotEmpty ? s[0].toUpperCase() + s.substring(1) : null;
    }
    return null;
  }

  static String? _reminderTitle(String lower) {
    final patterns = [
      RegExp(r'remind me to (.+?)(?:\s+(?:at|in|on|tomorrow|tonight|today|next|this|\d{1,2}[ap]m))', caseSensitive: false),
      RegExp(r'remind me to (.+)', caseSensitive: false),
      RegExp(r'remind me about (.+?)(?:\s+(?:at|in|on|tomorrow|tonight|today|next|this|\d{1,2}[ap]m))', caseSensitive: false),
      RegExp(r'remind me about (.+)', caseSensitive: false),
      RegExp(r'set (?:a |an )?reminder (?:to|for|about) (.+?)(?:\s+(?:at|in|on|tomorrow|tonight|today|next|this|\d{1,2}[ap]m))', caseSensitive: false),
      RegExp(r'set (?:a |an )?reminder (?:to|for|about) (.+)', caseSensitive: false),
      RegExp(r'remember to (.+?)(?:\s+(?:at|in|on|tomorrow|tonight|today|next|this|\d{1,2}[ap]m))', caseSensitive: false),
      RegExp(r'remember to (.+)', caseSensitive: false),
    ];
    for (final rx in patterns) {
      final m = rx.firstMatch(lower);
      if (m?.group(1) != null) {
        var t = m!.group(1)!.trim();
        // Strip trailing time fragment
        t = t.replaceAll(RegExp(r'\s+(?:at|in|on|tomorrow|tonight|today|next|this)\s*.*$'), '').trim();
        if (t.isNotEmpty) return t[0].toUpperCase() + t.substring(1);
      }
    }
    return null;
  }

  static String? _eventTitle(String lower) {
    final patterns = [
      RegExp(r'schedule (?:a |an )?(.+?)(?:\s+(?:at|in|on|for|with|tomorrow|tonight|today|next|this|\d{1,2}[ap]m))', caseSensitive: false),
      RegExp(r'schedule (?:a |an )?(.+)', caseSensitive: false),
      RegExp(r'add (?:a |an )?(.+?) to (?:my )?calendar', caseSensitive: false),
      RegExp(r'create (?:a |an )?(?:event|meeting|appointment) (?:called |named |for |about )?(.+?)(?:\s+(?:at|in|on|tomorrow|tonight|today|next|this|\d{1,2}[ap]m))', caseSensitive: false),
      RegExp(r'create (?:a |an )?(?:event|meeting|appointment) (?:called |named |for |about )?(.+)', caseSensitive: false),
      RegExp(r'book (?:a |an )?(.+?)(?:\s+(?:at|in|on|for|with|tomorrow|tonight|today|next|this|\d{1,2}[ap]m))', caseSensitive: false),
      RegExp(r'plan (?:a |an )?(.+?)(?:\s+(?:at|in|on|for|with|tomorrow|tonight|today|next|this|\d{1,2}[ap]m))', caseSensitive: false),
    ];
    for (final rx in patterns) {
      final m = rx.firstMatch(lower);
      if (m?.group(1) != null) {
        var t = m!.group(1)!.trim();
        if (t.isNotEmpty) return t[0].toUpperCase() + t.substring(1);
      }
    }
    return null;
  }

  // ── DateTime parsing ───────────────────────────────────────────────────────

  static DateTime? _parseDateTime(String lower) {
    final now = DateTime.now();

    // "in X minutes/hours/days/weeks"
    final relM = RegExp(r'in (\d+)\s*(minute|min|hour|hr|day|week)s?', caseSensitive: false).firstMatch(lower);
    if (relM != null) {
      final n = int.parse(relM.group(1)!);
      final unit = relM.group(2)!.toLowerCase();
      if (unit.startsWith('min')) return now.add(Duration(minutes: n));
      if (unit.startsWith('hour') || unit == 'hr') return now.add(Duration(hours: n));
      if (unit.startsWith('day')) return now.add(Duration(days: n));
      if (unit.startsWith('week')) return now.add(Duration(days: n * 7));
    }

    // Clock time: "at 3pm", "at 3:30 pm", "at 15:00"
    _TimeOfDay? timeOfDay;
    final clockM = RegExp(r'at (\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?:\s|$)', caseSensitive: false).firstMatch(lower);
    if (clockM != null) {
      var h = int.parse(clockM.group(1)!);
      final min = int.tryParse(clockM.group(2) ?? '0') ?? 0;
      final ampm = clockM.group(3)?.toLowerCase();
      if (ampm == 'pm' && h < 12) h += 12;
      if (ampm == 'am' && h == 12) h = 0;
      // If no am/pm and hour is ambiguous (1-7), assume PM
      if (ampm == null && h >= 1 && h <= 7) h += 12;
      timeOfDay = _TimeOfDay(h, min);
    }

    // Base date
    DateTime? base;
    if (lower.contains('tonight') || lower.contains('today')) {
      base = now;
    } else if (lower.contains('day after tomorrow')) {
      base = now.add(const Duration(days: 2));
    } else if (lower.contains('tomorrow')) {
      base = now.add(const Duration(days: 1));
    } else {
      // Named weekday — always pick the NEXT occurrence
      const wd = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'];
      for (int i = 0; i < wd.length; i++) {
        if (lower.contains(wd[i])) {
          var diff = (i + 1) - now.weekday;
          if (diff <= 0) diff += 7;
          base = now.add(Duration(days: diff));
          break;
        }
      }
    }

    if (base == null && timeOfDay == null) return null;

    final d = base ?? now;
    if (timeOfDay != null) {
      var dt = DateTime(d.year, d.month, d.day, timeOfDay.hour, timeOfDay.minute);
      // If the computed time is already past and no explicit date given, push to tomorrow
      if (dt.isBefore(now) && base == null) dt = dt.add(const Duration(days: 1));
      return dt;
    }
    // Date only — default to 9 AM
    return DateTime(d.year, d.month, d.day, 9, 0);
  }

  static Duration? _parseDuration(String lower) {
    final m = RegExp(r'for (\d+(?:\.\d+)?)\s*(minute|min|hour|hr)s?', caseSensitive: false).firstMatch(lower);
    if (m != null) {
      final n = double.parse(m.group(1)!);
      final unit = m.group(2)!.toLowerCase();
      if (unit.startsWith('min')) return Duration(minutes: n.round());
      return Duration(minutes: (n * 60).round());
    }
    return null;
  }

  // ── Display helpers ────────────────────────────────────────────────────────

  static String _fmtDateTime(DateTime dt) {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final dtDay = DateTime(dt.year, dt.month, dt.day);
    final diff = dtDay.difference(today).inDays;

    String dateStr;
    if (diff == 0) dateStr = 'today';
    else if (diff == 1) dateStr = 'tomorrow';
    else {
      const names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
      dateStr = names[dt.weekday - 1];
    }

    final h = dt.hour, m = dt.minute;
    final ampm = h < 12 ? 'AM' : 'PM';
    final h12 = h == 0 ? 12 : (h > 12 ? h - 12 : h);
    final mStr = m == 0 ? '' : ':${m.toString().padLeft(2, '0')}';
    return '$dateStr at $h12$mStr $ampm';
  }

  static String _fmtDuration(Duration d) {
    if (d.inMinutes < 60) return '${d.inMinutes} min';
    final h = d.inHours, m = d.inMinutes.remainder(60);
    return m == 0 ? '${h}h' : '${h}h ${m}m';
  }

  /// Human-readable one-liner for the confirmation card header.
  static String summary(ActionRequest req) {
    final when = req.when != null ? _fmtDateTime(req.when!) : null;
    switch (req.type) {
      case ActionType.createEvent:
        final dur = req.duration != null ? ' · ${_fmtDuration(req.duration!)}' : '';
        return '"${req.title ?? 'Untitled'}"${when != null ? ' — $when$dur' : ''}';
      case ActionType.setReminder:
        return '"${req.title ?? 'Reminder'}"${when != null ? ' — $when' : ''}';
      case ActionType.callContact:
        final ph = req.contactPhone != null ? ' (${req.contactPhone})' : '';
        return '${req.contactName ?? 'Unknown'}$ph';
      case ActionType.textContact:
        final preview = req.body != null && req.body!.isNotEmpty
            ? ' — "${req.body!.length > 40 ? '${req.body!.substring(0, 40)}…' : req.body!}"'
            : '';
        return '${req.contactName ?? 'Unknown'}$preview';
      case ActionType.draftEmail:
        final subj = req.title != null ? ' — "${req.title}"' : '';
        return '${req.contactName ?? req.contactEmail ?? 'Unknown'}$subj';
    }
  }
}

class _TimeOfDay {
  final int hour, minute;
  _TimeOfDay(this.hour, this.minute);
}
