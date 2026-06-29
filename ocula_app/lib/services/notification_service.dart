import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:timezone/data/latest_all.dart' as tz;
import 'package:timezone/timezone.dart' as tz;
import 'package:workmanager/workmanager.dart';

import 'local_data.dart';

/// Background task identifier for WorkManager.
const _kCalendarSyncTask = 'ocula_calendar_sync';
const _kDailyBriefingTask = 'ocula_daily_briefing';

/// Top-level callback for WorkManager (must be top-level or static).
@pragma('vm:entry-point')
void callbackDispatcher() {
  Workmanager().executeTask((taskName, inputData) async {
    final service = NotificationService();
    await service._initPlugin();

    switch (taskName) {
      case _kCalendarSyncTask:
        await service.scheduleCalendarReminders();
        break;
      case _kDailyBriefingTask:
        await service._showDailyBriefing();
        break;
    }
    return true;
  });
}

/// Manages local notifications for calendar reminders and daily briefings.
///
/// Privacy-first: no push notifications, no backend. Everything runs on-device.
class NotificationService {
  static final NotificationService _instance = NotificationService._();
  factory NotificationService() => _instance;
  NotificationService._();

  final FlutterLocalNotificationsPlugin _plugin =
      FlutterLocalNotificationsPlugin();

  bool _initialized = false;

  /// Callback when user taps a notification — receives the query to pre-fill.
  void Function(String query)? onNotificationTap;

  // ── Notification channel IDs ──
  static const _calendarChannelId = 'ocula_calendar';
  static const _briefingChannelId = 'ocula_briefing';

  // ── Prefs keys ──
  static const _kEnabled = 'notif_enabled';
  static const _kReminderMinutes = 'notif_reminder_minutes';
  static const _kBriefingHour = 'notif_briefing_hour';
  static const _kBriefingEnabled = 'notif_briefing_enabled';
  static const _kScheduledIds = 'notif_scheduled_ids';

  /// Initialize the notification plugin, channels, and background tasks.
  Future<void> init() async {
    if (_initialized) return;
    await _initPlugin();
    if (Platform.isAndroid || Platform.isIOS) {
      await _registerBackgroundTasks();
    }
    _initialized = true;
  }

  Future<void> _initPlugin() async {
    tz.initializeTimeZones();

    const androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');
    const darwinSettings = DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );

    await _plugin.initialize(
      const InitializationSettings(
        android: androidSettings,
        iOS: darwinSettings,
        macOS: darwinSettings,
      ),
      onDidReceiveNotificationResponse: _onNotificationTap,
    );
  }

  void _onNotificationTap(NotificationResponse response) {
    final payload = response.payload;
    if (payload != null && payload.isNotEmpty && onNotificationTap != null) {
      onNotificationTap!(payload);
    }
  }

  /// Request notification permission (Android 13+, iOS, macOS).
  Future<bool> requestPermission() async {
    if (Platform.isAndroid) {
      final android = _plugin.resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin>();
      final granted = await android?.requestNotificationsPermission();
      return granted ?? false;
    } else if (Platform.isIOS) {
      final ios = _plugin.resolvePlatformSpecificImplementation<
          IOSFlutterLocalNotificationsPlugin>();
      final granted = await ios?.requestPermissions(
        alert: true,
        badge: true,
        sound: true,
      );
      return granted ?? false;
    } else if (Platform.isMacOS) {
      final macos = _plugin.resolvePlatformSpecificImplementation<
          MacOSFlutterLocalNotificationsPlugin>();
      final granted = await macos?.requestPermissions(
        alert: true,
        badge: true,
        sound: true,
      );
      return granted ?? false;
    }
    return false;
  }

  Future<void> _registerBackgroundTasks() async {
    await Workmanager().initialize(callbackDispatcher, isInDebugMode: kDebugMode);

    // Periodic calendar sync — checks for new events every 6 hours
    await Workmanager().registerPeriodicTask(
      _kCalendarSyncTask,
      _kCalendarSyncTask,
      frequency: const Duration(hours: 6),
      constraints: Constraints(networkType: NetworkType.notRequired),
      existingWorkPolicy: ExistingPeriodicWorkPolicy.keep,
    );
  }

  // ── Calendar Reminders ──

  /// Schedule notifications for upcoming calendar events (next 7 days).
  /// De-duplicates against previously scheduled events.
  Future<void> scheduleCalendarReminders() async {
    final prefs = await SharedPreferences.getInstance();
    final enabled = prefs.getBool(_kEnabled) ?? true;
    if (!enabled) return;

    final reminderMinutes = prefs.getInt(_kReminderMinutes) ?? 30;

    try {
      final localData = LocalData();
      final now = DateTime.now();
      final events = await localData.getEvents(
        now,
        now.add(const Duration(days: 7)),
      );

      if (events.isEmpty) return;

      // Load already-scheduled event IDs to avoid duplicates
      final scheduledIds = Set<String>.from(
        prefs.getStringList(_kScheduledIds) ?? [],
      );

      final newScheduledIds = <String>[];

      for (final event in events) {
        final eventId = 'cal:${event.title}:${event.start.toIso8601String()}';

        // Skip if already scheduled or in the past
        final reminderTime = event.start.subtract(Duration(minutes: reminderMinutes));
        if (reminderTime.isBefore(now) || scheduledIds.contains(eventId)) {
          continue;
        }

        final notifId = eventId.hashCode.abs() % 2147483647; // int32 range

        final body = StringBuffer();
        if (event.location != null && event.location!.isNotEmpty) {
          body.write('at ${event.location} — ');
        }
        body.write(_formatTime(event.start));

        await _plugin.zonedSchedule(
          notifId,
          event.title,
          body.toString(),
          tz.TZDateTime.from(reminderTime, tz.local),
          NotificationDetails(
            android: AndroidNotificationDetails(
              _calendarChannelId,
              'Calendar Reminders',
              channelDescription: 'Reminders for upcoming calendar events',
              importance: Importance.high,
              priority: Priority.high,
            ),
            iOS: const DarwinNotificationDetails(
              presentAlert: true,
              presentBadge: true,
              presentSound: true,
            ),
          ),
          uiLocalNotificationDateInterpretation:
              UILocalNotificationDateInterpretation.absoluteTime,
          androidScheduleMode: AndroidScheduleMode.inexactAllowWhileIdle,
          payload: 'Tell me about my event "${event.title}" today',
          matchDateTimeComponents: null,
        );

        newScheduledIds.add(eventId);
        debugPrint('[NotificationService] Scheduled reminder: ${event.title} at $reminderTime');
      }

      // Persist scheduled IDs (keep recent ones, prune old)
      scheduledIds.addAll(newScheduledIds);
      // Only keep IDs for future events (prune entries older than 1 day)
      final cutoff = now.subtract(const Duration(days: 1)).toIso8601String();
      scheduledIds.removeWhere((id) {
        final parts = id.split(':');
        return parts.length >= 3 && parts.last.compareTo(cutoff) < 0;
      });
      await prefs.setStringList(_kScheduledIds, scheduledIds.toList());
    } catch (e) {
      debugPrint('[NotificationService] Calendar reminder error: $e');
    }
  }

  // ── Daily Briefing ──

  static const _upliftingMessages = [
    'You\'ve got this — make today count.',
    'Small steps forward are still steps forward.',
    'Today is full of possibility.',
    'Every morning is a fresh start.',
    'Your best work happens one moment at a time.',
    'Focus on what matters most today.',
    'Breathe, you\'re ready for this.',
    'A good day starts with intention.',
  ];

  /// Build a rich morning briefing body from today's calendar events.
  /// Detects medication, school, and errand keywords for extra emphasis.
  Future<String> _buildBriefingBody() async {
    try {
      final localData = LocalData();
      final now = DateTime.now();
      final todayStart = DateTime(now.year, now.month, now.day);
      final todayEnd = todayStart.add(const Duration(days: 1));
      final events = await localData.getEvents(todayStart, todayEnd);

      final parts = <String>[];

      if (events.isEmpty) {
        parts.add('No events today — enjoy the free time!');
      } else {
        final next = events.first;
        parts.add('${events.length} event${events.length > 1 ? 's' : ''} today. '
            'Next: ${next.title} at ${_formatTime(next.start)}.');

        // Surface high-priority reminders (medication, school, errands)
        final reminders = events.where((e) {
          final t = e.title.toLowerCase();
          return t.contains('medic') ||
              t.contains('pill') ||
              t.contains('tablet') ||
              t.contains('school') ||
              t.contains('drop') ||
              t.contains('pickup') ||
              t.contains('collect') ||
              t.contains('dentist') ||
              t.contains('doctor') ||
              t.contains('appoint');
        }).toList();

        if (reminders.isNotEmpty) {
          final labels = reminders.map((e) => e.title).join(', ');
          parts.add('Reminders: $labels.');
        }
      }

      // Append an uplifting thought
      final idx = now.day % _upliftingMessages.length;
      parts.add(_upliftingMessages[idx]);

      return parts.join(' ');
    } catch (_) {
      return 'Tap to see your schedule and start the day strong.';
    }
  }

  /// Show a daily briefing notification summarizing today's events.
  Future<void> _showDailyBriefing() async {
    final prefs = await SharedPreferences.getInstance();
    final enabled = prefs.getBool(_kBriefingEnabled) ?? true;
    if (!enabled) return;

    try {
      final body = await _buildBriefingBody();
      await _plugin.show(
        0,
        'Good morning!',
        body,
        NotificationDetails(
          android: AndroidNotificationDetails(
            _briefingChannelId,
            'Daily Briefing',
            channelDescription: 'Your daily schedule overview',
            importance: Importance.defaultImportance,
            priority: Priority.defaultPriority,
          ),
          iOS: const DarwinNotificationDetails(
            presentAlert: true,
            presentBadge: false,
            presentSound: true,
          ),
        ),
        payload: 'Morning briefing: summarise my schedule for today, any reminders, and an uplifting thought.',
      );
    } catch (e) {
      debugPrint('[NotificationService] Daily briefing error: $e');
    }
  }

  /// Schedule the daily briefing at the user's preferred time.
  Future<void> scheduleDailyBriefing() async {
    final prefs = await SharedPreferences.getInstance();
    final enabled = prefs.getBool(_kBriefingEnabled) ?? true;
    if (!enabled) return;

    final hour = prefs.getInt(_kBriefingHour) ?? 8;

    final now = tz.TZDateTime.now(tz.local);
    var briefingTime = tz.TZDateTime(tz.local, now.year, now.month, now.day, hour);
    if (briefingTime.isBefore(now)) {
      briefingTime = briefingTime.add(const Duration(days: 1));
    }

    await _plugin.zonedSchedule(
      1,
      'Good morning!',
      'Tap for your schedule, reminders, and a thought to start the day.',
      briefingTime,
      NotificationDetails(
        android: AndroidNotificationDetails(
          _briefingChannelId,
          'Daily Briefing',
          channelDescription: 'Your daily schedule overview',
          importance: Importance.defaultImportance,
          priority: Priority.defaultPriority,
        ),
        iOS: const DarwinNotificationDetails(
          presentAlert: true,
          presentBadge: false,
          presentSound: true,
        ),
      ),
      uiLocalNotificationDateInterpretation:
          UILocalNotificationDateInterpretation.absoluteTime,
      androidScheduleMode: AndroidScheduleMode.inexactAllowWhileIdle,
      payload: 'Morning briefing: summarise my schedule for today, any reminders, and an uplifting thought.',
      matchDateTimeComponents: DateTimeComponents.time,
    );
  }

  // ── Afternoon briefing ──

  /// Build the afternoon briefing body: remaining events + practical reminders.
  Future<String> _buildAfternoonBriefingBody() async {
    try {
      final localData = LocalData();
      final now = DateTime.now();
      final isWeekend = now.weekday == DateTime.saturday || now.weekday == DateTime.sunday;
      final remaining = await localData.getEvents(
        now,
        DateTime(now.year, now.month, now.day, 23, 59),
      );

      final parts = <String>[];

      if (isWeekend) {
        parts.add('Afternoon check-in!');
        if (remaining.isEmpty) {
          parts.add('No more plans today — perfect time for a walk, hobby, or just relaxing.');
        } else {
          parts.add('${remaining.length} thing${remaining.length > 1 ? 's' : ''} left today: ${remaining.first.title}.');
        }
      } else {
        parts.add('Afternoon check-in!');
        if (remaining.isEmpty) {
          parts.add('Nothing left on the calendar — wrap up and head home when ready.');
        } else {
          parts.add('${remaining.length} event${remaining.length > 1 ? 's' : ''} remaining. '
              'Next: ${remaining.first.title} at ${_formatTime(remaining.first.start)}.');
        }
        parts.add('Plan your commute if heading out soon.');
      }

      return parts.join(' ');
    } catch (_) {
      return 'Tap for your afternoon summary and what\'s coming up.';
    }
  }

  /// Show an afternoon check-in notification (around 1 PM on weekdays, noon on weekends).
  Future<void> _showAfternoonBriefing() async {
    final prefs = await SharedPreferences.getInstance();
    final enabled = prefs.getBool(_kBriefingEnabled) ?? true;
    if (!enabled) return;

    try {
      final body = await _buildAfternoonBriefingBody();
      await _plugin.show(
        2,
        'Afternoon check-in',
        body,
        NotificationDetails(
          android: AndroidNotificationDetails(
            _briefingChannelId,
            'Daily Briefing',
            channelDescription: 'Your daily schedule overview',
            importance: Importance.defaultImportance,
            priority: Priority.defaultPriority,
          ),
          iOS: const DarwinNotificationDetails(
            presentAlert: true,
            presentBadge: false,
            presentSound: false,
          ),
        ),
        payload: 'Afternoon briefing: what\'s left on my schedule today? Any traffic or commute heads-up? '
            'If weekend, suggest a relaxing or fun activity based on the time of year.',
      );
    } catch (e) {
      debugPrint('[NotificationService] Afternoon briefing error: $e');
    }
  }

  /// Schedule the afternoon check-in notification (daily at 13:00).
  Future<void> scheduleAfternoonBriefing() async {
    final prefs = await SharedPreferences.getInstance();
    final enabled = prefs.getBool(_kBriefingEnabled) ?? true;
    if (!enabled) return;

    final now = tz.TZDateTime.now(tz.local);
    var briefingTime = tz.TZDateTime(tz.local, now.year, now.month, now.day, 13);
    if (briefingTime.isBefore(now)) {
      briefingTime = briefingTime.add(const Duration(days: 1));
    }

    await _plugin.zonedSchedule(
      3,
      'Afternoon check-in',
      'Tap for your afternoon summary and what\'s coming up.',
      briefingTime,
      NotificationDetails(
        android: AndroidNotificationDetails(
          _briefingChannelId,
          'Daily Briefing',
          channelDescription: 'Your daily schedule overview',
          importance: Importance.defaultImportance,
          priority: Priority.defaultPriority,
        ),
        iOS: const DarwinNotificationDetails(
          presentAlert: true,
          presentBadge: false,
          presentSound: false,
        ),
      ),
      uiLocalNotificationDateInterpretation:
          UILocalNotificationDateInterpretation.absoluteTime,
      androidScheduleMode: AndroidScheduleMode.inexactAllowWhileIdle,
      payload: 'Afternoon briefing: what\'s left on my schedule today? Any traffic or commute heads-up? '
          'If weekend, suggest a relaxing or fun activity based on the time of year.',
      matchDateTimeComponents: DateTimeComponents.time,
    );
  }

  // ── Settings ──

  Future<bool> get isEnabled async =>
      (await SharedPreferences.getInstance()).getBool(_kEnabled) ?? true;

  Future<void> setEnabled(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_kEnabled, value);
    if (!value) {
      await _plugin.cancelAll();
    } else {
      await scheduleCalendarReminders();
      await scheduleDailyBriefing();
      await scheduleAfternoonBriefing();
    }
  }

  Future<int> get reminderMinutes async =>
      (await SharedPreferences.getInstance()).getInt(_kReminderMinutes) ?? 30;

  Future<void> setReminderMinutes(int minutes) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_kReminderMinutes, minutes);
    // Re-schedule with new timing
    await _plugin.cancelAll();
    await scheduleCalendarReminders();
    await scheduleDailyBriefing();
  }

  Future<bool> get isBriefingEnabled async =>
      (await SharedPreferences.getInstance()).getBool(_kBriefingEnabled) ?? true;

  Future<void> setBriefingEnabled(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_kBriefingEnabled, value);
    if (value) {
      await scheduleDailyBriefing();
    } else {
      await _plugin.cancel(1); // Cancel briefing notification
    }
  }

  Future<int> get briefingHour async =>
      (await SharedPreferences.getInstance()).getInt(_kBriefingHour) ?? 8;

  Future<void> setBriefingHour(int hour) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_kBriefingHour, hour);
    await _plugin.cancel(1);
    await scheduleDailyBriefing();
  }

  // ── Helpers ──

  String _formatTime(DateTime dt) {
    final hour = dt.hour > 12 ? dt.hour - 12 : (dt.hour == 0 ? 12 : dt.hour);
    final amPm = dt.hour >= 12 ? 'PM' : 'AM';
    final min = dt.minute.toString().padLeft(2, '0');
    return '$hour:$min $amPm';
  }
}
