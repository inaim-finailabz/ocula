import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';
import 'screens/sessions_screen.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'services/ai_manager.dart';
import 'services/speech_service.dart';
import 'services/export_service.dart';
import 'services/indexer.dart';
import 'services/local_data.dart';
import 'services/model_manager.dart';
import 'services/orchestrator.dart';
import 'services/ocula_db.dart';
import 'services/share_receiver.dart';
import 'screens/splash_screen.dart';
import 'screens/onboarding_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/recorder_screen.dart';
import 'widgets/ocula_orb.dart';
import 'widgets/help_tour.dart';
import 'services/env_config.dart';
import 'services/action_service.dart';
import 'services/notification_service.dart';
import 'services/tray_service.dart';
import 'services/user_profile.dart';
import 'widgets/action_card.dart';
import 'widgets/result_cards.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  debugPrint('[Ocula] ENV=${EnvConfig.env} SERVER=${EnvConfig.modelServerUrl}');
  if (Platform.isMacOS || Platform.isWindows) {
    await TrayService.instance.init();
  }
  runApp(const OculaApp());
}

class OculaApp extends StatelessWidget {
  const OculaApp({super.key});

  static const _colorScheme = ColorScheme(
    brightness: Brightness.dark,
    primary: Color(0xFF8B7CF6),
    onPrimary: Colors.white,
    primaryContainer: Color(0xFF2B2550),
    onPrimaryContainer: Color(0xFFE4DEFF),
    secondary: Color(0xFF2DD4BF),
    onSecondary: Colors.black,
    tertiary: Color(0xFFFF8A80),
    error: Color(0xFFFF6B6B),
    onError: Colors.white,
    surface: Color(0xFF111218),
    onSurface: Color(0xFFECECF1),
    onSurfaceVariant: Color(0xFF9A9BAA),
    surfaceContainer: Color(0xFF181922),
    surfaceContainerHigh: Color(0xFF1E1F2A),
    surfaceContainerHighest: Color(0xFF252633),
    outline: Color(0xFF3A3B4C),
    outlineVariant: Color(0xFF2A2B38),
  );

  @override
  Widget build(BuildContext context) {
    final base = ThemeData(colorScheme: _colorScheme, useMaterial3: true);
    return MaterialApp(
      title: 'Ocula',
      debugShowCheckedModeBanner: false,
      theme: base.copyWith(
        scaffoldBackgroundColor: _colorScheme.surface,
        textTheme: base.textTheme.apply(
          bodyColor: _colorScheme.onSurface,
          displayColor: _colorScheme.onSurface,
        ),
        appBarTheme: AppBarTheme(
          backgroundColor: _colorScheme.surface,
          elevation: 0,
          scrolledUnderElevation: 0,
        ),
        iconButtonTheme: IconButtonThemeData(
          style: IconButton.styleFrom(
            foregroundColor: _colorScheme.onSurfaceVariant,
            visualDensity: VisualDensity.compact,
          ),
        ),
        bottomSheetTheme: BottomSheetThemeData(
          backgroundColor: _colorScheme.surfaceContainer,
          surfaceTintColor: Colors.transparent,
          showDragHandle: true,
          dragHandleColor: _colorScheme.outline,
          shape: const RoundedRectangleBorder(
            borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
          ),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(14),
            ),
          ),
        ),
        snackBarTheme: SnackBarThemeData(
          behavior: SnackBarBehavior.floating,
          backgroundColor: _colorScheme.surfaceContainerHighest,
          contentTextStyle: TextStyle(color: _colorScheme.onSurface),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        popupMenuTheme: PopupMenuThemeData(
          color: _colorScheme.surfaceContainerHigh,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
        dividerTheme: DividerThemeData(color: _colorScheme.outlineVariant),
      ),
      initialRoute: '/',
      routes: {
        '/': (context) => const OculaSplashScreen(),
        '/onboarding': (context) => const OnboardingScreen(),
        '/home': (context) => const AssistantScreen(),
        '/recorder': (context) => const RecorderScreen(),
      },
    );
  }
}

/// One screen. One assistant. Talk, type, or show it something.
///
/// Layout: Chat transcript fills the screen with a gradient background.
/// The orb floats on top as a 3D overlay — large when speaking/listening,
/// small when idle. Tap the orb to toggle views.
class AssistantScreen extends StatefulWidget {
  const AssistantScreen({super.key});

  @override
  State<AssistantScreen> createState() => _AssistantScreenState();
}

class _AssistantScreenState extends State<AssistantScreen>
    with SingleTickerProviderStateMixin {
  final _ai = AIManager();
  final _orchestrator = Orchestrator();
  final _export = ExportService();
  final _shareReceiver = ShareReceiver();
  late final SpeechService _speech;
  final _textController = TextEditingController();
  final _inputFocus = FocusNode();
  final _scrollController = ScrollController();
  final _modelManager = OculaModelManager();
  StreamSubscription? _downloadProgressSubscription;
  StreamSubscription? _featureReadySubscription;
  StreamSubscription? _tierChangeSubscription;
  StreamSubscription? _freeModelStatusSubscription;
  StreamSubscription? _stepSubscription;
  String? _currentStepLabel;

  bool _freeModelFailed = false;
  static final bool _isDesktop =
      Platform.isMacOS || Platform.isWindows || Platform.isLinux;
  bool _isTabletOrDesktop = _isDesktop;

  // ── Help Tour ──
  static final _orbKey = GlobalKey();
  static final _chatListKey = GlobalKey();
  static final _cameraButtonKey = GlobalKey();
  static final _inputBarKey = GlobalKey();
  bool _showingHelpTour = false;
  String _sessionId = const Uuid().v4();

  void _startNewSession() {
    if (mounted) {
      setState(() {
        _messages.clear();
        _sessionId = const Uuid().v4();
      });
    }
  }

  void _openSessionHistory(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => SessionsScreen(onStartNewChat: _startNewSession),
    );
  }

  /// Called when the free model finishes installing in the background.
  /// Loads it into memory and shows a brief snackbar.
  Future<void> _autoLoadFreeModel() async {
    try {
      if (!await _modelManager.isFreeModelReady) return;
      await _ai.switchEngine(AITier.free);
      if (mounted) _showSnackbar('AI engine ready');
    } catch (e) {
      debugPrint('[Home] Free model auto-load failed: $e');
    }
  }

  /// Ensure Lite is activated if files are already installed and verified.
  /// Keeps existing startup flow intact and only repairs late activation races.
  Future<void> _ensureLiteActivatedIfReady() async {
    if (_ai.isModelLoaded) return;
    if (!await _modelManager.isFreeModelReady) return;
    await _autoLoadFreeModel();
  }

  /// Retry a failed free-model install from the home screen download banner.
  void _retryFreeModelInstall() {
    if (!mounted) return;
    setState(() => _freeModelFailed = false);
    _modelManager.ensureFreeModelReady().catchError((e) {
      debugPrint('[Home] Retry install error: $e');
      return false;
    });
  }

  /// On tablet/desktop: silently download Plus in the background after Lite loads.
  /// Plus is the recommended default for these form factors. Progress is shown
  /// in the home screen banner. The AIManager auto-switches when it's ready.
  Future<void> _autoDownloadPlusIfNeeded() async {
    if (await _ai.isTierDownloaded(AITier.plus)) return;
    if (!await _ai.canDeviceRunTier(AITier.plus)) return;
    debugPrint('[Home] Tablet/desktop: auto-downloading Plus tier...');
    _modelManager
        .downloadTierWithProgress(AITier.plus)
        .then((ok) {
          if (!ok && mounted)
            setState(() => _backgroundDownloadFailed = AITier.plus);
        })
        .catchError((e) {
          debugPrint('[Home] Plus auto-download error: $e');
          if (mounted) setState(() => _backgroundDownloadFailed = AITier.plus);
        });
  }

  /// Proactively send a morning briefing as the first AI message of the day.
  /// Pulls calendar context via the orchestrator; no user turn is shown.
  /// Load user preferences from SharedPreferences and return a context snippet.
  static Future<String> _loadUserPrefsContext() async {
    final p = await SharedPreferences.getInstance();
    final parts = <String>[];
    final teams = p.getString('pref_sports_teams') ?? '';
    final music = p.getString('pref_music_artists') ?? '';
    final movies = p.getString('pref_movies_actors') ?? '';
    final shopping = p.getString('pref_shopping_interests') ?? '';
    final other = p.getString('pref_other_interests') ?? '';
    if (teams.isNotEmpty) parts.add('sports teams: $teams');
    if (music.isNotEmpty) parts.add('music: $music');
    if (movies.isNotEmpty) parts.add('films/actors: $movies');
    if (shopping.isNotEmpty) parts.add('shopping interests: $shopping');
    if (other.isNotEmpty) parts.add('other interests: $other');
    if (parts.isEmpty) return '';
    return 'User preferences — ${parts.join('; ')}.';
  }

  /// Build the contextual briefing prompt based on time of day and day type.
  static String _briefingPrompt(DateTime now) {
    final day = _weekdayName(now.weekday);
    final month = _monthName(now.month);
    final date = '${day}, ${month} ${now.day}';
    final isWeekend =
        now.weekday == DateTime.saturday || now.weekday == DateTime.sunday;
    final hour = now.hour;

    if (hour >= 6 && hour < 12) {
      // Morning
      if (isWeekend) {
        return 'Good morning! Today is $date (weekend). '
            'Check my calendar for any plans today. '
            'Then suggest one relaxing or fun activity — a walk, hobby, sport, home project, or outing — '
            'based on the season (${month}). Keep it friendly and brief.';
      } else {
        return 'Morning briefing — today is $date. '
            'Summarise my schedule: meetings, events, and calendar reminders '
            '(medication, school drop-off, pickups, appointments). '
            'Recommend a good breakfast if it is still early. '
            'Add one short uplifting thought. Be concise and friendly.';
      }
    } else if (hour >= 12 && hour < 17) {
      // Afternoon
      if (isWeekend) {
        return 'Afternoon check-in — today is $date (weekend). '
            'Any plans left on the calendar? '
            'Suggest a fun afternoon activity for the season (${month}): '
            'a sport, game, outing, picnic, DIY project, or similar. '
            'Recommend a light lunch or afternoon snack. Keep it short and upbeat.';
      } else {
        return 'Afternoon check-in — today is $date. '
            'What events are still ahead today? '
            'Is there anything to prepare for the commute home? '
            'Suggest a good lunch or afternoon snack if around midday. '
            'Keep it brief and practical.';
      }
    } else {
      // Evening / fallback
      if (isWeekend) {
        return 'Evening check-in — today is $date (weekend). '
            'Any evening plans on the calendar? '
            'Suggest a relaxing activity: a film, book, game, or gentle walk. '
            'Keep it short and friendly.';
      } else {
        return 'Evening wrap-up — today is $date. '
            'Any remaining events tonight? '
            'Highlight anything to prepare for tomorrow from the calendar. '
            'Suggest a light dinner or wind-down activity. Keep it brief.';
      }
    }
  }

  /// Today's remaining events (plus tomorrow's after 17:00) as grounded
  /// model context, tappable calendar cards, and a plain fallback text.
  static Future<
    ({String context, List<LinkedAsset> assets, String fallbackText})
  >
  _loadBriefingSchedule(DateTime now) async {
    final dayStart = DateTime(now.year, now.month, now.day);
    final morning = now.hour < 12;
    final evening = now.hour >= 17;
    final from = morning ? dayStart : now.subtract(const Duration(hours: 1));
    final to = dayStart.add(Duration(days: evening ? 2 : 1));
    final events =
        (await LocalData().getEvents(
            from,
            to,
          )).where((e) => e.end.isAfter(from)).toList()
          ..sort((a, b) => a.start.compareTo(b.start));

    String hhmm(DateTime d) =>
        '${d.hour.toString().padLeft(2, '0')}:${d.minute.toString().padLeft(2, '0')}';
    String when(LocalEvent e) {
      final allDay = e.end.difference(e.start).inHours >= 23;
      final dayLabel = e.start.isBefore(dayStart.add(const Duration(days: 1)))
          ? 'Today'
          : 'Tomorrow';
      return allDay
          ? '$dayLabel · all day'
          : '$dayLabel · ${hhmm(e.start)}–${hhmm(e.end)}';
    }

    final lines = <String>[];
    final assets = <LinkedAsset>[];
    for (final e in events) {
      final loc = (e.location ?? '').trim();
      lines.add('- ${when(e)}: ${e.title}${loc.isEmpty ? '' : ' @ $loc'}');
      assets.add(
        LinkedAsset(
          sourceId: 'cal:${e.title}:${e.start.toIso8601String()}',
          assetType: 'calendar',
          assetRef: 'cal:${e.start.millisecondsSinceEpoch}',
          label: e.title,
          snippet: [when(e), if (loc.isNotEmpty) loc].join('\n'),
        ),
      );
    }

    final context = events.isEmpty
        ? '[SOURCE 1]\nType: Calendar\nContent: No events on the calendar '
              'for ${evening ? 'the rest of today or tomorrow' : 'today'}.'
        : '[SOURCE 1]\nType: Calendar (device, live)\nContent:\n${lines.join('\n')}';
    final fallbackText = events.isEmpty
        ? 'Your calendar is clear ${evening ? 'for tonight and tomorrow' : 'today'}.'
        : 'You have ${events.length} event${events.length == 1 ? '' : 's'} coming up:';
    return (context: context, assets: assets, fallbackText: fallbackText);
  }

  /// Fire a contextual briefing as a proactive AI message (no user turn shown).
  Future<void> _triggerMorningBriefing() async {
    if (!mounted || _isThinking || !_ai.isModelLoaded) return;

    final now = DateTime.now();
    final today =
        '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}';
    final prefs = await SharedPreferences.getInstance();
    final prefKey = (now.hour >= 12 && now.hour < 17)
        ? 'last_afternoon_briefing_date'
        : 'last_morning_briefing_date';
    await prefs.setString(prefKey, today);

    setState(() {
      _isThinking = true;
      _orbState = OrbState.thinking;
      _orbExpanded = true;
    });
    _orbSizeController.forward();

    try {
      // Read the device calendar directly — semantic search over indexed
      // chunks can't reliably pick "today", which made the briefing claim
      // the schedule was empty.
      final schedule = await _loadBriefingSchedule(now);
      final prefsCtx = await _loadUserPrefsContext();
      final prompt = [
        _briefingPrompt(now),
        if (prefsCtx.isNotEmpty)
          '$prefsCtx\nUse these preferences to enrich activity suggestions.',
        'Only mention events listed in the sources. If none are listed, say '
            'the calendar is clear — do not invent events.',
      ].join('\n\n');
      final result = await _orchestrator
          .run(
            prompt,
            retrievalScope: RetrievalScope.calendar,
            sessionId: _sessionId,
            groundedContext: schedule.context,
          )
          .timeout(
            const Duration(seconds: 45),
            onTimeout: () => OrchestratorResult(''),
          );

      if (!mounted) return;
      // Event cards are shown even if the model times out or returns nothing.
      final text = result.response.isNotEmpty
          ? result.response
          : schedule.fallbackText;
      if (text.isNotEmpty) {
        setState(() {
          _messages.add(
            _Message(text: text, isUser: false, linkedAssets: schedule.assets),
          );
          _isThinking = false;
          _orbState = OrbState.idle;
          _orbExpanded = false;
        });
        _scrollToBottom();
      } else {
        setState(() {
          _isThinking = false;
          _orbState = OrbState.idle;
          _orbExpanded = false;
        });
      }
    } catch (e) {
      debugPrint('[Ocula] Briefing error: $e');
      if (mounted) {
        setState(() {
          _isThinking = false;
          _orbState = OrbState.idle;
          _orbExpanded = false;
        });
      }
    }
    _orbSizeController.reverse();
  }

  static String _weekdayName(int weekday) => const [
    '',
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
  ][weekday];

  static String _monthName(int month) => const [
    '',
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
  ][month];

  void _startHelpTour() {
    if (mounted) setState(() => _showingHelpTour = true);
  }

  final List<_Message> _messages = [];
  bool _isThinking = false;
  bool _isListening = false;
  bool _isSpeaking = false;
  bool _stopRequested = false;
  OrbState _orbState = OrbState.idle;
  RetrievalScope _retrievalScope = RetrievalScope.all;
  bool _morningBriefingPending = false;
  bool _showDataSourcesBanner = false;
  File? _attachedImage;
  File? _attachedDocument;
  String? _attachedDocName;

  bool _orbExpanded = true;
  Map<String, double> _activeDownloads = {};
  AITier? _backgroundDownloadFailed;
  AvatarStyle _avatarStyle = AvatarStyle.face;
  String? _customAvatarPath;

  late final AnimationController _orbSizeController;

  @override
  void initState() {
    super.initState();
    _speech = SpeechService(aiManager: _ai);
    _speech.init();
    // If the splash navigated before the model was ready (background install
    // path), auto-load when ensureFreeModelReady signals success.
    _freeModelStatusSubscription = _modelManager.freeModelStatusStream.listen((
      ok,
    ) {
      if (!mounted) return;
      if (ok) {
        setState(() => _freeModelFailed = false);
        _ensureLiteActivatedIfReady().then((_) {
          if (_morningBriefingPending && mounted) {
            _morningBriefingPending = false;
            _triggerMorningBriefing();
          }
        });
        // On tablet/desktop, auto-download Plus after Lite is ready.
        // Plus is the default tier for these form factors (better quality,
        // vision support). On phones this stays strictly on-demand.
        if (_isTabletOrDesktop) {
          _autoDownloadPlusIfNeeded();
        }
      } else {
        setState(() => _freeModelFailed = true);
      }
    });
    // Handle race: model may have finished installing before home screen opened.
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      // Detect tablet (iPad or large-screen Android) by screen size.
      final shortestSide = MediaQuery.sizeOf(context).shortestSide;
      if (shortestSide > 600 && !_isDesktop) {
        _isTabletOrDesktop = true;
      }
      await _ensureLiteActivatedIfReady();
    });

    _orchestrator.onAskCapability = (_) => _showInternetDialog();
    _orchestrator.onConnectivityNeeded = _showConnectivityDialog;

    _stepSubscription = _orchestrator.stepStream.listen((step) {
      if (!mounted) return;
      setState(() {
        _currentStepLabel = step.type == AgentStepType.complete
            ? null
            : step.label;
      });
    });
    Indexer().startBackgroundIndexing();

    // Initialize notifications — calendar reminders + daily briefing
    _initNotifications();
    _textController.addListener(() => setState(() {}));

    _orbSizeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
    );

    // Listen for shared content from other apps (share sheet).
    _shareReceiver.onSharedContent = (displayText, query) {
      if (mounted) {
        // Show what was shared and auto-send to assistant
        setState(() {
          _messages.add(_Message(text: displayText, isUser: true));
        });
        _scrollToBottom();
        _send(query);
      }
    };
    _shareReceiver.init();

    // Listen for feature-ready notifications (user-friendly, no model names).
    _featureReadySubscription = _modelManager.featureReadyStream.listen((
      featureName,
    ) {
      _showFeatureReady(featureName);
    });

    // Track active background downloads for the in-chat progress banner.
    _downloadProgressSubscription = _modelManager.downloadProgressStream.listen(
      (progress) {
        if (mounted) {
          setState(() {
            _activeDownloads = Map.from(progress)
              ..removeWhere((_, v) => v >= 1.0);
          });
        }
      },
    );

    // Update tier badge when model switches (e.g. from settings or auto-route).
    _tierChangeSubscription = _ai.activeTierStream.listen((_) {
      if (mounted) setState(() {});
    });

    // Check if we should show the help tour after first onboarding.
    // Also restore avatar style preference and check for morning briefing.
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      UserProfile().load().catchError((_) {});
      final prefs = await SharedPreferences.getInstance();
      if ((prefs.getBool('show_help_tour') ?? false) && mounted) {
        _startHelpTour();
      }
      final styleIndex = prefs.getInt('avatar_style') ?? AvatarStyle.face.index;
      final customPath = prefs.getString('avatar_custom_path');
      if (mounted) {
        setState(() {
          _avatarStyle = AvatarStyle
              .values[styleIndex.clamp(0, AvatarStyle.values.length - 1)];
          _customAvatarPath = customPath;
        });
      }

      // Data-sources banner: shown once until dismissed when email is not yet set up.
      final emailConfigured = await LocalData().isEmailConfigured;
      final bannerDismissed =
          prefs.getBool('data_sources_banner_dismissed') ?? false;
      if (!emailConfigured && !bannerDismissed && mounted) {
        setState(() => _showDataSourcesBanner = true);
      }

      // Contextual briefing: morning (6–11) or afternoon (12–17), once per window.
      final now = DateTime.now();
      final today =
          '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}';
      final isMorningWindow = now.hour >= 6 && now.hour < 11;
      final isAfternoonWindow = now.hour >= 12 && now.hour < 17;
      if (isMorningWindow || isAfternoonWindow) {
        final prefKey = isMorningWindow
            ? 'last_morning_briefing_date'
            : 'last_afternoon_briefing_date';
        final lastBriefing = prefs.getString(prefKey) ?? '';
        if (lastBriefing != today) {
          if (_ai.isModelLoaded && mounted) {
            _triggerMorningBriefing();
          } else {
            setState(() => _morningBriefingPending = true);
          }
        }
      }
    });
  }

  void _showAvatarStylePicker() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Theme.of(context).colorScheme.surfaceContainerHighest,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) {
        final colors = Theme.of(ctx).colorScheme;
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Assistant Appearance',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: colors.onSurface,
                  ),
                ),
                const SizedBox(height: 16),
                _avatarTile(
                  ctx: ctx,
                  colors: colors,
                  style: AvatarStyle.face,
                  icon: Icons.grid_on_outlined,
                  title: 'Wireframe Face',
                  subtitle: '3D digital face, drag to rotate',
                ),
                _avatarTile(
                  ctx: ctx,
                  colors: colors,
                  style: AvatarStyle.orb,
                  icon: Icons.blur_circular_outlined,
                  title: 'Orb',
                  subtitle: 'Animated energy sphere',
                ),
                _avatarTile(
                  ctx: ctx,
                  colors: colors,
                  style: AvatarStyle.custom,
                  icon: Icons.image_outlined,
                  title: 'Custom Image',
                  subtitle:
                      'PNG, JPG or SVG — transparent backgrounds supported',
                  onTap: () async {
                    Navigator.pop(ctx);
                    await _pickCustomAvatar();
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _avatarTile({
    required BuildContext ctx,
    required ColorScheme colors,
    required AvatarStyle style,
    required IconData icon,
    required String title,
    required String subtitle,
    VoidCallback? onTap,
  }) {
    final selected = _avatarStyle == style;
    return ListTile(
      leading: Icon(icon, color: selected ? colors.primary : colors.onSurface),
      title: Text(title),
      subtitle: Text(
        subtitle,
        style: TextStyle(fontSize: 12, color: colors.onSurface.withAlpha(140)),
      ),
      trailing: selected
          ? Icon(Icons.check_circle, color: colors.primary)
          : null,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      tileColor: selected ? colors.primary.withAlpha(18) : Colors.transparent,
      onTap:
          onTap ??
          () async {
            Navigator.pop(ctx);
            setState(() => _avatarStyle = style);
            final prefs = await SharedPreferences.getInstance();
            await prefs.setInt('avatar_style', style.index);
          },
    );
  }

  /// Pick a PNG/JPG/SVG file for the custom avatar.
  Future<void> _pickCustomAvatar() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['png', 'jpg', 'jpeg', 'webp', 'svg'],
        allowMultiple: false,
      );
      if (result == null || result.files.single.path == null) return;
      final path = result.files.single.path!;
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('avatar_custom_path', path);
      await prefs.setInt('avatar_style', AvatarStyle.custom.index);
      if (mounted) {
        setState(() {
          _customAvatarPath = path;
          _avatarStyle = AvatarStyle.custom;
        });
      }
    } catch (e) {
      if (mounted) _showSnackbar('Could not pick image: $e');
    }
  }

  void _cancelBannerDownload() {
    final downloading = List<String>.from(_activeDownloads.keys);
    for (final fileName in downloading) {
      _modelManager.cancelDownload(fileName);
    }
    setState(() => _backgroundDownloadFailed = null);
  }

  void _resumeBannerDownload(AITier tier) {
    setState(() => _backgroundDownloadFailed = null);
    _modelManager
        .downloadTierWithProgress(tier)
        .then((ok) {
          if (!ok && mounted) setState(() => _backgroundDownloadFailed = tier);
        })
        .catchError((e) {
          debugPrint('[Home] Resume download error: $e');
          if (mounted) setState(() => _backgroundDownloadFailed = tier);
        });
  }

  /// Thin banner below the top bar for background model downloads and failures.
  Widget _buildDownloadBanner(ColorScheme colors) {
    // ── Failure state: free model install failed — show retry ──
    if (_freeModelFailed && _activeDownloads.isEmpty) {
      return Container(
        padding: const EdgeInsets.fromLTRB(16, 6, 12, 8),
        decoration: BoxDecoration(
          color: colors.errorContainer.withAlpha(50),
          border: Border(
            bottom: BorderSide(color: colors.error.withAlpha(40), width: 0.5),
          ),
        ),
        child: Row(
          children: [
            Icon(Icons.warning_amber_rounded, size: 14, color: colors.error),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                'AI engine failed to install',
                style: TextStyle(fontSize: 11, color: colors.error),
              ),
            ),
            TextButton(
              onPressed: _retryFreeModelInstall,
              style: TextButton.styleFrom(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 4,
                ),
                minimumSize: Size.zero,
                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                foregroundColor: colors.error,
              ),
              child: const Text('Retry', style: TextStyle(fontSize: 11)),
            ),
          ],
        ),
      );
    }

    // ── Background Plus/Pro download failed — show resume ──
    if (_backgroundDownloadFailed != null && _activeDownloads.isEmpty) {
      final tier = _backgroundDownloadFailed!;
      final tierName = tier == AITier.plus ? 'Plus' : 'Pro';
      return Container(
        padding: const EdgeInsets.fromLTRB(16, 6, 4, 8),
        decoration: BoxDecoration(
          color: colors.errorContainer.withAlpha(50),
          border: Border(
            bottom: BorderSide(color: colors.error.withAlpha(40), width: 0.5),
          ),
        ),
        child: Row(
          children: [
            Icon(Icons.warning_amber_rounded, size: 14, color: colors.error),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                '$tierName download failed',
                style: TextStyle(fontSize: 11, color: colors.error),
              ),
            ),
            TextButton(
              onPressed: () => _resumeBannerDownload(tier),
              style: TextButton.styleFrom(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 4,
                ),
                minimumSize: Size.zero,
                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                foregroundColor: colors.primary,
              ),
              child: const Text('Resume', style: TextStyle(fontSize: 11)),
            ),
            IconButton(
              onPressed: () => setState(() => _backgroundDownloadFailed = null),
              icon: Icon(
                Icons.close,
                size: 14,
                color: colors.onSurface.withAlpha(120),
              ),
              padding: const EdgeInsets.all(6),
              constraints: const BoxConstraints(),
              style: IconButton.styleFrom(
                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              ),
            ),
          ],
        ),
      );
    }

    // ── Normal download progress ──
    // Prefer showing the main (non-projector, non-embed) model being downloaded.
    // Falls back to any downloading file so progress is never hidden.
    final primary =
        _activeDownloads.entries.where((e) {
          final m = OculaModelManager.models
              .where((m) => m.fileName == e.key)
              .firstOrNull;
          return m != null && !m.isVisionProjector && !m.isEmbeddingModel;
        }).firstOrNull ??
        _activeDownloads.entries.firstOrNull;

    if (primary == null) return const SizedBox.shrink();

    final modelInfo = OculaModelManager.models
        .where((m) => m.fileName == primary.key)
        .firstOrNull;
    final tierForCancel = modelInfo?.tier;
    final label = modelInfo?.displayName ?? primary.key.split('.').first;
    final pct = (primary.value * 100).toStringAsFixed(0);

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 6, 4, 8),
      decoration: BoxDecoration(
        color: colors.primaryContainer.withAlpha(45),
        border: Border(
          bottom: BorderSide(color: colors.primary.withAlpha(30), width: 0.5),
        ),
      ),
      child: Row(
        children: [
          SizedBox(
            width: 14,
            height: 14,
            child: CircularProgressIndicator(
              value: primary.value > 0 ? primary.value : null,
              strokeWidth: 1.5,
              color: colors.primary,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'Installing $label ($pct%)',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w500,
                    color: colors.primary,
                  ),
                ),
                const SizedBox(height: 3),
                ClipRRect(
                  borderRadius: BorderRadius.circular(1),
                  child: LinearProgressIndicator(
                    value: primary.value > 0 ? primary.value : null,
                    minHeight: 2,
                    backgroundColor: colors.primary.withAlpha(25),
                    valueColor: AlwaysStoppedAnimation<Color>(colors.primary),
                  ),
                ),
              ],
            ),
          ),
          if (tierForCancel != null && tierForCancel != AITier.free)
            IconButton(
              onPressed: _cancelBannerDownload,
              icon: Icon(
                Icons.close,
                size: 14,
                color: colors.onSurface.withAlpha(150),
              ),
              padding: const EdgeInsets.all(6),
              constraints: const BoxConstraints(),
              style: IconButton.styleFrom(
                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              ),
              tooltip: 'Cancel download',
            ),
        ],
      ),
    );
  }

  /// Show a friendly notification when a new feature becomes available.
  void _showFeatureReady(String featureName) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const Icon(Icons.check_circle, color: Colors.greenAccent, size: 20),
            const SizedBox(width: 10),
            Expanded(child: Text('$featureName is ready to use')),
          ],
        ),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        duration: const Duration(seconds: 4),
        margin: const EdgeInsets.fromLTRB(16, 0, 16, 24),
      ),
    );
  }

  Future<void> _initNotifications() async {
    try {
      final notifService = NotificationService();
      await notifService.init();
      await notifService.requestPermission();
      // Pre-fill a query when user taps a notification
      notifService.onNotificationTap = (query) {
        if (mounted) {
          _send(query);
        }
      };
      // Schedule initial reminders + morning and afternoon briefings
      await notifService.scheduleCalendarReminders();
      await notifService.scheduleDailyBriefing();
      await notifService.scheduleAfternoonBriefing();
    } catch (e) {
      debugPrint('[Ocula] Notification init error: $e');
    }
  }

  Widget _buildDataSourcesBanner(ColorScheme colors) {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 8, 8, 8),
      decoration: BoxDecoration(
        color: colors.primaryContainer.withAlpha(60),
        border: Border(
          bottom: BorderSide(color: colors.primary.withAlpha(30), width: 0.5),
        ),
      ),
      child: Row(
        children: [
          Icon(Icons.hub_outlined, size: 14, color: colors.primary),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              'Connect email & messaging to unlock more context',
              style: TextStyle(
                fontSize: 11,
                color: colors.onSurface.withAlpha(180),
              ),
            ),
          ),
          TextButton(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => SettingsScreen(speech: _speech),
                ),
              );
            },
            style: TextButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              minimumSize: Size.zero,
              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              foregroundColor: colors.primary,
            ),
            child: const Text('Set up', style: TextStyle(fontSize: 11)),
          ),
          IconButton(
            icon: Icon(
              Icons.close,
              size: 14,
              color: colors.onSurface.withAlpha(120),
            ),
            padding: const EdgeInsets.all(6),
            constraints: const BoxConstraints(),
            style: IconButton.styleFrom(
              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
            ),
            tooltip: 'Dismiss',
            onPressed: () async {
              setState(() => _showDataSourcesBanner = false);
              final prefs = await SharedPreferences.getInstance();
              await prefs.setBool('data_sources_banner_dismissed', true);
            },
          ),
        ],
      ),
    );
  }

  void _showSnackbar(String message, {Duration? duration}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        duration: duration ?? const Duration(seconds: 4),
        action: SnackBarAction(
          label: 'OK',
          onPressed: () {
            ScaffoldMessenger.of(context).hideCurrentSnackBar();
          },
        ),
      ),
    );
  }

  Future<bool> _showInternetDialog() async {
    final result = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        icon: const Icon(Icons.wifi, size: 32),
        title: const Text('Go online?'),
        content: const Text(
          'This query needs the internet to find the best answer. '
          'Allow Ocula to search the web just this once?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Stay Offline'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Allow'),
          ),
        ],
      ),
    );
    return result ?? false;
  }

  Future<bool> _showConnectivityDialog() async {
    final openSettings = await showDialog<bool>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        icon: const Icon(Icons.wifi_off, size: 32),
        title: const Text('Internet is off'),
        content: const Text(
          'This request needs internet, but Wi-Fi/mobile data appears to be off. '
          'Allow Ocula to open Settings so you can enable connectivity?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Not now'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Open Settings'),
          ),
        ],
      ),
    );
    if (openSettings == true) {
      if (!Platform.isMacOS) await openAppSettings();
      return true;
    }
    return false;
  }

  static const _cameraChannel = MethodChannel('com.finailabz.ai.ocula/camera');

  /// Open the native macOS camera capture panel and return the saved path.
  Future<void> _takeMacOSPhoto() async {
    try {
      final path = await _cameraChannel.invokeMethod<String>('capturePhoto');
      if (path != null && mounted) {
        setState(() => _attachedImage = File(path));
      }
    } catch (e) {
      if (mounted) _showSnackbar('Camera error: $e');
    }
  }

  void _showAttachmentPicker() {
    final colors = Theme.of(context).colorScheme;

    void pick(BuildContext ctx, VoidCallback action) {
      Navigator.pop(ctx);
      action();
    }

    final items = [
      (
        Icons.camera_alt_rounded,
        colors.primary,
        'Camera',
        () => _pickImage(fromCamera: true),
      ),
      (
        Icons.photo_library_rounded,
        Colors.green,
        'Photos',
        () => _pickImage(fromCamera: false),
      ),
      (Icons.videocam_rounded, Colors.deepOrange, 'Video', () => _pickVideo()),
      (
        Icons.description_rounded,
        Colors.blueGrey,
        'Document',
        () => _pickDocument(),
      ),
    ];

    showModalBottomSheet(
      context: context,
      backgroundColor: colors.surfaceContainerHighest,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Attach',
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: colors.onSurface.withAlpha(140),
                  letterSpacing: 0.5,
                ),
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.start,
                children: items.map((item) {
                  final (icon, color, label, action) = item;
                  return Padding(
                    padding: const EdgeInsets.only(right: 20),
                    child: GestureDetector(
                      onTap: () => pick(ctx, action),
                      behavior: HitTestBehavior.opaque,
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Container(
                            width: 60,
                            height: 60,
                            decoration: BoxDecoration(
                              color: color.withAlpha(30),
                              borderRadius: BorderRadius.circular(16),
                              border: Border.all(
                                color: color.withAlpha(60),
                                width: 1,
                              ),
                            ),
                            child: Icon(icon, color: color, size: 28),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            label,
                            style: TextStyle(
                              fontSize: 12,
                              color: colors.onSurface.withAlpha(180),
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                }).toList(),
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }

  /// Empty-state shortcuts: run a ready query, or narrow the scope and
  /// prefill the input so the user only types the name / topic.
  void _applyQuickStart(_QuickStart q) {
    if (q.sendNow) {
      _send(q.text);
      return;
    }
    setState(() => _retrievalScope = q.scope);
    _textController.text = q.text;
    _textController.selection = TextSelection.collapsed(offset: q.text.length);
    _inputFocus.requestFocus();
  }

  void _showScopeFilter() {
    final options = [
      (
        RetrievalScope.all,
        Icons.all_inclusive,
        'All sources',
        'Search everything',
      ),
      (
        RetrievalScope.contacts,
        Icons.contacts_outlined,
        'Contacts',
        'People & phone numbers',
      ),
      (
        RetrievalScope.calendar,
        Icons.calendar_today_outlined,
        'Calendar',
        'Events & schedule',
      ),
      (RetrievalScope.email, Icons.email_outlined, 'Email', 'Messages & inbox'),
      (
        RetrievalScope.docs,
        Icons.description_outlined,
        'Documents',
        'Files, PDFs & notes',
      ),
      (
        RetrievalScope.images,
        Icons.photo_library_outlined,
        'Photos',
        'Images on your device',
      ),
    ];
    showModalBottomSheet(
      context: context,
      backgroundColor: Theme.of(context).colorScheme.surfaceContainerHighest,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) {
        final colors = Theme.of(ctx).colorScheme;
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 8, 20, 4),
                  child: Text(
                    'Ask about',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: colors.onSurface.withAlpha(140),
                      letterSpacing: 0.5,
                    ),
                  ),
                ),
                ...options.map((opt) {
                  final (scope, icon, label, subtitle) = opt;
                  final selected = _retrievalScope == scope;
                  return ListTile(
                    leading: Icon(
                      icon,
                      color: selected
                          ? colors.primary
                          : colors.onSurface.withAlpha(180),
                    ),
                    title: Text(
                      label,
                      style: TextStyle(
                        fontWeight: selected
                            ? FontWeight.w600
                            : FontWeight.normal,
                      ),
                    ),
                    subtitle: Text(
                      subtitle,
                      style: TextStyle(
                        fontSize: 12,
                        color: colors.onSurface.withAlpha(120),
                      ),
                    ),
                    trailing: selected
                        ? Icon(
                            Icons.check_circle,
                            color: colors.primary,
                            size: 20,
                          )
                        : null,
                    tileColor: selected
                        ? colors.primary.withAlpha(18)
                        : Colors.transparent,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 2,
                    ),
                    onTap: () {
                      Navigator.pop(ctx);
                      setState(() => _retrievalScope = scope);
                    },
                  );
                }),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _pickImage({bool fromCamera = false}) async {
    if (fromCamera) {
      if (_isDesktop) {
        await _takeMacOSPhoto();
        return;
      }
      final picked = await ImagePicker().pickImage(
        source: ImageSource.camera,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );
      if (picked != null && mounted)
        setState(() => _attachedImage = File(picked.path));
    } else {
      if (_isDesktop) {
        await _pickImageDesktop();
        return;
      }
      final picked = await ImagePicker().pickImage(
        source: ImageSource.gallery,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );
      if (picked != null && mounted)
        setState(() => _attachedImage = File(picked.path));
    }
  }

  Future<void> _pickImageDesktop() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        allowMultiple: false,
      );
      if (result != null && result.files.single.path != null) {
        setState(() => _attachedImage = File(result.files.single.path!));
      }
    } catch (e) {
      if (mounted) _showSnackbar('Could not pick image: $e');
    }
  }

  Future<void> _pickVideo() async {
    try {
      if (_isDesktop) {
        // Desktop: pick via file picker
        final result = await FilePicker.platform.pickFiles(
          type: FileType.video,
          allowMultiple: false,
        );
        if (result != null && result.files.single.path != null) {
          final name = result.files.single.name;
          setState(() {
            _attachedDocument = File(result.files.single.path!);
            _attachedDocName = name;
          });
        }
        return;
      }
      final picked = await ImagePicker().pickVideo(source: ImageSource.gallery);
      if (picked != null && mounted) {
        setState(() {
          _attachedDocument = File(picked.path);
          _attachedDocName = picked.name;
        });
      }
    } catch (e) {
      if (mounted) _showSnackbar('Could not pick video: $e');
    }
  }

  Future<void> _pickDocument() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: [
          // Office documents
          'pdf', 'docx', 'pptx', 'xlsx',
          // Plain text / markup
          'txt', 'md', 'csv', 'json', 'xml', 'html', 'htm',
          // Code
          'py', 'js', 'ts', 'dart', 'java', 'swift', 'yaml',
        ],
      );
      if (result != null && result.files.single.path != null) {
        final path = result.files.single.path!;
        final name = result.files.single.name;
        setState(() {
          _attachedDocument = File(path);
          _attachedDocName = name;
        });

        // Index the file immediately so RAG can use it
        final indexed = await Indexer().indexFilePath(path);
        if (indexed) {
          _showSnackbar('Indexed "$name" — you can ask about it now');
        } else {
          final ext = name.contains('.')
              ? name.split('.').last.toLowerCase()
              : '';
          final msg = ext == 'pdf'
              ? '"$name" appears to be a scanned/image PDF — text could not be extracted'
              : 'Could not extract text from "$name"';
          _showSnackbar(msg);
        }
      }
    } catch (e) {
      _showSnackbar('Could not pick file: $e');
    }
  }

  void _toggleOrb() {
    setState(() => _orbExpanded = !_orbExpanded);
    if (_orbExpanded) {
      _orbSizeController.forward();
    } else {
      _orbSizeController.reverse();
    }
  }

  Future<void> _stopEverything() async {
    // Signal to _send() that any in-flight result should be discarded
    _stopRequested = true;

    // Fire-and-forget: stop generation + TTS + listening in parallel.
    // Don't await TTS stop — it may block until the platform confirms,
    // and we want the UI to reset instantly.
    _orchestrator.stop();
    _speech.stopSpeaking();
    if (_isListening) {
      _speech.stopListening();
    }
    if (mounted) {
      setState(() {
        _isThinking = false;
        _isListening = false;
        _isSpeaking = false;
        _orbState = OrbState.idle;
      });
    }
  }

  Future<void> _send(
    String text, {
    String? queryOverride,
    String? displayTextOverride,
    AITier? forcedTier,
  }) async {
    await _ensureLiteActivatedIfReady();

    final image = _attachedImage;
    final doc = _attachedDocument;
    final docName = _attachedDocName;

    // If image attached with no text, provide a default prompt
    if (text.trim().isEmpty && image != null) {
      text = 'Describe this image in detail. What do you see?';
    }
    // If doc attached with no text, provide a default prompt
    if (text.trim().isEmpty && doc != null) {
      text = 'Summarize this document and highlight the key points.';
    }
    if (text.trim().isEmpty) return;
    _textController.clear();

    // If already generating or speaking, stop that first
    if (_isThinking || _isSpeaking) {
      await _stopEverything();
    }

    // If a document is attached, prepend its content to the query
    String queryText = queryOverride ?? text;
    var runScope = _retrievalScope;
    if (doc != null && docName != null) {
      runScope = RetrievalScope.docs;
      // Use LocalData.readFileContent so PDFs and Office docs are properly
      // extracted (Syncfusion text extraction) rather than read as raw bytes.
      final content = await LocalData().readFileContent(doc.path);
      if (content != null && content.isNotEmpty) {
        final truncated = content.length > 4000
            ? content.substring(0, 4000)
            : content;
        queryText =
            '[Attached document: $docName]\n'
            '--- DOCUMENT CONTENT ---\n$truncated\n--- END DOCUMENT ---\n\n'
            'User request: $text';
      } else {
        // Text extraction returned nothing — likely a scanned/image-only PDF.
        // Tell the user directly rather than silently falling through to RAG.
        final ext = docName.contains('.')
            ? docName.split('.').last.toLowerCase()
            : '';
        final reason = ext == 'pdf'
            ? 'This PDF appears to be scanned or image-based — its text could '
                  'not be extracted. Only searchable (text-layer) PDFs can be read.'
            : 'The text in "$docName" could not be extracted.';
        queryText =
            '[File attached: $docName — content unavailable]\n'
            '$reason\n'
            'Please tell the user you cannot read this file\'s content and '
            'explain why. Do NOT invent or guess any document contents.\n'
            'User request: $text';
      }
    }

    // Display text — show the actual user text, not the augmented query
    final displayText =
        displayTextOverride ??
        ((image != null &&
                text == 'Describe this image in detail. What do you see?')
            ? 'Analyze this image'
            : text);

    // Check for action intent before going to the RAG pipeline.
    // Only detect on plain text queries (no image, no doc attachment).
    final actionReq = (image == null && doc == null)
        ? ActionService.detect(queryText)
        : null;

    // Reset stop flag — this new query is intentional
    _stopRequested = false;

    // Capture the most recent exchange (before this turn is appended) so the
    // model can see what it just asked and avoid repeating itself.
    String? previousUserMessage;
    String? previousAssistantMessage;
    for (int i = _messages.length - 1; i >= 0; i--) {
      if (!_messages[i].isUser && _messages[i].actionRequest == null) {
        previousAssistantMessage = _messages[i].text;
        for (int j = i - 1; j >= 0; j--) {
          if (_messages[j].isUser) {
            previousUserMessage = _messages[j].text;
            break;
          }
        }
        break;
      }
    }

    setState(() {
      _messages.add(_Message(text: displayText, isUser: true, image: image));
      _attachedImage = null;
      _attachedDocument = null;
      _attachedDocName = null;
      _isThinking = actionReq == null;
      _orbState = actionReq == null ? OrbState.thinking : OrbState.idle;
      _orbExpanded = actionReq == null;
    });

    if (actionReq != null) {
      // Add the action card immediately — contact resolution happens inside ActionCard.
      if (mounted) {
        setState(() {
          _messages.add(
            _Message(text: '', isUser: false, actionRequest: actionReq),
          );
        });
        _scrollToBottom();
      }
      return;
    }

    _orbSizeController.forward();
    _scrollToBottom();

    try {
      final result = await _orchestrator
          .run(
            queryText,
            hasImage: image != null,
            imagePath: image?.path,
            retrievalScope: runScope,
            forcedTier: forcedTier,
            sessionId: _sessionId,
            previousUserMessage: previousUserMessage,
            previousAssistantMessage: previousAssistantMessage,
          )
          .timeout(
            const Duration(minutes: 2),
            onTimeout: () => OrchestratorResult(
              'Sorry, I took too long to respond. Please try again.',
            ),
          );

      if (!mounted) return;

      // Discard result if user hit stop while we were generating
      if (_stopRequested) {
        _stopRequested = false;
        return;
      }

      // Empty response = cancelled by user (stop or new query)
      if (result.response.isEmpty) {
        setState(() {
          _isThinking = false;
          _orbState = OrbState.idle;
        });
        return;
      }
      // On-device preference learning — fire and forget.
      UserProfile()
          .recordInteraction(
            userQuery: queryText,
            assistantResponse: result.response,
            isFollowUp: previousUserMessage != null,
          )
          .catchError((_) {});
      setState(() {
        _messages.add(
          _Message(
            text: result.response,
            isUser: false,
            linkedAssets: result.linkedAssets,
          ),
        );
        _isThinking = false;
        _orbState = OrbState.speaking;
        _isSpeaking = true;
      });
      _scrollToBottom();

      await _speech.speak(result.response);
      // Only reset speaking state if we weren't stopped/interrupted.
      // If _stopRequested, _stopEverything() already handled the UI reset.
      if (mounted && !_stopRequested) {
        setState(() {
          _orbState = OrbState.idle;
          _isSpeaking = false;
          if (_messages.isNotEmpty) {
            _orbExpanded = false;
            _orbSizeController.reverse();
          }
        });
      }
    } on ModelNotReadyException catch (e) {
      if (!mounted) return;
      _showSnackbar(e.toString());
      setState(() {
        _isThinking = false;
        _orbState = OrbState.idle;
      });
    } catch (e) {
      if (!mounted) return;
      _showSnackbar('An error occurred: $e');
      setState(() {
        _isThinking = false;
        _orbState = OrbState.idle;
      });
    }
  }

  void _toggleVoice() {
    if (_isListening) {
      _speech.stopListening();
      setState(() {
        _isListening = false;
        _orbState = OrbState.idle;
      });
    } else {
      setState(() {
        _isListening = true;
        _orbState = OrbState.listening;
        _orbExpanded = true;
      });
      _orbSizeController.forward();
      try {
        _speech.startListening(
          onResult: (text) {
            _textController.text = text;
          },
          onAIResponse: (response) {
            setState(() {
              _messages.add(_Message(text: _textController.text, isUser: true));
              _messages.add(_Message(text: response, isUser: false));
              _isListening = false;
              _orbState = OrbState.idle;
              _textController.clear();
              _orbExpanded = false;
            });
            _orbSizeController.reverse();
            _scrollToBottom();
          },
          onError: () {
            _showSnackbar(
              'Could not start listening. Please check microphone permissions.',
            );
            setState(() {
              _isListening = false;
              _orbState = OrbState.idle;
            });
          },
        );
      } on ModelNotReadyException catch (e) {
        _showSnackbar(e.toString());
        setState(() {
          _isListening = false;
          _orbState = OrbState.idle;
        });
      } catch (e) {
        _showSnackbar('An error occurred: $e');
        setState(() {
          _isListening = false;
          _orbState = OrbState.idle;
        });
      }
    }
  }

  void _copyToClipboard(String text) {
    Clipboard.setData(ClipboardData(text: text));
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Text('Copied to clipboard'),
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  void dispose() {
    Indexer().stopBackgroundIndexing();
    _shareReceiver.dispose();
    _ai.dispose();
    _textController.dispose();
    _inputFocus.dispose();
    _scrollController.dispose();
    _orbSizeController.dispose();
    _downloadProgressSubscription?.cancel();
    _featureReadySubscription?.cancel();
    _tierChangeSubscription?.cancel();
    _freeModelStatusSubscription?.cancel();
    _stepSubscription?.cancel();
    _orchestrator.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final screenHeight = MediaQuery.of(context).size.height;
    final hasMessages = _messages.isNotEmpty || _isThinking;

    const expandedSize = 160.0;
    const miniSize = 56.0;

    return PopScope(
      // On Android, warn before discarding an active conversation.
      canPop: _messages.isEmpty,
      onPopInvokedWithResult: (didPop, _) async {
        if (didPop) return;
        final confirmed = await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: const Text('Leave conversation?'),
            content: const Text(
              'Your current chat will be lost. Start a new session from the home screen.',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx, false),
                child: const Text('Stay'),
              ),
              FilledButton(
                onPressed: () => Navigator.pop(ctx, true),
                child: const Text('Leave'),
              ),
            ],
          ),
        );
        if ((confirmed ?? false) && context.mounted) {
          Navigator.of(context).pop();
        }
      },
      child: Scaffold(
        resizeToAvoidBottomInset: true,
        body: Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [colors.surface, const Color(0xFF0B0C11)],
            ),
          ),
          child: SafeArea(
            child: Stack(
              children: [
                // ── Layer 1: Chat transcript ──
                Column(
                  children: [
                    // Top bar
                    Padding(
                      padding: const EdgeInsets.fromLTRB(16, 4, 4, 4),
                      child: Row(
                        children: [
                          ShaderMask(
                            shaderCallback: (bounds) => const LinearGradient(
                              colors: [Color(0xFF6C5CE7), Color(0xFF00CEC9)],
                            ).createShader(bounds),
                            child: const Text(
                              'Ocula',
                              style: TextStyle(
                                fontSize: 22,
                                fontWeight: FontWeight.w700,
                                letterSpacing: -0.5,
                                color: Colors.white,
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          // Ready indicator dot — green when model loaded, amber while loading
                          AnimatedContainer(
                            duration: const Duration(milliseconds: 400),
                            width: 7,
                            height: 7,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: _ai.isModelLoaded
                                  ? const Color(0xFF00E676)
                                  : const Color(0xFFFFB300),
                              boxShadow: [
                                BoxShadow(
                                  color:
                                      (_ai.isModelLoaded
                                              ? const Color(0xFF00E676)
                                              : const Color(0xFFFFB300))
                                          .withAlpha(120),
                                  blurRadius: 6,
                                  spreadRadius: 1,
                                ),
                              ],
                            ),
                          ),
                          const Spacer(),
                          // Source filter — tap to narrow RAG scope
                          Stack(
                            clipBehavior: Clip.none,
                            children: [
                              IconButton(
                                key: _cameraButtonKey,
                                icon: const Icon(
                                  Icons.filter_list_rounded,
                                  size: 20,
                                ),
                                tooltip: 'Filter source',
                                onPressed: _showScopeFilter,
                              ),
                              if (_retrievalScope != RetrievalScope.all)
                                Positioned(
                                  right: 6,
                                  top: 6,
                                  child: Container(
                                    width: 8,
                                    height: 8,
                                    decoration: BoxDecoration(
                                      color: Theme.of(
                                        context,
                                      ).colorScheme.primary,
                                      shape: BoxShape.circle,
                                    ),
                                  ),
                                ),
                            ],
                          ),
                          // Export button — visible in the top bar only on
                          // tablet/desktop where there is room; phones access it
                          // via the overflow menu below.
                          if (_isTabletOrDesktop &&
                              _messages.any((m) => !m.isUser))
                            Builder(
                              builder: (ctx) => IconButton(
                                icon: const Icon(Icons.ios_share, size: 20),
                                tooltip: 'Export',
                                onPressed: () {
                                  final lastResponse = _messages
                                      .lastWhere((m) => !m.isUser)
                                      .text;
                                  final box =
                                      ctx.findRenderObject() as RenderBox?;
                                  final origin = box != null
                                      ? box.localToGlobal(Offset.zero) &
                                            box.size
                                      : null;
                                  _export.exportAndShare(
                                    lastResponse,
                                    origin: origin,
                                  );
                                },
                              ),
                            ),
                          IconButton(
                            icon: const Icon(
                              Icons.add_comment_outlined,
                              size: 20,
                            ),
                            tooltip: 'New chat',
                            onPressed: _startNewSession,
                          ),
                          IconButton(
                            icon: const Icon(Icons.settings_outlined, size: 20),
                            tooltip: 'Settings',
                            onPressed: () {
                              Navigator.of(context).push(
                                MaterialPageRoute(
                                  builder: (_) => SettingsScreen(
                                    speech: _speech,
                                    onRequestHelpTour: _startHelpTour,
                                  ),
                                ),
                              );
                            },
                          ),
                          // Overflow menu: history + help + export (on phone)
                          Builder(
                            builder: (ctx) => PopupMenuButton<String>(
                              icon: const Icon(Icons.more_vert, size: 20),
                              tooltip: 'More',
                              onSelected: (value) {
                                if (value == 'record') {
                                  Navigator.of(context).pushNamed('/recorder');
                                } else if (value == 'history') {
                                  _openSessionHistory(ctx);
                                } else if (value == 'help') {
                                  _startHelpTour();
                                } else if (value == 'export') {
                                  final lastResponse = _messages
                                      .lastWhere((m) => !m.isUser)
                                      .text;
                                  final box =
                                      ctx.findRenderObject() as RenderBox?;
                                  final origin = box != null
                                      ? box.localToGlobal(Offset.zero) &
                                            box.size
                                      : null;
                                  _export.exportAndShare(
                                    lastResponse,
                                    origin: origin,
                                  );
                                }
                              },
                              itemBuilder: (_) => [
                                const PopupMenuItem(
                                  value: 'record',
                                  child: Row(
                                    children: [
                                      Icon(Icons.mic_none_rounded, size: 18),
                                      SizedBox(width: 10),
                                      Text('Record meeting'),
                                    ],
                                  ),
                                ),
                                const PopupMenuItem(
                                  value: 'history',
                                  child: Row(
                                    children: [
                                      Icon(Icons.history_outlined, size: 18),
                                      SizedBox(width: 10),
                                      Text('Chat history'),
                                    ],
                                  ),
                                ),
                                const PopupMenuItem(
                                  value: 'help',
                                  child: Row(
                                    children: [
                                      Icon(Icons.help_outline, size: 18),
                                      SizedBox(width: 10),
                                      Text('Help tour'),
                                    ],
                                  ),
                                ),
                                // Export last response — shown on phone only
                                // (tablet/desktop have the dedicated icon button)
                                if (!_isTabletOrDesktop &&
                                    _messages.any((m) => !m.isUser))
                                  const PopupMenuItem(
                                    value: 'export',
                                    child: Row(
                                      children: [
                                        Icon(Icons.ios_share, size: 18),
                                        SizedBox(width: 10),
                                        Text('Export response'),
                                      ],
                                    ),
                                  ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),

                    // Show failure banner always (needs user action).
                    // Show download progress for free-tier (first install) and
                    // for Plus/Pro when explicitly downloading from the picker
                    // or auto-downloading on tablet/desktop.
                    if (_freeModelFailed ||
                        _backgroundDownloadFailed != null ||
                        (_activeDownloads.isNotEmpty &&
                            (_ai.activeTier == null ||
                                _activeDownloads.keys.any(
                                  (k) => OculaModelManager.models.any(
                                    (m) =>
                                        m.fileName == k &&
                                        (m.tier == AITier.plus ||
                                            m.tier == AITier.pro),
                                  ),
                                ))))
                      _buildDownloadBanner(colors),
                    if (_showDataSourcesBanner) _buildDataSourcesBanner(colors),

                    // Chat messages
                    Expanded(
                      child: Container(
                        // Key used by HelpTour to find the correct bounds of the
                        // chat area. Must be on a widget that always renders with
                        // the correct dimensions (Container fills Expanded space
                        // even when the ListView is absent).
                        key: _chatListKey,
                        child: hasMessages
                            ? ListView.builder(
                                controller: _scrollController,
                                padding: EdgeInsets.only(
                                  left: 16,
                                  right: 16,
                                  top: _orbExpanded
                                      ? expandedSize + 40
                                      : miniSize + 20,
                                  bottom: 8,
                                ),
                                itemCount:
                                    _messages.length + (_isThinking ? 1 : 0),
                                itemBuilder: (context, index) {
                                  if (_isThinking &&
                                      index == _messages.length) {
                                    return _ThinkingBubble(
                                      stepLabel: _currentStepLabel,
                                    );
                                  }
                                  final msg = _messages[index];
                                  // Find the user query that preceded this AI response
                                  final precedingQuery =
                                      (!msg.isUser && index > 0)
                                      ? _messages[index - 1].text
                                      : null;
                                  return _MessageBubble(
                                    message: msg,
                                    onCopy: () => _copyToClipboard(msg.text),
                                    onSearchWeb:
                                        (!msg.isUser && precedingQuery != null)
                                        ? () => _send(
                                            'search online for: $precedingQuery',
                                          )
                                        : null,
                                  );
                                },
                              )
                            : const SizedBox.shrink(),
                      ),
                    ),

                    // ── Quick starts (empty state) ──
                    if (!hasMessages &&
                        _attachedImage == null &&
                        _attachedDocument == null)
                      _QuickStarts(onPick: _applyQuickStart),

                    // ── Attached image preview ──
                    if (_attachedImage != null)
                      Container(
                        padding: const EdgeInsets.fromLTRB(16, 6, 16, 0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                ClipRRect(
                                  borderRadius: BorderRadius.circular(10),
                                  child: Image.file(
                                    _attachedImage!,
                                    width: 56,
                                    height: 56,
                                    fit: BoxFit.cover,
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: Text(
                                    'Image attached',
                                    style: TextStyle(
                                      color: colors.onSurface.withAlpha(150),
                                      fontSize: 13,
                                    ),
                                  ),
                                ),
                                IconButton(
                                  icon: const Icon(Icons.close, size: 18),
                                  onPressed: () =>
                                      setState(() => _attachedImage = null),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),

                    // ── Attached document preview ──
                    if (_attachedDocument != null)
                      Container(
                        padding: const EdgeInsets.fromLTRB(16, 6, 16, 0),
                        child: Row(
                          children: [
                            Container(
                              width: 40,
                              height: 40,
                              decoration: BoxDecoration(
                                color: colors.primary.withAlpha(30),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Icon(
                                Icons.description,
                                size: 22,
                                color: colors.primary,
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                _attachedDocName ?? 'Document attached',
                                style: TextStyle(
                                  color: colors.onSurface.withAlpha(150),
                                  fontSize: 13,
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            IconButton(
                              icon: const Icon(Icons.close, size: 18),
                              onPressed: () => setState(() {
                                _attachedDocument = null;
                                _attachedDocName = null;
                              }),
                            ),
                          ],
                        ),
                      ),

                    // ── Input bar ──
                    SafeArea(
                      top: false,
                      child: Container(
                        key: _inputBarKey,
                        margin: const EdgeInsets.fromLTRB(10, 6, 10, 8),
                        padding: const EdgeInsets.fromLTRB(4, 4, 6, 4),
                        decoration: BoxDecoration(
                          color: colors.surfaceContainerHigh,
                          borderRadius: BorderRadius.circular(28),
                          border: Border.all(color: colors.outlineVariant),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withAlpha(70),
                              blurRadius: 18,
                              offset: const Offset(0, 6),
                            ),
                          ],
                        ),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            // Attach — camera, photo/video, document
                            IconButton(
                              icon: const Icon(
                                Icons.add_circle_outline,
                                size: 24,
                              ),
                              tooltip: 'Attach file, photo or video',
                              padding: const EdgeInsets.symmetric(
                                horizontal: 4,
                              ),
                              onPressed: _showAttachmentPicker,
                            ),
                            // Text input
                            Expanded(
                              child: Padding(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 4,
                                ),
                                child: TextField(
                                  controller: _textController,
                                  focusNode: _inputFocus,
                                  style: const TextStyle(fontSize: 15),
                                  maxLines: 5,
                                  minLines: 1,
                                  keyboardType: TextInputType.multiline,
                                  textCapitalization:
                                      TextCapitalization.sentences,
                                  textInputAction: TextInputAction.newline,
                                  autocorrect: true,
                                  enableSuggestions: true,
                                  decoration: InputDecoration(
                                    hintText: 'Ask Ocula…',
                                    hintStyle: TextStyle(
                                      color: colors.onSurface.withAlpha(80),
                                    ),
                                    border: InputBorder.none,
                                    contentPadding: const EdgeInsets.symmetric(
                                      vertical: 12,
                                    ),
                                  ),
                                ),
                              ),
                            ),
                            // Send / Stop / Mic — context-dependent action
                            _isThinking || _isSpeaking
                                ? _InputAction(
                                    icon: Icons.stop_circle,
                                    label: 'Stop',
                                    color: const Color(0xFFFF7675),
                                    onTap: _stopEverything,
                                  )
                                : _isListening
                                ? _InputAction(
                                    icon: Icons.stop_circle,
                                    label: 'Stop',
                                    color: const Color(0xFFFF7675),
                                    onTap: _toggleVoice,
                                  )
                                : _textController.text.isNotEmpty
                                ? _SendButton(
                                    onTap: () => _send(_textController.text),
                                    color: colors.primary,
                                  )
                                : _InputAction(
                                    icon: Icons.mic,
                                    label: 'Voice',
                                    color: colors.primary,
                                    onTap: _toggleVoice,
                                  ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),

                // ── Layer 2: Floating 3D Orb overlay ──
                AnimatedPositioned(
                  duration: const Duration(milliseconds: 400),
                  curve: Curves.easeOutCubic,
                  top: _orbExpanded
                      ? (hasMessages ? 60 : screenHeight * 0.15)
                      : 60,
                  left: 0,
                  right: _orbExpanded ? 0 : null,
                  child: GestureDetector(
                    key: _orbKey,
                    onTap: _orbExpanded && hasMessages
                        ? _toggleOrb
                        : _toggleVoice,
                    onLongPress: _showAvatarStylePicker,
                    child: AnimatedAlign(
                      duration: const Duration(milliseconds: 400),
                      curve: Curves.easeOutCubic,
                      alignment: _orbExpanded
                          ? Alignment.center
                          : Alignment.centerLeft,
                      child: Padding(
                        padding: EdgeInsets.only(left: _orbExpanded ? 0 : 16),
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 400),
                          curve: Curves.easeOutCubic,
                          width: _orbExpanded
                              ? expandedSize * 1.5
                              : miniSize * 1.5,
                          height: _orbExpanded
                              ? expandedSize * 1.5
                              : miniSize * 1.5,
                          child: OculaOrb(
                            state: _orbState,
                            size: _orbExpanded ? expandedSize : miniSize,
                            avatarStyle: _avatarStyle,
                            customImagePath: _customAvatarPath,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),

                // ── Layer 3: "Show chat" hint ──
                if (_orbExpanded && hasMessages)
                  Positioned(
                    top: expandedSize + 110,
                    left: 0,
                    right: 0,
                    child: Center(
                      child: GestureDetector(
                        onTap: _toggleOrb,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 18,
                            vertical: 8,
                          ),
                          decoration: BoxDecoration(
                            color: colors.surfaceContainerHighest.withAlpha(
                              220,
                            ),
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(
                              color: colors.outline.withAlpha(30),
                            ),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                Icons.keyboard_arrow_down,
                                size: 18,
                                color: colors.onSurface.withAlpha(150),
                              ),
                              const SizedBox(width: 4),
                              Text(
                                'Show chat',
                                style: TextStyle(
                                  fontSize: 13,
                                  color: colors.onSurface.withAlpha(150),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),

                // ── Layer 4: Help Tour overlay ──
                if (_showingHelpTour)
                  HelpTour(
                    steps: [
                      HelpStep(
                        targetKey: _orbKey,
                        title: 'Talk to Ocula',
                        description:
                            'Tap the orb to start speaking. Ocula listens and responds aloud.',
                      ),
                      HelpStep(
                        targetKey: _chatListKey,
                        title: 'Chat Window',
                        description:
                            'Your conversation appears here. Ocula runs fully on-device — no cloud.',
                      ),
                      HelpStep(
                        targetKey: _inputBarKey,
                        title: 'Type a Question',
                        description:
                            'Type anything here. Tap + to attach photos, documents, or files.',
                      ),
                      HelpStep(
                        targetKey: _cameraButtonKey,
                        title: 'Attach Assets',
                        description:
                            'Tap + to attach a photo, video, PDF, Word or spreadsheet — Ocula reads and analyzes it on-device.',
                      ),
                    ],
                    onComplete: () {
                      setState(() => _showingHelpTour = false);
                      SharedPreferences.getInstance().then(
                        (p) => p.setBool('show_help_tour', false),
                      );
                    },
                  ),
              ],
            ),
          ),
        ),
      ), // Scaffold
    ); // PopScope
  }
}

/// A single chat message.
class _Message {
  final String text;
  final bool isUser;
  final File? image;
  final List<LinkedAsset> linkedAssets;
  final ActionRequest? actionRequest;

  _Message({
    required this.text,
    required this.isUser,
    this.image,
    this.linkedAssets = const [],
    this.actionRequest,
  });
}

/// Chat bubble with long-press copy and selectable text.
class _MessageBubble extends StatelessWidget {
  final _Message message;
  final VoidCallback? onCopy;

  /// Called when the user taps "Search Web" on a no-data response.
  final VoidCallback? onSearchWeb;

  const _MessageBubble({required this.message, this.onCopy, this.onSearchWeb});

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final isUser = message.isUser;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: GestureDetector(
        onLongPress: onCopy,
        child: Container(
          margin: const EdgeInsets.only(bottom: 8),
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
          constraints: BoxConstraints(
            maxWidth:
                MediaQuery.of(context).size.width * (isUser ? 0.78 : 0.88),
          ),
          decoration: BoxDecoration(
            gradient: isUser
                ? const LinearGradient(
                    colors: [Color(0xFF8B7CF6), Color(0xFF6D5BE8)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  )
                : null,
            color: isUser ? null : colors.surfaceContainer,
            borderRadius: BorderRadius.circular(18).copyWith(
              bottomRight: isUser ? const Radius.circular(6) : null,
              bottomLeft: !isUser ? const Radius.circular(6) : null,
            ),
            border: isUser ? null : Border.all(color: colors.outlineVariant),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              // Action confirmation card replaces the normal text body
              if (!isUser && message.actionRequest != null) ...[
                ActionCard(request: message.actionRequest!),
              ] else ...[
                if (message.image != null) ...[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Image.file(
                      message.image!,
                      width: 200,
                      fit: BoxFit.cover,
                    ),
                  ),
                  const SizedBox(height: 8),
                ],
                SelectableText(
                  message.text,
                  style: TextStyle(
                    color: isUser ? colors.onPrimary : colors.onSurface,
                    fontSize: 14.5,
                    height: 1.4,
                  ),
                ),
                // Structured result cards (contacts, photos, documents, events)
                if (!isUser && message.linkedAssets.isNotEmpty) ...[
                  const SizedBox(height: 4),
                  ResultCards(assets: message.linkedAssets),
                ],
                // "Search Web" offer — shown when local data was insufficient
                if (!isUser &&
                    onSearchWeb != null &&
                    _isNoDataResponse(message.text)) ...[
                  const SizedBox(height: 10),
                  GestureDetector(
                    onTap: onSearchWeb,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 6,
                      ),
                      decoration: BoxDecoration(
                        color: colors.primaryContainer.withAlpha(180),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: colors.primary.withAlpha(60)),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.language_outlined,
                            size: 14,
                            color: colors.primary,
                          ),
                          const SizedBox(width: 6),
                          Text(
                            'Search the web for more info',
                            style: TextStyle(
                              fontSize: 12,
                              color: colors.primary,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ], // end else (non-action messages)
            ],
          ),
        ),
      ),
    );
  }
}

/// Returns true if the AI response indicates it couldn't find the answer
/// in local data — used to decide whether to offer a web search.
bool _isNoDataResponse(String text) {
  final lower = text.toLowerCase();
  return lower.contains("don't have that") ||
      lower.contains("don't have that in your data") ||
      lower.contains("not in your data") ||
      lower.contains("couldn't find") ||
      lower.contains("could not find") ||
      lower.contains("no information") ||
      lower.contains("not available in your") ||
      lower.contains("i don't have") ||
      lower.contains("i do not have") ||
      lower.contains("not found in your");
}

class _QuickStart {
  final IconData icon;
  final String label;
  final String text;
  final RetrievalScope scope;
  final bool sendNow;
  const _QuickStart(
    this.icon,
    this.label,
    this.text,
    this.scope, {
    this.sendNow = false,
  });
}

/// Horizontal row of shortcut chips shown before the first message.
class _QuickStarts extends StatelessWidget {
  final ValueChanged<_QuickStart> onPick;
  const _QuickStarts({required this.onPick});

  static const _items = [
    _QuickStart(
      Icons.wb_sunny_outlined,
      'My day',
      "What's on my schedule today?",
      RetrievalScope.calendar,
      sendNow: true,
    ),
    _QuickStart(
      Icons.person_search_outlined,
      'Find a contact',
      'Find ',
      RetrievalScope.contacts,
    ),
    _QuickStart(
      Icons.photo_library_outlined,
      'Photos',
      'Show photos of ',
      RetrievalScope.images,
    ),
    _QuickStart(
      Icons.description_outlined,
      'Documents',
      'Find document ',
      RetrievalScope.docs,
    ),
  ];

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return SizedBox(
      height: 44,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 12),
        itemCount: _items.length,
        separatorBuilder: (_, _) => const SizedBox(width: 8),
        itemBuilder: (_, i) {
          final q = _items[i];
          return ActionChip(
            avatar: Icon(q.icon, size: 16, color: colors.primary),
            label: Text(q.label),
            labelStyle: TextStyle(fontSize: 13, color: colors.onSurface),
            backgroundColor: colors.surfaceContainerHigh,
            side: BorderSide(color: colors.outlineVariant),
            shape: const StadiumBorder(),
            onPressed: () => onPick(q),
          );
        },
      ),
    );
  }
}

/// Animated "thinking" indicator with pulsing dots and optional step label.
class _ThinkingBubble extends StatefulWidget {
  final String? stepLabel;
  const _ThinkingBubble({this.stepLabel});

  @override
  State<_ThinkingBubble> createState() => _ThinkingBubbleState();
}

class _ThinkingBubbleState extends State<_ThinkingBubble>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        decoration: BoxDecoration(
          color: colors.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(
            20,
          ).copyWith(bottomLeft: const Radius.circular(4)),
          border: Border.all(color: colors.outline.withAlpha(25)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            AnimatedBuilder(
              animation: _controller,
              builder: (context, _) {
                return Row(
                  mainAxisSize: MainAxisSize.min,
                  children: List.generate(3, (i) {
                    final delay = i * 0.2;
                    final t = ((_controller.value - delay) % 1.0).clamp(
                      0.0,
                      1.0,
                    );
                    final y = -4.0 * (1.0 - (2.0 * t - 1.0) * (2.0 * t - 1.0));
                    return Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 3),
                      child: Transform.translate(
                        offset: Offset(0, y),
                        child: Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: colors.primary.withAlpha(
                              (150 + (t * 105)).toInt(),
                            ),
                          ),
                        ),
                      ),
                    );
                  }),
                );
              },
            ),
            if (widget.stepLabel != null) ...[
              const SizedBox(height: 6),
              Text(
                widget.stepLabel!,
                style: TextStyle(
                  fontSize: 11,
                  color: colors.onSurface.withAlpha(120),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// Labeled input action button (camera, mic).
class _InputAction extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback? onTap;
  final Color? color;

  const _InputAction({
    required this.icon,
    required this.label,
    this.onTap,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    final c = color ?? Theme.of(context).colorScheme.onSurface;
    return InkWell(
      borderRadius: BorderRadius.circular(12),
      onTap: onTap == null
          ? null
          : () {
              HapticFeedback.lightImpact();
              onTap!();
            },
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 22, color: c),
            const SizedBox(height: 2),
            Text(
              label,
              style: TextStyle(fontSize: 10, color: c.withAlpha(180)),
            ),
          ],
        ),
      ),
    );
  }
}

/// Circular send button with gradient.
class _SendButton extends StatelessWidget {
  final VoidCallback onTap;
  final Color color;

  const _SendButton({required this.onTap, required this.color});

  @override
  Widget build(BuildContext context) {
    return Tooltip(message: 'Send', child: _buildButton());
  }

  Widget _buildButton() {
    return GestureDetector(
      onTap: () {
        HapticFeedback.mediumImpact();
        onTap();
      },
      child: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          gradient: LinearGradient(
            colors: [color, color.withAlpha(200)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          boxShadow: [
            BoxShadow(
              color: color.withAlpha(80),
              blurRadius: 12,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: const Icon(Icons.arrow_upward, size: 20, color: Colors.white),
      ),
    );
  }
}
