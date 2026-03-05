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
import 'package:url_launcher/url_launcher.dart';
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
import 'services/notification_service.dart';
import 'services/tray_service.dart';

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
    primary: Color(0xFF6C5CE7),
    onPrimary: Colors.white,
    secondary: Color(0xFF00CEC9),
    onSecondary: Colors.black,
    tertiary: Color(0xFFFF7675),
    error: Color(0xFFFF7675),
    onError: Colors.white,
    surface: Color(0xFF1E1E2E),
    onSurface: Color(0xFFE0E0E0),
    surfaceContainerHighest: Color(0xFF2A2A3E),
    outline: Color(0xFF444475),
  );

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ocula',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: _colorScheme,
        useMaterial3: true,
        scaffoldBackgroundColor: _colorScheme.surface,
        appBarTheme: AppBarTheme(
          backgroundColor: _colorScheme.surface,
          elevation: 0,
        ),
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
  final _scrollController = ScrollController();
  final _modelManager = OculaModelManager();
  StreamSubscription? _downloadProgressSubscription;
  StreamSubscription? _featureReadySubscription;
  StreamSubscription? _tierChangeSubscription;
  StreamSubscription? _freeModelStatusSubscription;

  bool _freeModelFailed = false;

  // ── Help Tour ──
  static final _orbKey = GlobalKey();
  static final _chatListKey = GlobalKey();
  static final _scopeChipsKey = GlobalKey();
  static final _cameraButtonKey = GlobalKey();
  static final _fileButtonKey = GlobalKey();
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

  void _startHelpTour() {
    if (mounted) setState(() => _showingHelpTour = true);
  }

  final List<_Message> _messages = [];
  bool _isThinking = false;
  bool _isListening = false;
  bool _isSpeaking = false;
  bool _isRecordingNotes = false;
  bool _stopRequested = false;
  OrbState _orbState = OrbState.idle;
  RetrievalScope _retrievalScope = RetrievalScope.all;
  File? _attachedImage;
  File? _attachedDocument;
  String? _attachedDocName;

  bool _orbExpanded = true;
  Map<String, double> _activeDownloads = {};

  late final AnimationController _orbSizeController;

  @override
  void initState() {
    super.initState();
    _speech = SpeechService(aiManager: _ai);
    _speech.init();
    // If the splash navigated before the model was ready (background install
    // path), auto-load when ensureFreeModelReady signals success.
    _freeModelStatusSubscription = _modelManager.freeModelStatusStream.listen((ok) {
      if (!mounted) return;
      if (ok) {
        setState(() => _freeModelFailed = false);
        _ensureLiteActivatedIfReady();
        // Plus/Pro download is strictly on-demand from Settings.
        // Auto-downloading here causes iOS to terminate the app when it
        // goes to background mid-download (HttpClient has no background
        // download privileges on iOS).
      } else {
        setState(() => _freeModelFailed = true);
      }
    });
    // Handle race: model may have finished installing before home screen opened.
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      await _ensureLiteActivatedIfReady();
    });

    _orchestrator.onAskInternet = _showInternetDialog;
    _orchestrator.onAskEnableConnectivity = _showConnectivityDialog;
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
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      final prefs = await SharedPreferences.getInstance();
      if ((prefs.getBool('show_help_tour') ?? false) && mounted) {
        _startHelpTour();
      }
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
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
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

    // ── Normal download progress ──
    // Show only the free-tier main model in the chat banner.
    // Plus/Pro progress is shown in Settings, not here.
    final primary = _activeDownloads.entries.where((e) {
      final m = OculaModelManager.models
          .where((m) => m.fileName == e.key)
          .firstOrNull;
      return m != null &&
          m.tier == AITier.free &&
          !m.isVisionProjector &&
          !m.isEmbeddingModel;
    }).firstOrNull;

    if (primary == null) return const SizedBox.shrink();

    final modelInfo = OculaModelManager.models
        .where((m) => m.fileName == primary.key)
        .firstOrNull;
    final label = modelInfo?.displayName ?? primary.key.split('.').first;
    final pct = (primary.value * 100).toStringAsFixed(0);

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 6, 16, 8),
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
      // Schedule initial reminders + daily briefing
      await notifService.scheduleCalendarReminders();
      await notifService.scheduleDailyBriefing();
    } catch (e) {
      debugPrint('[Ocula] Notification init error: $e');
    }
  }

  String _tierLabel(AITier tier) {
    switch (tier) {
      case AITier.free:
        return 'Ocula Lite';
      case AITier.plus:
        return 'Ocula Plus';
      case AITier.pro:
        return 'Ocula Pro';
      case AITier.enterprise:
        return 'Enterprise';
    }
  }

  Color _tierColor(AITier tier) {
    switch (tier) {
      case AITier.free:
        return const Color(0xFF00CEC9);
      case AITier.plus:
        return const Color(0xFF6C5CE7);
      case AITier.pro:
        return const Color(0xFFFD79A8);
      case AITier.enterprise:
        return const Color(0xFFFDCB6E);
    }
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
      await openAppSettings();
      return true;
    }
    return false;
  }

  void _pickImage() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Theme.of(context).colorScheme.surfaceContainerHighest,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.camera_alt),
                title: const Text('Take Photo'),
                onTap: () async {
                  Navigator.pop(ctx);
                  final picked = await ImagePicker().pickImage(
                    source: ImageSource.camera,
                    maxWidth: 1024,
                    maxHeight: 1024,
                    imageQuality: 85,
                  );
                  if (picked != null) {
                    setState(() => _attachedImage = File(picked.path));
                  }
                },
              ),
              ListTile(
                leading: const Icon(Icons.photo_library),
                title: const Text('Choose from Gallery'),
                onTap: () async {
                  Navigator.pop(ctx);
                  final picked = await ImagePicker().pickImage(
                    source: ImageSource.gallery,
                    maxWidth: 1024,
                    maxHeight: 1024,
                    imageQuality: 85,
                  );
                  if (picked != null) {
                    setState(() => _attachedImage = File(picked.path));
                  }
                },
              ),
            ],
          ),
        ),
      ),
    );
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
        _isRecordingNotes = false;
        _orbState = OrbState.idle;
      });
    }
  }

  String _recordingSummaryPrompt({
    required String transcript,
    required String contextType,
  }) {
    return 'You are an expert note-taking assistant. Summarize this '
        '$contextType recording into clear, structured notes.\n\n'
        'Required format:\n'
        '1) One-line summary\n'
        '2) Key points (bullets)\n'
        '3) Decisions (if any)\n'
        '4) Action items with owner and due date if mentioned\n'
        '5) Follow-up questions / unclear points\n\n'
        'If something is missing, write "Not mentioned". Keep it concise.\n\n'
        'Transcript:\n'
        '$transcript';
  }

  Future<void> _startRecordingSummary({required String contextType}) async {
    if (_isThinking || _isSpeaking) {
      await _stopEverything();
    }
    if (_isListening) {
      await _speech.stopListening();
      if (!mounted) return;
      setState(() {
        _isListening = false;
        _isRecordingNotes = false;
        _orbState = OrbState.idle;
      });
      return;
    }

    if (mounted) {
      setState(() {
        _isListening = true;
        _isRecordingNotes = true;
        _orbState = OrbState.listening;
        _orbExpanded = true;
      });
      _textController.clear();
      _orbSizeController.forward();
    }

    try {
      _speech.startListening(
        onResult: (text) {
          _textController.text = text;
        },
        onFinalText: (transcript) {
          final cleaned = transcript.trim();
          if (!mounted) return;
          setState(() {
            _isListening = false;
            _isRecordingNotes = false;
            _orbState = OrbState.idle;
            _textController.clear();
          });
          if (cleaned.isEmpty) {
            _showSnackbar('No speech captured. Try recording again.');
            return;
          }
          _send(
            'Summarize my $contextType recording',
            queryOverride: _recordingSummaryPrompt(
              transcript: cleaned,
              contextType: contextType,
            ),
            displayTextOverride: 'Recorded $contextType notes',
            forcedTier: AITier.pro,
          );
        },
        onError: () {
          if (!mounted) return;
          setState(() {
            _isListening = false;
            _isRecordingNotes = false;
            _orbState = OrbState.idle;
          });
          _showSnackbar(
            'Could not start listening. Please check microphone permissions.',
          );
        },
      );
    } on ModelNotReadyException catch (e) {
      _showSnackbar(e.toString());
      if (!mounted) return;
      setState(() {
        _isListening = false;
        _isRecordingNotes = false;
        _orbState = OrbState.idle;
      });
    } catch (e) {
      _showSnackbar('An error occurred: $e');
      if (!mounted) return;
      setState(() {
        _isListening = false;
        _isRecordingNotes = false;
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

    // Reset stop flag — this new query is intentional
    _stopRequested = false;

    setState(() {
      _messages.add(_Message(text: displayText, isUser: true, image: image));
      _attachedImage = null;
      _attachedDocument = null;
      _attachedDocName = null;
      _isThinking = true;
      _orbState = OrbState.thinking;
      _orbExpanded = true;
    });
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
        _isRecordingNotes = false;
        _orbState = OrbState.idle;
      });
    } else {
      setState(() {
        _isListening = true;
        _isRecordingNotes = false;
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
              _isRecordingNotes = false;
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
              _isRecordingNotes = false;
              _orbState = OrbState.idle;
            });
          },
        );
      } on ModelNotReadyException catch (e) {
        _showSnackbar(e.toString());
        setState(() {
          _isListening = false;
          _isRecordingNotes = false;
          _orbState = OrbState.idle;
        });
      } catch (e) {
        _showSnackbar('An error occurred: $e');
        setState(() {
          _isListening = false;
          _isRecordingNotes = false;
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
    _scrollController.dispose();
    _orbSizeController.dispose();
    _downloadProgressSubscription?.cancel();
    _featureReadySubscription?.cancel();
    _tierChangeSubscription?.cancel();
    _freeModelStatusSubscription?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final screenHeight = MediaQuery.of(context).size.height;
    final hasMessages = _messages.isNotEmpty || _isThinking;

    const expandedSize = 160.0;
    const miniSize = 56.0;

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [colors.surface, const Color(0xFF16162A)],
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
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 12,
                    ),
                    child: Row(
                      children: [
                        ShaderMask(
                          shaderCallback: (bounds) => const LinearGradient(
                            colors: [Color(0xFF6C5CE7), Color(0xFF00CEC9)],
                          ).createShader(bounds),
                          child: const Text(
                            'Ocula',
                            style: TextStyle(
                              fontSize: 26,
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
                            ),
                          ),
                        ),
                        const SizedBox(width: 10),
                        // Active model badge
                        if (_ai.activeTier != null)
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 3,
                            ),
                            decoration: BoxDecoration(
                              color: _tierColor(_ai.activeTier!).withAlpha(40),
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(
                                color: _tierColor(
                                  _ai.activeTier!,
                                ).withAlpha(80),
                                width: 0.5,
                              ),
                            ),
                            child: Text(
                              _tierLabel(_ai.activeTier!),
                              style: TextStyle(
                                fontSize: 10,
                                fontWeight: FontWeight.w600,
                                color: _tierColor(_ai.activeTier!),
                              ),
                            ),
                          ),
                        const Spacer(),
                        if (_messages.any((m) => !m.isUser))
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
                                    ? box.localToGlobal(Offset.zero) & box.size
                                    : null;
                                _export.exportAndShare(
                                  lastResponse,
                                  origin: origin,
                                );
                              },
                            ),
                          ),
                        IconButton(
                          icon: const Icon(Icons.add_comment_outlined, size: 20),
                          tooltip: 'New chat',
                          onPressed: _startNewSession,
                        ),
                        IconButton(
                          icon: const Icon(Icons.mic_none_rounded, size: 20),
                          tooltip: 'Record meeting / lecture',
                          onPressed: () {
                            Navigator.of(context).pushNamed('/recorder');
                          },
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
                        // Overflow menu: history + help
                        Builder(
                          builder: (ctx) => PopupMenuButton<String>(
                            icon: const Icon(Icons.more_vert, size: 20),
                            tooltip: 'More',
                            onSelected: (value) {
                              if (value == 'history') {
                                _openSessionHistory(ctx);
                              } else if (value == 'help') {
                                _startHelpTour();
                              }
                            },
                            itemBuilder: (_) => const [
                              PopupMenuItem(
                                value: 'history',
                                child: Row(
                                  children: [
                                    Icon(Icons.history_outlined, size: 18),
                                    SizedBox(width: 10),
                                    Text('Chat history'),
                                  ],
                                ),
                              ),
                              PopupMenuItem(
                                value: 'help',
                                child: Row(
                                  children: [
                                    Icon(Icons.help_outline, size: 18),
                                    SizedBox(width: 10),
                                    Text('Help tour'),
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
                  // Show download progress only for free-tier install and only
                  // while no model is active (first-time install). Plus/Pro
                  // progress stays in Settings — not here — to avoid confusion.
                  if (_freeModelFailed ||
                      (_activeDownloads.keys.any((k) =>
                              OculaModelManager.models.any(
                                (m) => m.fileName == k && m.tier == AITier.free,
                              )) &&
                          _ai.activeTier == null))
                    _buildDownloadBanner(colors),

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
                            itemCount: _messages.length + (_isThinking ? 1 : 0),
                            itemBuilder: (context, index) {
                              if (_isThinking && index == _messages.length) {
                                return const _ThinkingBubble();
                              }
                              final msg = _messages[index];
                              // Find the user query that preceded this AI response
                              final precedingQuery = (!msg.isUser && index > 0)
                                  ? _messages[index - 1].text
                                  : null;
                              return _MessageBubble(
                                message: msg,
                                onCopy: () => _copyToClipboard(msg.text),
                                onSearchWeb: (!msg.isUser && precedingQuery != null)
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

                  // ── Attached image preview ──
                  if (_attachedImage != null)
                    Container(
                      padding: const EdgeInsets.fromLTRB(16, 6, 16, 0),
                      child: Row(
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

                  // ── Retrieval scope chips ──
                  SizedBox(
                    key: _scopeChipsKey,
                    height: 40,
                    child: ListView(
                      scrollDirection: Axis.horizontal,
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      children: [
                        _ScopeChip(
                          label: 'All',
                          icon: Icons.tune,
                          selected: _retrievalScope == RetrievalScope.all,
                          onTap: () => setState(
                            () => _retrievalScope = RetrievalScope.all,
                          ),
                        ),
                        _ScopeChip(
                          label: 'Docs',
                          icon: Icons.description_outlined,
                          selected: _retrievalScope == RetrievalScope.docs,
                          onTap: () => setState(
                            () => _retrievalScope = RetrievalScope.docs,
                          ),
                        ),
                        _ScopeChip(
                          label: 'Images',
                          icon: Icons.photo_outlined,
                          selected: _retrievalScope == RetrievalScope.images,
                          onTap: () => setState(
                            () => _retrievalScope = RetrievalScope.images,
                          ),
                        ),
                        _ScopeChip(
                          label: 'Location',
                          icon: Icons.place_outlined,
                          selected: _retrievalScope == RetrievalScope.location,
                          onTap: () => setState(
                            () => _retrievalScope = RetrievalScope.location,
                          ),
                        ),
                      ],
                    ),
                  ),

                  // ── Quick voice-note actions ──
                  SizedBox(
                    height: 36,
                    child: ListView(
                      scrollDirection: Axis.horizontal,
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      children: [
                        _QuickActionChip(
                          label: _isRecordingNotes
                              ? 'Stop Recording'
                              : 'Meeting Recap',
                          icon: _isRecordingNotes ? Icons.stop : Icons.mic,
                          selected: _isRecordingNotes,
                          onTap: () =>
                              _startRecordingSummary(contextType: 'meeting'),
                        ),
                        _QuickActionChip(
                          label: 'Class Notes',
                          icon: Icons.school_outlined,
                          onTap: () => _startRecordingSummary(
                            contextType: 'college class',
                          ),
                        ),
                      ],
                    ),
                  ),

                  // ── Enhanced input bar ──
                  Container(
                    key: _inputBarKey,
                    padding: const EdgeInsets.fromLTRB(6, 8, 6, 8),
                    decoration: BoxDecoration(
                      color: colors.surfaceContainerHighest,
                      borderRadius: const BorderRadius.vertical(
                        top: Radius.circular(24),
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withAlpha(40),
                          blurRadius: 12,
                          offset: const Offset(0, -4),
                        ),
                      ],
                    ),
                    child: SafeArea(
                      top: false,
                      child: Row(
                        children: [
                          // Camera button
                          SizedBox(
                            key: _cameraButtonKey,
                            child: _InputAction(
                              icon: Icons.camera_alt_outlined,
                              label: 'Camera',
                              onTap: _pickImage,
                            ),
                          ),
                          // Document attach button
                          SizedBox(
                            key: _fileButtonKey,
                            child: _InputAction(
                              icon: Icons.attach_file,
                              label: 'File',
                              onTap: _pickDocument,
                            ),
                          ),
                          // Text input
                          Expanded(
                            child: Container(
                              margin: const EdgeInsets.symmetric(horizontal: 8),
                              padding: const EdgeInsets.symmetric(
                                horizontal: 14,
                              ),
                              decoration: BoxDecoration(
                                color: colors.surface,
                                borderRadius: BorderRadius.circular(24),
                                border: Border.all(
                                  color: colors.outline.withAlpha(40),
                                ),
                              ),
                              child: TextField(
                                controller: _textController,
                                style: const TextStyle(fontSize: 15),
                                decoration: InputDecoration(
                                  hintText: 'Ask anything...',
                                  hintStyle: TextStyle(
                                    color: colors.onSurface.withAlpha(80),
                                  ),
                                  border: InputBorder.none,
                                  contentPadding: const EdgeInsets.symmetric(
                                    vertical: 12,
                                  ),
                                ),
                                onSubmitted: _send,
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
                          color: colors.surfaceContainerHighest.withAlpha(220),
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
                      targetKey: _scopeChipsKey,
                      title: 'Focus Your Search',
                      description:
                          'Filter by All / Docs / Images / Location to narrow what Ocula searches.',
                    ),
                    HelpStep(
                      targetKey: _inputBarKey,
                      title: 'Type a Question',
                      description:
                          'Type anything here. Use the buttons below to attach files or photos.',
                    ),
                    HelpStep(
                      targetKey: _cameraButtonKey,
                      title: 'Analyze Images',
                      description:
                          'Take or pick a photo — Ocula describes and analyzes it on-device.',
                    ),
                    HelpStep(
                      targetKey: _fileButtonKey,
                      title: 'Attach Documents',
                      description:
                          'Attach PDF, Word, Excel, PowerPoint or text files. Ocula reads and answers questions about them.',
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
    );
  }
}

/// A single chat message.
class _Message {
  final String text;
  final bool isUser;
  final File? image;
  final List<LinkedAsset> linkedAssets;

  _Message({
    required this.text,
    required this.isUser,
    this.image,
    this.linkedAssets = const [],
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
          margin: const EdgeInsets.only(bottom: 10),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          constraints: BoxConstraints(
            maxWidth: MediaQuery.of(context).size.width * 0.8,
          ),
          decoration: BoxDecoration(
            gradient: isUser
                ? const LinearGradient(
                    colors: [Color(0xFF6C5CE7), Color(0xFF5A4BD1)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  )
                : null,
            color: isUser ? null : colors.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(20).copyWith(
              bottomRight: isUser ? const Radius.circular(4) : null,
              bottomLeft: !isUser ? const Radius.circular(4) : null,
            ),
            border: isUser
                ? null
                : Border.all(color: colors.outline.withAlpha(25)),
            boxShadow: [
              BoxShadow(
                color: isUser
                    ? const Color(0xFF6C5CE7).withAlpha(40)
                    : Colors.black.withAlpha(20),
                blurRadius: 8,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
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
                  fontSize: 15,
                  height: 1.45,
                ),
              ),
              // Linked asset chips
              if (!isUser && message.linkedAssets.isNotEmpty) ...[
                const SizedBox(height: 8),
                Wrap(
                  spacing: 6,
                  runSpacing: 4,
                  children: [
                    ...message.linkedAssets.take(12).map((asset) {
                      return _AssetChip(asset: asset);
                    }),
                    if (message.linkedAssets.length > 12)
                      _OverflowChip(extra: message.linkedAssets.length - 12),
                  ],
                ),
              ],
              // "Search Web" offer — shown when local data was insufficient
              if (!isUser && onSearchWeb != null && _isNoDataResponse(message.text)) ...[
                const SizedBox(height: 10),
                GestureDetector(
                  onTap: onSearchWeb,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: colors.primaryContainer.withAlpha(180),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: colors.primary.withAlpha(60)),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.language_outlined, size: 14, color: colors.primary),
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

class _OverflowChip extends StatelessWidget {
  final int extra;

  const _OverflowChip({required this.extra});

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: colors.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: colors.outline.withAlpha(40)),
      ),
      child: Text(
        '+$extra more',
        style: TextStyle(
          color: colors.onSurface,
          fontSize: 12,
          fontWeight: FontWeight.w500,
        ),
      ),
    );
  }
}

/// Tappable chip for a linked asset (file, photo, email, contact, calendar, URL).
class _AssetChip extends StatelessWidget {
  final LinkedAsset asset;

  const _AssetChip({required this.asset});

  IconData get _icon {
    switch (asset.assetType) {
      case 'file':
        final ext = asset.assetRef.contains('.')
            ? asset.assetRef.split('.').last.toLowerCase()
            : '';
        if (ext == 'pdf') return Icons.picture_as_pdf_outlined;
        return Icons.insert_drive_file_outlined;
      case 'photo':
        return Icons.photo_outlined;
      case 'email':
        return Icons.email_outlined;
      case 'contact':
        return Icons.person_outline;
      case 'calendar':
        return Icons.calendar_today_outlined;
      case 'phone':
        return Icons.phone_outlined;
      case 'link':
        return Icons.link;
      default:
        return Icons.attach_file;
    }
  }

  Uri? _uriForAsset() {
    switch (asset.assetType) {
      case 'file':
      case 'photo':
        if (asset.assetRef.startsWith('file://')) {
          return Uri.tryParse(asset.assetRef);
        }
        return Uri.file(asset.assetRef);
      case 'email':
        if (asset.assetRef.startsWith('mailto:')) {
          return Uri.tryParse(asset.assetRef);
        }
        return Uri(scheme: 'mailto', path: asset.assetRef.trim());
      case 'phone':
        if (asset.assetRef.startsWith('tel:')) {
          return Uri.tryParse(asset.assetRef);
        }
        final digits = asset.assetRef.replaceAll(RegExp(r'[^0-9+]'), '');
        return digits.isEmpty ? null : Uri(scheme: 'tel', path: digits);
      case 'contact':
        if (asset.assetRef.startsWith('tel:') ||
            asset.assetRef.startsWith('mailto:')) {
          return Uri.tryParse(asset.assetRef);
        }
        if (asset.assetRef.contains('@')) {
          return Uri(scheme: 'mailto', path: asset.assetRef.trim());
        }
        final digits = asset.assetRef.replaceAll(RegExp(r'[^0-9+]'), '');
        return digits.isEmpty ? null : Uri(scheme: 'tel', path: digits);
      case 'link':
        final raw = asset.assetRef.trim();
        final parsed = Uri.tryParse(raw);
        if (parsed != null && parsed.hasScheme) return parsed;
        return Uri.tryParse('https://$raw');
      default:
        return null;
    }
  }

  Future<void> _onTap(BuildContext context) async {
    final uri = _uriForAsset();
    if (uri == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('This source cannot be opened yet.')),
      );
      return;
    }
    final launched = await launchUrl(uri, mode: LaunchMode.externalApplication);
    if (!launched && context.mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Could not open source: ${asset.label ?? asset.assetRef}',
          ),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return GestureDetector(
      onTap: () => _onTap(context),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: colors.primaryContainer.withAlpha(80),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: colors.primary.withAlpha(40)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(_icon, size: 14, color: colors.primary),
            const SizedBox(width: 4),
            Flexible(
              child: Text(
                asset.label ?? asset.assetRef.split('/').last,
                style: TextStyle(
                  color: colors.primary,
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                ),
                overflow: TextOverflow.ellipsis,
                maxLines: 1,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Animated "thinking" indicator with pulsing dots.
class _ThinkingBubble extends StatefulWidget {
  const _ThinkingBubble();

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
        child: AnimatedBuilder(
          animation: _controller,
          builder: (context, _) {
            return Row(
              mainAxisSize: MainAxisSize.min,
              children: List.generate(3, (i) {
                final delay = i * 0.2;
                final t = ((_controller.value - delay) % 1.0).clamp(0.0, 1.0);
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
      ),
    );
  }
}

class _ScopeChip extends StatelessWidget {
  final String label;
  final IconData icon;
  final bool selected;
  final VoidCallback onTap;

  const _ScopeChip({
    required this.label,
    required this.icon,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
          decoration: BoxDecoration(
            color: selected
                ? colors.primary.withAlpha(40)
                : colors.surfaceContainerHighest.withAlpha(120),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: selected
                  ? colors.primary.withAlpha(110)
                  : colors.outline.withAlpha(45),
            ),
          ),
          child: Row(
            children: [
              Icon(
                icon,
                size: 14,
                color: selected ? colors.primary : colors.onSurface,
              ),
              const SizedBox(width: 5),
              Text(
                label,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: selected ? colors.primary : colors.onSurface,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _QuickActionChip extends StatelessWidget {
  final String label;
  final IconData icon;
  final bool selected;
  final VoidCallback onTap;

  const _QuickActionChip({
    required this.label,
    required this.icon,
    required this.onTap,
    this.selected = false,
  });

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: InkWell(
        borderRadius: BorderRadius.circular(18),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: selected
                ? const Color(0xFFFF7675).withAlpha(35)
                : colors.secondary.withAlpha(26),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(
              color: selected
                  ? const Color(0xFFFF7675).withAlpha(110)
                  : colors.secondary.withAlpha(90),
            ),
          ),
          child: Row(
            children: [
              Icon(
                icon,
                size: 14,
                color: selected ? const Color(0xFFFF7675) : colors.secondary,
              ),
              const SizedBox(width: 5),
              Text(
                label,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: selected ? const Color(0xFFFF7675) : colors.secondary,
                ),
              ),
            ],
          ),
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
      onTap: onTap,
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
    return GestureDetector(
      onTap: onTap,
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
