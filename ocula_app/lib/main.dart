import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import 'package:url_launcher/url_launcher.dart';
import 'services/ai_manager.dart';
import 'services/speech_service.dart';
import 'services/export_service.dart';
import 'services/indexer.dart';
import 'services/model_manager.dart';
import 'services/orchestrator.dart';
import 'services/ocula_db.dart';
import 'services/share_receiver.dart';
import 'screens/splash_screen.dart';
import 'screens/onboarding_screen.dart';
import 'screens/settings_screen.dart';
import 'widgets/ocula_orb.dart';
import 'services/env_config.dart';
import 'services/notification_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  debugPrint('[Ocula] ENV=${EnvConfig.env} SERVER=${EnvConfig.modelServerUrl}');
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

  final List<_Message> _messages = [];
  bool _isThinking = false;
  bool _isListening = false;
  bool _isSpeaking = false;
  bool _stopRequested = false;
  OrbState _orbState = OrbState.idle;
  File? _attachedImage;
  File? _attachedDocument;
  String? _attachedDocName;

  bool _orbExpanded = true;

  late final AnimationController _orbSizeController;

  @override
  void initState() {
    super.initState();
    _speech = SpeechService(aiManager: _ai);
    _speech.init();
    // Free-tier model already loaded during splash — no need to call switchEngine here.

    _orchestrator.onAskInternet = _showInternetDialog;
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

    // Update tier badge when model switches (e.g. from settings or auto-route).
    _tierChangeSubscription = _ai.activeTierStream.listen((_) {
      if (mounted) setState(() {});
    });
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
          'txt',
          'md',
          'csv',
          'json',
          'xml',
          'html',
          'pdf',
          'py',
          'js',
          'ts',
          'dart',
          'java',
          'swift',
          'yaml',
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

  Future<void> _send(String text) async {
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
    String queryText = text;
    if (doc != null && docName != null) {
      try {
        final content = await doc.readAsString();
        if (content.isNotEmpty) {
          // Truncate to avoid overflowing context
          final truncated = content.length > 4000
              ? content.substring(0, 4000)
              : content;
          queryText =
              '[Attached document: $docName]\n'
              '--- DOCUMENT CONTENT ---\n$truncated\n--- END DOCUMENT ---\n\n'
              'User request: $text';
        }
      } catch (_) {
        // Binary file — just reference the name
        queryText =
            '[Attached file: $docName — binary file, cannot read text content] $text';
      }
    }

    // Display text — show the actual user text, not the augmented query
    final displayText =
        image != null &&
            text == 'Describe this image in detail. What do you see?'
        ? 'Analyze this image'
        : text;

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
          .run(queryText, hasImage: image != null, imagePath: image?.path)
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
    _scrollController.dispose();
    _orbSizeController.dispose();
    _downloadProgressSubscription?.cancel();
    _featureReadySubscription?.cancel();
    _tierChangeSubscription?.cancel();
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
                          icon: const Icon(Icons.settings_outlined, size: 20),
                          tooltip: 'Settings',
                          onPressed: () {
                            Navigator.of(context).push(
                              MaterialPageRoute(
                                builder: (_) => SettingsScreen(speech: _speech),
                              ),
                            );
                          },
                        ),
                      ],
                    ),
                  ),

                  // Chat messages
                  Expanded(
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
                              return _MessageBubble(
                                message: _messages[index],
                                onCopy: () =>
                                    _copyToClipboard(_messages[index].text),
                              );
                            },
                          )
                        : const SizedBox.shrink(),
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

                  // ── Enhanced input bar ──
                  Container(
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
                          _InputAction(
                            icon: Icons.camera_alt_outlined,
                            label: 'Camera',
                            onTap: _pickImage,
                          ),
                          // Document attach button
                          _InputAction(
                            icon: Icons.attach_file,
                            label: 'File',
                            onTap: _pickDocument,
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

  const _MessageBubble({required this.message, this.onCopy});

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
                  children: message.linkedAssets.take(5).map((asset) {
                    return _AssetChip(asset: asset);
                  }).toList(),
                ),
              ],
            ],
          ),
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
