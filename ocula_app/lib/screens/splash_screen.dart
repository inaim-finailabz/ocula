import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/ai_manager.dart';
import '../services/indexer.dart';
import '../services/model_manager.dart';

class OculaSplashScreen extends StatefulWidget {
  const OculaSplashScreen({super.key});

  @override
  State<OculaSplashScreen> createState() => _OculaSplashScreenState();
}

class _OculaSplashScreenState extends State<OculaSplashScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _fadeController;
  String _statusText = 'Initializing...';
  double _downloadProgress = 0.0;
  String? _errorText;
  DateTime? _startTime;
  String _timeInfo = '';
  bool _showRetry = false;

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _startTime = DateTime.now();
    _initializeApp();
  }

  void _updateTimeInfo() {
    if (_startTime == null) return;
    final elapsed = DateTime.now().difference(_startTime!).inSeconds;
    final minutes = elapsed ~/ 60;
    final seconds = elapsed % 60;
    
    String timeStr;
    if (minutes > 0) {
      timeStr = '${minutes}m ${seconds}s';
    } else {
      timeStr = '${seconds}s';
    }
    
    setState(() {
      _timeInfo = 'Elapsed: $timeStr';
    });
  }

  Stopwatch _stepWatch = Stopwatch();

  void _logStep(String step, [String? detail]) {
    final elapsed = _stepWatch.elapsedMilliseconds;
    final msg = '[Splash] [$step] ${elapsed}ms${detail != null ? ' — $detail' : ''}';
    debugPrint(msg);
    // Only show internal step names on-screen in debug builds.
    // In release builds the progress callbacks handle the status text.
    if (kDebugMode && mounted) {
      setState(() => _statusText = '$step${detail != null ? '\n$detail' : ''}');
    }
  }

  Future<void> _initializeApp() async {
    _stepWatch = Stopwatch()..start();
    // Update timer every second
    final timer = Stream.periodic(const Duration(seconds: 1))
        .listen((_) => _updateTimeInfo());

    try {
      // ── Step 0: Platform & network diagnostics ──
      _logStep('INIT', 'Starting initialization sequence');
      debugPrint('[Splash] ═══════════════════════════════════════');
      debugPrint('[Splash] Platform: ${Platform.operatingSystem} ${Platform.operatingSystemVersion}');
      debugPrint('[Splash] Dart version: ${Platform.version}');
      debugPrint('[Splash] ═══════════════════════════════════════');

      // Local dev server check — only in debug builds to avoid a 5 s timeout
      // on every production/TestFlight launch where localhost:8080 is never up.
      if (kDebugMode) {
        final serverHost = Platform.isAndroid ? '10.0.2.2' : 'localhost';
        _logStep('NET-CHECK', 'Testing connection to $serverHost:8080...');
        try {
          final httpClient = HttpClient();
          httpClient.connectionTimeout = const Duration(seconds: 5);
          final req = await httpClient.getUrl(Uri.parse('http://$serverHost:8080/'));
          final resp = await req.close();
          await resp.drain();
          _logStep('NET-CHECK', '$serverHost:8080 reachable — HTTP ${resp.statusCode}');
          httpClient.close();
        } catch (netErr) {
          _logStep('NET-CHECK', '$serverHost:8080 UNREACHABLE — $netErr');
        }
      }

      _logStep('PREFS', 'Loading SharedPreferences...');
      final prefs = await SharedPreferences.getInstance();
      _logStep('PREFS', 'SharedPreferences loaded');

      final modelManager = OculaModelManager();

      // ── Step 0.5: Clear stale "downloading_*" flags from crashed sessions ──
      // In-memory download state doesn't survive restarts; without this,
      // the model management screen shows "downloading" for models not downloading.
      await modelManager.clearStaleDownloadFlags();

      // ── Step 1: Resolve models directory ──
      _logStep('MODELS-DIR', 'Resolving models directory...');
      final dir = await modelManager.modelsDir;
      _logStep('MODELS-DIR', 'Path: $dir');
      final dirExists = await Directory(dir).exists();
      _logStep('MODELS-DIR', 'Exists: $dirExists');
      if (dirExists) {
        final files = await Directory(dir).list().toList();
        _logStep('MODELS-DIR', '${files.length} files found');
        for (final f in files) {
          final stat = await f.stat();
          debugPrint('[Splash]   • ${f.path.split('/').last} (${(stat.size / (1024 * 1024)).toStringAsFixed(1)} MB)');
        }
      }

      // ── Step 2: Check if model already exists ──
      // On desktop, prefer the highest-quality downloaded tier (Pro > Plus > Free).
      final isDesktopCheck =
          Platform.isMacOS || Platform.isWindows || Platform.isLinux;
      AITier? bestReadyTier;
      if (isDesktopCheck) {
        for (final tier in [AITier.pro, AITier.plus, AITier.free]) {
          final mainModel = OculaModelManager.models
              .where((m) =>
                  m.tier == tier &&
                  !m.isVisionProjector &&
                  !m.isEmbeddingModel)
              .firstOrNull;
          if (mainModel != null &&
              await modelManager.isDownloaded(mainModel.fileName)) {
            bestReadyTier = tier;
            _logStep('MODEL-CHECK', 'Best downloaded tier: ${tier.name}');
            break;
          }
        }
      }
      _logStep('MODEL-CHECK', 'Checking isFreeModelReady...');
      final alreadyReady = bestReadyTier != null
          ? true
          : await modelManager.isFreeModelReady;
      final tierToLoad = bestReadyTier ?? AITier.free;
      _logStep('MODEL-CHECK', 'alreadyReady=$alreadyReady tier=${tierToLoad.name}');

      if (alreadyReady) {
        _logStep('MODEL-LOAD', 'Model already on disk — loading into memory...');
        if (mounted) setState(() => _statusText = 'Loading AI engine...');
        final path = await modelManager.mainModelPath(tierToLoad);
        _logStep('MODEL-LOAD', 'mainModelPath = $path');
        if (path != null) {
          final fileSize = await File(path).length();
          _logStep('MODEL-LOAD', 'File size: ${(fileSize / (1024 * 1024)).toStringAsFixed(1)} MB');
        }
        try {
          await AIManager().switchEngine(tierToLoad);
          _logStep('MODEL-LOAD', 'switchEngine(${tierToLoad.name}) succeeded ✓');
        } catch (loadErr, loadStack) {
          _logStep('MODEL-LOAD', 'switchEngine FAILED: $loadErr');
          debugPrint('[Splash] Load stack: $loadStack');
          rethrow;
        }
        _completeInitialization(prefs, modelManager);
        timer.cancel();
        return;
      }

      // ── Step 3: Ensure model is ready ──
      // On desktop (macOS/Windows/Linux) there is no ODR/PAD delivery.
      // macOS downloads Pro tier on first run (best quality, no storage constraints).
      // Windows/Linux fall back to Free.
      // Mobile keeps fire-and-forget (ODR/PAD handles it).
      final isDesktop =
          Platform.isMacOS || Platform.isWindows || Platform.isLinux;

      if (isDesktop) {
        // On macOS/Windows/Linux: download Plus as default (Pro is on-demand from Settings).
        // Plus covers vision + text and fits all Apple Silicon Macs comfortably.
        _logStep('ENSURE-MODEL', 'Downloading Plus model (desktop first-run)...');
        if (mounted) setState(() => _statusText = 'Downloading AI model...');

        bool ok;
        if (Platform.isMacOS || Platform.isWindows || Platform.isLinux) {
          ok = await modelManager.downloadTier(
            AITier.plus,
            onProgress: (progress, status) {
              if (mounted) {
                setState(() {
                  _downloadProgress = progress;
                  _statusText = status;
                });
              }
            },
          );
          if (!ok) {
            // Fall back to Free if Plus download fails
            debugPrint('[Splash] Plus download failed, falling back to Free');
            ok = await modelManager.ensureFreeModelReady(
              onProgress: (progress, status) {
                if (mounted) {
                  setState(() {
                    _downloadProgress = progress;
                    _statusText = status;
                  });
                }
              },
            );
          }
        } else {
          ok = await modelManager.ensureFreeModelReady(
            onProgress: (progress, status) {
              if (mounted) {
                setState(() {
                  _downloadProgress = progress;
                  _statusText = status;
                });
              }
            },
          );
        }

        if (!ok && mounted) {
          setState(() {
            _errorText = 'Model download failed. Check your connection and retry.';
            _showRetry = true;
          });
          timer.cancel();
          return;
        }
        // Model is on disk — load it into memory before navigating.
        // Re-run tier detection to pick the best downloaded model.
        AITier? reloadTier;
        for (final tier in [AITier.pro, AITier.plus, AITier.free]) {
          final mainModel = OculaModelManager.models
              .where((m) => m.tier == tier && !m.isVisionProjector && !m.isEmbeddingModel)
              .firstOrNull;
          if (mainModel != null && await modelManager.isDownloaded(mainModel.fileName)) {
            reloadTier = tier;
            break;
          }
        }
        if (mounted) setState(() => _statusText = 'Loading AI engine...');
        try {
          await AIManager().switchEngine(reloadTier ?? AITier.free);
        } catch (e) {
          debugPrint('[Splash] Desktop model load error: $e');
        }
      } else {
        _logStep('ENSURE-MODEL', 'Firing ensureFreeModelReady in background...');
        modelManager.ensureFreeModelReady().catchError((e) {
          debugPrint('[Splash] Background model install error: $e');
        });
      }

      _completeInitialization(prefs, modelManager);
      timer.cancel();

    } on ModelNotReadyException catch (e) {
      timer.cancel();
      _logStep('ERROR', 'ModelNotReadyException: $e');
      setState(() {
        _errorText = 'AI engine not ready: ${e.toString()}';
        _showRetry = true;
      });
    } catch (e, stack) {
      timer.cancel();
      _logStep('ERROR', 'FATAL: $e');
      debugPrint('[Splash] ═══════════════════════════════════════');
      debugPrint('[Splash] ERROR TYPE: ${e.runtimeType}');
      debugPrint('[Splash] ERROR MSG:  $e');
      debugPrint('[Splash] STACK:\n$stack');
      debugPrint('[Splash] ═══════════════════════════════════════');
      setState(() {
        _errorText = 'Setup failed: ${e.toString()}';
        _showRetry = true;
      });
    }
  }
  
  Future<void> _completeInitialization(
    SharedPreferences prefs,
    OculaModelManager modelManager,
  ) async {
    // ── Step 4: Start background indexing (non-blocking) ──
    setState(() => _statusText = 'Finalizing setup...');
    Indexer().runFullIndex();

    // ── Step 5: Navigate ──
    // Plus/Pro are downloaded on-demand from Settings, not auto-downloaded here.
    // Free-tier projector and embedding are handled inside ensureFreeModelReady().
    await Future.delayed(const Duration(milliseconds: 800)); // Brief pause
    
    final isFirstLaunch = !(prefs.getBool('onboarding_complete') ?? false);

    if (mounted) {
      await _fadeController.forward();
      Navigator.pushReplacementNamed(
        context,
        isFirstLaunch ? '/onboarding' : '/home',
      );
    }
  }
  
  void _retry() {
    setState(() {
      _errorText = null;
      _showRetry = false;
      _statusText = 'Retrying...';
      _downloadProgress = 0.0;
      _timeInfo = '';
    });
    _startTime = DateTime.now();
    _initializeApp();
  }

  @override
  void dispose() {
    _fadeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0A0A),
      body: FadeTransition(
        opacity: Tween<double>(begin: 1.0, end: 0.0).animate(_fadeController),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Neural core animation
              Lottie.asset(
                'assets/animations/neural_core.json',
                width: 250,
                height: 250,
                // Falls back gracefully if file doesn't exist yet
                errorBuilder: (context, error, stackTrace) => Container(
                  width: 250,
                  height: 250,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(
                      color: Colors.deepPurple.withAlpha(100),
                      width: 2,
                    ),
                  ),
                  child: const Center(
                    child: CircularProgressIndicator(
                      color: Colors.deepPurple,
                      strokeWidth: 2,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 32),

              // App name
              const Text(
                'OCULA',
                style: TextStyle(
                  color: Colors.white,
                  letterSpacing: 8,
                  fontWeight: FontWeight.w200,
                  fontSize: 28,
                ),
              ),
              const SizedBox(height: 16),

              // Status text
              if (_errorText == null) ...[
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 32),
                  child: Text(
                    _statusText,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.white.withAlpha(120),
                      fontSize: 12,
                      letterSpacing: 1.5,
                      height: 1.4,
                    ),
                  ),
                ),
                if (_timeInfo.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  Text(
                    _timeInfo,
                    style: TextStyle(
                      color: Colors.white.withAlpha(80),
                      fontSize: 10,
                      letterSpacing: 1.2,
                    ),
                  ),
                ],
                if (_downloadProgress > 0.0 && _downloadProgress < 1.0) ...[
                  const SizedBox(height: 16),
                  SizedBox(
                    width: 200,
                    child: Column(
                      children: [
                        ClipRRect(
                          borderRadius: BorderRadius.circular(6),
                          child: LinearProgressIndicator(
                            value: _downloadProgress,
                            backgroundColor: Colors.white.withAlpha(20),
                            valueColor: const AlwaysStoppedAnimation<Color>(
                              Color(0xFF6C5CE7),
                            ),
                            minHeight: 6,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          '${(_downloadProgress * 100).toStringAsFixed(0)}% complete',
                          style: TextStyle(
                            color: Colors.white.withAlpha(100),
                            fontSize: 10,
                            letterSpacing: 1.0,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ],

              // Error text and retry button
              if (_errorText != null) ...[
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 32.0),
                  child: Column(
                    children: [
                      Text(
                        _errorText!,
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: Theme.of(context).colorScheme.error,
                          fontSize: 14,
                          letterSpacing: 1.2,
                          height: 1.4,
                        ),
                      ),
                      if (_showRetry) ...[
                        const SizedBox(height: 20),
                        ElevatedButton(
                          onPressed: _retry,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF6C5CE7),
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            padding: const EdgeInsets.symmetric(
                              horizontal: 24,
                              vertical: 12,
                            ),
                          ),
                          child: const Text('Retry'),
                        ),
                      ],
                    ],
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
