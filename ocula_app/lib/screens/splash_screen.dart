import 'dart:async';
import 'dart:io';
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
    // Also show on-screen while debugging
    if (mounted) {
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

      // Network connectivity test to model server
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
        debugPrint('[Splash] Model server not reachable — bundled model will be used.');
      }

      _logStep('PREFS', 'Loading SharedPreferences...');
      final prefs = await SharedPreferences.getInstance();
      _logStep('PREFS', 'SharedPreferences loaded');

      final modelManager = OculaModelManager();

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
      _logStep('MODEL-CHECK', 'Checking isFreeModelReady...');
      final alreadyReady = await modelManager.isFreeModelReady;
      _logStep('MODEL-CHECK', 'isFreeModelReady = $alreadyReady');

      if (alreadyReady) {
        _logStep('MODEL-LOAD', 'Model already on disk — loading into memory...');
        final path = await modelManager.mainModelPath(AITier.free);
        _logStep('MODEL-LOAD', 'mainModelPath = $path');
        if (path != null) {
          final fileSize = await File(path).length();
          _logStep('MODEL-LOAD', 'File size: ${(fileSize / (1024 * 1024)).toStringAsFixed(1)} MB');
        }
        try {
          await AIManager().switchEngine(AITier.free);
          _logStep('MODEL-LOAD', 'switchEngine(free) succeeded ✓');
        } catch (loadErr, loadStack) {
          _logStep('MODEL-LOAD', 'switchEngine FAILED: $loadErr');
          debugPrint('[Splash] Load stack: $loadStack');
          rethrow;
        }
        _completeInitialization(prefs, modelManager);
        timer.cancel();
        return;
      }

      // ── Step 3: Ensure AI model ready (bundled or download) ──
      _logStep('ENSURE-MODEL', 'ensureFreeModelReady starting...');

      final freeReady = await modelManager.ensureFreeModelReady(
        onProgress: (progress, status) {
          final pct = (progress * 100).toStringAsFixed(0);
          debugPrint('[Splash] [DOWNLOAD] $pct% — $status');
          if (mounted) {
            String statusMessage;
            if (status.contains('Bundled')) {
              statusMessage = status;
            } else if (status.contains('Downloading')) {
              final estimatedTotal = progress > 0.1
                  ? (DateTime.now().difference(_startTime!).inSeconds / progress).round()
                  : null;

              statusMessage = 'Downloading AI engine... $pct%';
              if (estimatedTotal != null && estimatedTotal > 0) {
                final remaining = estimatedTotal - DateTime.now().difference(_startTime!).inSeconds;
                if (remaining > 0) {
                  final remainingMin = remaining ~/ 60;
                  final remainingSec = remaining % 60;
                  if (remainingMin > 0) {
                    statusMessage += '\n~${remainingMin}m ${remainingSec}s remaining';
                  } else {
                    statusMessage += '\n~${remainingSec}s remaining';
                  }
                }
              }
            } else if (progress < 1.0) {
              statusMessage = '$status $pct%';
            } else {
              statusMessage = status;
            }

            setState(() {
              _downloadProgress = progress;
              _statusText = statusMessage;
            });
          }
        },
      ).timeout(
        const Duration(minutes: 5),
        onTimeout: () {
          throw Exception('Setup timed out. If downloading, please check your internet connection.');
        },
      );

      _logStep('ENSURE-MODEL', 'ensureFreeModelReady returned: $freeReady');

      if (!freeReady) {
        _logStep('ENSURE-MODEL', 'Model NOT ready — showing retry');
        setState(() {
          _errorText = 'Could not prepare the AI engine. '
              'Please check your internet connection and try again.';
          _showRetry = true;
        });
        timer.cancel();
        return;
      }

      // ── Step 4: Load the free model into memory ──
      _logStep('MODEL-LOAD', 'Loading model into memory...');
      setState(() {
        _statusText = 'Loading AI engine into memory...';
        _downloadProgress = 1.0;
      });

      final modelPath = await modelManager.mainModelPath(AITier.free);
      _logStep('MODEL-LOAD', 'mainModelPath = $modelPath');

      if (modelPath != null) {
        final fileSize = await File(modelPath).length();
        _logStep('MODEL-LOAD', 'File size: ${(fileSize / (1024 * 1024)).toStringAsFixed(1)} MB');
      }

      try {
        await AIManager().switchEngine(AITier.free).timeout(
          const Duration(minutes: 2),
          onTimeout: () {
            throw Exception('AI engine loading timed out after 2 minutes.');
          },
        );
        _logStep('MODEL-LOAD', 'switchEngine(free) succeeded ✓');
      } catch (loadErr, loadStack) {
        _logStep('MODEL-LOAD', 'switchEngine FAILED: $loadErr');
        debugPrint('[Splash] Load stack: $loadStack');
        rethrow;
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

    // ── Step 5: Kick off background downloads for Plus/Pro ──
    modelManager.downloadRemainingInBackground();

    // ── Step 6: Navigate ──
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
