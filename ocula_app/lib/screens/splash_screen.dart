import 'dart:async';
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

  Future<void> _initializeApp() async {
    // Update timer every second
    final timer = Stream.periodic(const Duration(seconds: 1))
        .listen((_) => _updateTimeInfo());
    
    try {
      final prefs = await SharedPreferences.getInstance();
      final modelManager = OculaModelManager();

      // ── Step 1: Check if model already exists ──
      setState(() => _statusText = 'Checking AI engine...');
      await Future.delayed(const Duration(milliseconds: 500)); // Show the message
      
      final alreadyReady = await modelManager.isFreeModelReady;
      
      if (alreadyReady) {
        setState(() => _statusText = 'AI engine found, loading...');
        await AIManager().switchEngine(AITier.free);
        _completeInitialization(prefs, modelManager);
        timer.cancel();
        return;
      }

      // ── Step 2: Ensure AI model ready (bundled or download) ──
      setState(() => _statusText = 'Preparing AI engine...');

      final freeReady = await modelManager.ensureFreeModelReady(
        onProgress: (progress, status) {
          if (mounted) {
            final percentage = (progress * 100).toStringAsFixed(0);
            
            // Handle different status messages from model manager
            String statusMessage;
            if (status.contains('Bundled')) {
              statusMessage = status; // Use the bundled message directly
            } else if (status.contains('Downloading')) {
              // Only show time estimates for actual downloads
              final estimatedTotal = progress > 0.1 
                  ? (DateTime.now().difference(_startTime!).inSeconds / progress).round()
                  : null;
              
              statusMessage = 'Downloading AI engine... $percentage%';
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
              statusMessage = '$status $percentage%';
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
        const Duration(minutes: 5), // Reduced timeout since bundled models are instant
        onTimeout: () {
          throw Exception('Setup timed out. If downloading, please check your internet connection.');
        },
      );

      if (!freeReady) {
        setState(() {
          _errorText = 'Could not download the AI engine. '
              'Please check your internet connection and try again.';
          _showRetry = true;
        });
        timer.cancel();
        return;
      }

      // ── Step 3: Load the free model into memory ──
      setState(() {
        _statusText = 'Loading AI engine into memory...';
        _downloadProgress = 1.0;
      });
      
      await AIManager().switchEngine(AITier.free).timeout(
        const Duration(minutes: 2),
        onTimeout: () {
          throw Exception('AI engine loading timed out. Please restart the app.');
        },
      );

      _completeInitialization(prefs, modelManager);
      timer.cancel();
      
    } on ModelNotReadyException catch (e) {
      timer.cancel();
      setState(() {
        _errorText = 'AI engine not ready: ${e.toString()}';
        _showRetry = true;
      });
    } catch (e) {
      timer.cancel();
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
