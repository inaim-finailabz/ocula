import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/ai_manager.dart';
import '../services/indexer.dart';

class OculaSplashScreen extends StatefulWidget {
  const OculaSplashScreen({super.key});

  @override
  State<OculaSplashScreen> createState() => _OculaSplashScreenState();
}

class _OculaSplashScreenState extends State<OculaSplashScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _fadeController;
  String _statusText = 'Initializing...';

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _initializeApp();
  }

  Future<void> _initializeApp() async {
    // 1. Load saved user tier
    final prefs = await SharedPreferences.getInstance();
    final tier = prefs.getString('user_tier') ?? 'free';

    setState(() => _statusText = 'Loading AI engine...');

    // 2. Pre-warm the AI engine while animation plays
    final aiTier = tier == 'pro'
        ? AITier.pro
        : (tier == 'plus' ? AITier.plus : AITier.free);
    await AIManager().switchEngine(aiTier);

    setState(() => _statusText = 'Indexing your data...');

    // 3. Start background indexer (non-blocking)
    Indexer().runFullIndex();

    // 4. Minimum splash duration so animation completes
    await Future.delayed(const Duration(seconds: 3));

    // 5. Fade out and navigate
    final isFirstLaunch = !(prefs.getBool('onboarding_complete') ?? false);

    if (mounted) {
      await _fadeController.forward();
      Navigator.pushReplacementNamed(
        context,
        isFirstLaunch ? '/onboarding' : '/home',
      );
    }
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
              Text(
                _statusText,
                style: TextStyle(
                  color: Colors.white.withAlpha(120),
                  fontSize: 12,
                  letterSpacing: 1.5,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
