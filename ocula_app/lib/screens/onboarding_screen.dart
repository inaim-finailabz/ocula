import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// First-launch onboarding. Shows once, then never again.
/// Focus: Privacy + Multi-Modal + Assistant identity.
class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final _controller = PageController();
  int _currentPage = 0;

  static const _pages = [
    _OnboardingPage(
      icon: Icons.auto_awesome,
      title: 'Meet Ocula.',
      subtitle: 'Your personal assistant that lives entirely on your device.',
      color: Colors.deepPurple,
    ),
    _OnboardingPage(
      icon: Icons.cloud_off_rounded,
      title: 'Cloud-Free AI.',
      subtitle: 'Your thoughts, photos, and voice never leave this phone. Total privacy by design.',
      color: Colors.teal,
    ),
    _OnboardingPage(
      icon: Icons.mic_none_rounded,
      title: 'Voice First.',
      subtitle: 'Talk to Ocula like a friend. It listens and learns offline.',
      color: Colors.blue,
    ),
    _OnboardingPage(
      icon: Icons.security_rounded,
      title: 'Zero Data Leak.',
      subtitle: 'We physically cannot see your data. It never touches a server.',
      color: Colors.green,
    ),
  ];

  Future<void> _complete() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('onboarding_complete', true);
    if (mounted) Navigator.pushReplacementNamed(context, '/home');
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isLast = _currentPage == _pages.length - 1;

    return Scaffold(
      backgroundColor: const Color(0xFF0A0A0A),
      body: SafeArea(
        child: Column(
          children: [
            // Skip button
            Align(
              alignment: Alignment.topRight,
              child: TextButton(
                onPressed: _complete,
                child: Text(
                  'Skip',
                  style: TextStyle(color: Colors.white.withAlpha(120)),
                ),
              ),
            ),

            // Pages
            Expanded(
              child: PageView.builder(
                controller: _controller,
                itemCount: _pages.length,
                onPageChanged: (i) => setState(() => _currentPage = i),
                itemBuilder: (context, index) {
                  final page = _pages[index];
                  return _buildPage(page);
                },
              ),
            ),

            // Dots + Button
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 0, 24, 48),
              child: Row(
                children: [
                  // Page dots
                  Row(
                    children: List.generate(_pages.length, (i) {
                      return AnimatedContainer(
                        duration: const Duration(milliseconds: 300),
                        margin: const EdgeInsets.only(right: 8),
                        width: i == _currentPage ? 24 : 8,
                        height: 8,
                        decoration: BoxDecoration(
                          color: i == _currentPage
                              ? _pages[_currentPage].color
                              : Colors.white.withAlpha(50),
                          borderRadius: BorderRadius.circular(4),
                        ),
                      );
                    }),
                  ),
                  const Spacer(),
                  // Next / Get Started
                  FilledButton(
                    onPressed: () {
                      if (isLast) {
                        _complete();
                      } else {
                        _controller.nextPage(
                          duration: const Duration(milliseconds: 400),
                          curve: Curves.easeInOut,
                        );
                      }
                    },
                    style: FilledButton.styleFrom(
                      backgroundColor: _pages[_currentPage].color,
                      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
                    ),
                    child: Text(isLast ? 'Get Started' : 'Next'),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPage(_OnboardingPage page) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Animated orb
          _AssistantOrb(color: page.color),
          const SizedBox(height: 48),
          Text(
            page.title,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 28,
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 16),
          Text(
            page.subtitle,
            style: TextStyle(
              color: Colors.white.withAlpha(150),
              fontSize: 16,
              height: 1.5,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}

class _OnboardingPage {
  final IconData icon;
  final String title;
  final String subtitle;
  final Color color;

  const _OnboardingPage({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.color,
  });
}

/// Pulsing assistant orb animation.
class _AssistantOrb extends StatefulWidget {
  final Color color;
  const _AssistantOrb({required this.color});

  @override
  State<_AssistantOrb> createState() => _AssistantOrbState();
}

class _AssistantOrbState extends State<_AssistantOrb>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        final scale = 1.0 + (_controller.value * 0.15);
        final opacity = 0.3 + (_controller.value * 0.4);

        return Container(
          width: 140,
          height: 140,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(
                color: widget.color.withAlpha((opacity * 255).toInt()),
                blurRadius: 60,
                spreadRadius: 20,
              ),
            ],
          ),
          child: Transform.scale(
            scale: scale,
            child: Container(
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  colors: [
                    widget.color,
                    widget.color.withAlpha(50),
                    Colors.transparent,
                  ],
                  stops: const [0.0, 0.5, 1.0],
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
