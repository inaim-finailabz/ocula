import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';

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

  static final _pages = [
    const _OnboardingPage(
      icon: Icons.auto_awesome,
      title: 'Meet Ocula.',
      subtitle: 'Your personal assistant that lives entirely on your device.',
      color: Colors.deepPurple,
    ),
    const _OnboardingPage(
      icon: Icons.cloud_off_rounded,
      title: 'Cloud-Free AI.',
      subtitle: 'Your thoughts, photos, and voice never leave this phone. Total privacy by design.',
      color: Colors.teal,
    ),
    const _OnboardingPage(
      icon: Icons.mic_none_rounded,
      title: 'Voice First.',
      subtitle: 'Talk to Ocula like a friend. It listens and learns offline.',
      color: Colors.blue,
    ),
    const _OnboardingPage(
      icon: Icons.security_rounded,
      title: 'Zero Data Leak.',
      subtitle: 'We physically cannot see your data. It never touches a server.',
      color: Colors.green,
    ),
    const _OnboardingPage(
      icon: Icons.chat_bubble_outline,
      title: 'Ask Anything.',
      subtitle: 'Type or speak any question. Ocula reasons on-device and answers instantly.',
      color: Color(0xFF6C5CE7),
    ),
    const _OnboardingPage(
      icon: Icons.lock_person_rounded,
      title: 'Your Private Data.',
      subtitle: 'Search docs, photos, contacts, and calendar — without the cloud.',
      color: Colors.teal,
    ),
  ];

  Future<void> _complete() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('onboarding_complete', true);
    await prefs.setBool('show_help_tour', true);
    if (mounted) Navigator.pushReplacementNamed(context, '/home');
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  // Total pages = onboarding slides + 1 privacy consent page.
  int get _totalPages => _pages.length + 1;
  bool get _isConsentPage => _currentPage == _pages.length;

  Color get _currentColor => _isConsentPage
      ? const Color(0xFF6C5CE7)
      : _pages[_currentPage].color;

  @override
  Widget build(BuildContext context) {
    final isLast = _currentPage == _totalPages - 1;

    return Scaffold(
      backgroundColor: const Color(0xFF0A0A0A),
      body: SafeArea(
        child: Column(
          children: [
            // Skip button — hidden on the consent page (must tap Get Started)
            Align(
              alignment: Alignment.topRight,
              child: _isConsentPage
                  ? const SizedBox(height: 40)
                  : TextButton(
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
                itemCount: _totalPages,
                onPageChanged: (i) => setState(() => _currentPage = i),
                itemBuilder: (context, index) {
                  if (index == _pages.length) return _buildConsentPage();
                  if (index >= 4) return _buildFeaturePage(index);
                  return _buildPage(_pages[index]);
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
                    children: List.generate(_totalPages, (i) {
                      return AnimatedContainer(
                        duration: const Duration(milliseconds: 300),
                        margin: const EdgeInsets.only(right: 8),
                        width: i == _currentPage ? 24 : 8,
                        height: 8,
                        decoration: BoxDecoration(
                          color: i == _currentPage
                              ? _currentColor
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
                      backgroundColor: _currentColor,
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

  Widget _buildConsentPage() {
    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
      child: Column(
        children: [
          const SizedBox(height: 16),
          _AssistantOrb(color: const Color(0xFF6C5CE7)),
          const SizedBox(height: 32),
          const Text(
            'Before You Begin',
            style: TextStyle(
              color: Colors.white,
              fontSize: 26,
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 14),
          Text(
            'Ocula is a fully on-device AI assistant. '
            'Here is what it may access on this device, and how:',
            style: TextStyle(
              color: Colors.white.withAlpha(160),
              fontSize: 15,
              height: 1.5,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 24),
          _ConsentItem(
            icon: Icons.mic_none_rounded,
            title: 'Microphone & Voice',
            detail: 'Transcribed on-device using Apple\'s on-device Speech Recognition. '
                'Audio is never sent to Ocula servers or any third-party AI service.',
          ),
          _ConsentItem(
            icon: Icons.photo_library_outlined,
            title: 'Photos',
            detail: 'Analyzed locally by the AI model stored on this device. '
                'No image data leaves your phone.',
          ),
          _ConsentItem(
            icon: Icons.people_outline,
            title: 'Contacts',
            detail: 'Read locally to answer questions about people you know. '
                'Contact data is never uploaded.',
          ),
          _ConsentItem(
            icon: Icons.event_outlined,
            title: 'Calendar',
            detail: 'Read locally to help with scheduling questions. '
                'Calendar data stays on-device.',
          ),
          _ConsentItem(
            icon: Icons.insert_drive_file_outlined,
            title: 'Documents',
            detail: 'Indexed locally for search and Q&A. '
                'File contents are never transmitted.',
          ),
          const SizedBox(height: 20),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: const Color(0xFF6C5CE7).withAlpha(30),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: const Color(0xFF6C5CE7).withAlpha(80)),
            ),
            child: Text(
              'All AI inference runs on a model stored on your device. '
              'Ocula does not use any cloud-based or third-party AI service. '
              'No personal data is ever transmitted to Ocula, Finailabz, or any third party.',
              style: TextStyle(
                color: Colors.white.withAlpha(200),
                fontSize: 13,
                height: 1.5,
              ),
              textAlign: TextAlign.center,
            ),
          ),
          const SizedBox(height: 16),
          TextButton(
            onPressed: () => launchUrl(
              Uri.parse('https://finailabz.com/privacy'),
              mode: LaunchMode.externalApplication,
            ),
            child: const Text(
              'View Privacy Policy',
              style: TextStyle(color: Color(0xFF6C5CE7), fontSize: 14),
            ),
          ),
          const SizedBox(height: 8),
        ],
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

  Widget _buildFeaturePage(int index) {
    final page = _pages[index];
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          _AssistantOrb(color: page.color),
          const SizedBox(height: 40),
          Text(
            page.title,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 28,
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 12),
          Text(
            page.subtitle,
            style: TextStyle(
              color: Colors.white.withAlpha(150),
              fontSize: 16,
              height: 1.5,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 32),
          // Page 4: mock chat bubbles
          if (index == 4) ...[
            Align(
              alignment: Alignment.centerRight,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                decoration: BoxDecoration(
                  color: page.color.withAlpha(180),
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(16),
                    topRight: Radius.circular(16),
                    bottomLeft: Radius.circular(16),
                  ),
                ),
                child: const Text(
                  'What\'s the capital of France?',
                  style: TextStyle(color: Colors.white, fontSize: 14),
                ),
              ),
            ),
            const SizedBox(height: 10),
            Align(
              alignment: Alignment.centerLeft,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                decoration: BoxDecoration(
                  color: Colors.white.withAlpha(20),
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(16),
                    topRight: Radius.circular(16),
                    bottomRight: Radius.circular(16),
                  ),
                ),
                child: const Text(
                  'Paris is the capital of France.',
                  style: TextStyle(color: Colors.white, fontSize: 14),
                ),
              ),
            ),
          ],
          // Page 5: private data chips
          if (index == 5)
            Wrap(
              spacing: 12,
              runSpacing: 12,
              alignment: WrapAlignment.center,
              children: const [
                _DataChip(icon: Icons.insert_drive_file, label: 'Documents'),
                _DataChip(icon: Icons.photo_library, label: 'Photos'),
                _DataChip(icon: Icons.people, label: 'Contacts'),
                _DataChip(icon: Icons.event, label: 'Calendar'),
              ],
            ),
        ],
      ),
    );
  }
}

class _DataChip extends StatelessWidget {
  final IconData icon;
  final String label;
  const _DataChip({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.white.withAlpha(20),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withAlpha(40)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: Colors.white),
          const SizedBox(width: 6),
          Text(label, style: const TextStyle(color: Colors.white, fontSize: 13)),
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

class _ConsentItem extends StatelessWidget {
  final IconData icon;
  final String title;
  final String detail;
  const _ConsentItem({required this.icon, required this.title, required this.detail});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: const Color(0xFF6C5CE7), size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  detail,
                  style: TextStyle(
                    color: Colors.white.withAlpha(130),
                    fontSize: 13,
                    height: 1.4,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
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
