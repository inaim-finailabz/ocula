import 'dart:math';
import 'package:flutter/material.dart';

/// The Ocula assistant orb — fully dynamic, procedural animation.
/// No static image. The orb is a living, breathing sphere with
/// wave distortions that respond to state: idle, listening, thinking, speaking.
enum OrbState { idle, listening, thinking, speaking }

class OculaOrb extends StatefulWidget {
  final OrbState state;
  final double size;

  const OculaOrb({
    super.key,
    this.state = OrbState.idle,
    this.size = 120,
  });

  @override
  State<OculaOrb> createState() => _OculaOrbState();
}

class _OculaOrbState extends State<OculaOrb> with TickerProviderStateMixin {
  late final AnimationController _waveController;
  late final AnimationController _pulseController;
  late final AnimationController _glowController;

  @override
  void initState() {
    super.initState();

    _waveController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 3000),
    )..repeat();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000),
    )..repeat(reverse: true);

    _glowController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);
  }

  @override
  void didUpdateWidget(OculaOrb oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.state != widget.state) {
      _updateAnimationSpeed();
    }
  }

  void _updateAnimationSpeed() {
    switch (widget.state) {
      case OrbState.idle:
        _waveController.duration = const Duration(milliseconds: 3000);
        _pulseController.duration = const Duration(milliseconds: 2500);
        _glowController.duration = const Duration(milliseconds: 2000);
        break;
      case OrbState.listening:
        _waveController.duration = const Duration(milliseconds: 800);
        _pulseController.duration = const Duration(milliseconds: 600);
        _glowController.duration = const Duration(milliseconds: 500);
        break;
      case OrbState.thinking:
        _waveController.duration = const Duration(milliseconds: 1500);
        _pulseController.duration = const Duration(milliseconds: 1200);
        _glowController.duration = const Duration(milliseconds: 1000);
        break;
      case OrbState.speaking:
        _waveController.duration = const Duration(milliseconds: 600);
        _pulseController.duration = const Duration(milliseconds: 500);
        _glowController.duration = const Duration(milliseconds: 400);
        break;
    }
  }

  Color _primaryColor() {
    switch (widget.state) {
      case OrbState.idle:
        return const Color(0xFF7C3AED); // purple
      case OrbState.listening:
        return const Color(0xFFEF4444); // red
      case OrbState.thinking:
        return const Color(0xFFF59E0B); // amber
      case OrbState.speaking:
        return const Color(0xFF06B6D4); // cyan
    }
  }

  Color _secondaryColor() {
    switch (widget.state) {
      case OrbState.idle:
        return const Color(0xFF3B82F6); // blue
      case OrbState.listening:
        return const Color(0xFFF97316); // orange
      case OrbState.thinking:
        return const Color(0xFF10B981); // green
      case OrbState.speaking:
        return const Color(0xFF8B5CF6); // violet
    }
  }

  double _waveIntensity() {
    switch (widget.state) {
      case OrbState.idle:
        return 0.03;
      case OrbState.listening:
        return 0.12;
      case OrbState.thinking:
        return 0.06;
      case OrbState.speaking:
        return 0.15;
    }
  }

  @override
  void dispose() {
    _waveController.dispose();
    _pulseController.dispose();
    _glowController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: Listenable.merge([_waveController, _pulseController, _glowController]),
      builder: (context, child) {
        final pulseScale = 1.0 + (_pulseController.value * 0.06);
        final glowIntensity = 0.3 + (_glowController.value * 0.5);
        final primary = _primaryColor();
        final secondary = _secondaryColor();

        return SizedBox(
          width: widget.size * 1.5,
          height: widget.size * 1.5,
          child: Stack(
            alignment: Alignment.center,
            children: [
              // Outer glow
              Container(
                width: widget.size * 1.35,
                height: widget.size * 1.35,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: primary.withAlpha((glowIntensity * 100).toInt()),
                      blurRadius: 50 + (_glowController.value * 30),
                      spreadRadius: 8 + (_glowController.value * 15),
                    ),
                    BoxShadow(
                      color: secondary.withAlpha((glowIntensity * 50).toInt()),
                      blurRadius: 30 + (_glowController.value * 20),
                      spreadRadius: 4,
                    ),
                  ],
                ),
              ),

              // The dynamic orb sphere
              Transform.scale(
                scale: pulseScale,
                child: CustomPaint(
                  size: Size(widget.size, widget.size),
                  painter: _OrbPainter(
                    wavePhase: _waveController.value * 2 * pi,
                    waveIntensity: _waveIntensity(),
                    primaryColor: primary,
                    secondaryColor: secondary,
                    glowValue: _glowController.value,
                  ),
                ),
              ),

              // State label
              Positioned(
                bottom: 4,
                child: AnimatedSwitcher(
                  duration: const Duration(milliseconds: 300),
                  child: widget.state != OrbState.idle
                      ? Container(
                          key: ValueKey(widget.state),
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 4,
                          ),
                          decoration: BoxDecoration(
                            color: primary.withAlpha(200),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            _stateLabel(),
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 10,
                              fontWeight: FontWeight.w600,
                              letterSpacing: 1.5,
                            ),
                          ),
                        )
                      : const SizedBox.shrink(),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  String _stateLabel() {
    switch (widget.state) {
      case OrbState.idle:
        return '';
      case OrbState.listening:
        return 'LISTENING';
      case OrbState.thinking:
        return 'THINKING';
      case OrbState.speaking:
        return 'SPEAKING';
    }
  }
}

/// Custom painter that draws a dynamic, procedural orb with wave distortions.
/// The sphere surface is made of concentric rings with sine-wave displacement,
/// creating a living, organic look.
class _OrbPainter extends CustomPainter {
  final double wavePhase;
  final double waveIntensity;
  final Color primaryColor;
  final Color secondaryColor;
  final double glowValue;

  _OrbPainter({
    required this.wavePhase,
    required this.waveIntensity,
    required this.primaryColor,
    required this.secondaryColor,
    required this.glowValue,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2;

    // ── Background sphere gradient ──
    final bgPaint = Paint()
      ..shader = RadialGradient(
        center: const Alignment(-0.3, -0.3),
        radius: 1.0,
        colors: [
          primaryColor.withAlpha(220),
          Color.lerp(primaryColor, secondaryColor, 0.5)!.withAlpha(180),
          secondaryColor.withAlpha(100),
          const Color(0xFF0A0A0A),
        ],
        stops: const [0.0, 0.35, 0.65, 1.0],
      ).createShader(Rect.fromCircle(center: center, radius: radius));

    canvas.drawCircle(center, radius, bgPaint);

    // ── Wave rings — concentric distorted circles ──
    final ringCount = 8;
    for (int i = 0; i < ringCount; i++) {
      final ringRadius = radius * (0.3 + (i / ringCount) * 0.65);
      final path = Path();
      final segments = 60;

      for (int j = 0; j <= segments; j++) {
        final angle = (j / segments) * 2 * pi;

        // Multiple wave frequencies for organic look
        final wave1 = sin(angle * 3 + wavePhase + i * 0.7) * waveIntensity * radius;
        final wave2 = sin(angle * 5 - wavePhase * 1.3 + i * 0.4) * waveIntensity * radius * 0.5;
        final wave3 = cos(angle * 2 + wavePhase * 0.8 + i) * waveIntensity * radius * 0.3;
        final displacement = wave1 + wave2 + wave3;

        final r = ringRadius + displacement;
        final x = center.dx + r * cos(angle);
        final y = center.dy + r * sin(angle);

        if (j == 0) {
          path.moveTo(x, y);
        } else {
          path.lineTo(x, y);
        }
      }
      path.close();

      final alpha = ((1.0 - i / ringCount) * 80 + 20).toInt();
      final ringPaint = Paint()
        ..color = Color.lerp(
          primaryColor.withAlpha(alpha),
          secondaryColor.withAlpha(alpha),
          i / ringCount,
        )!
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5 + (glowValue * 1.0);

      canvas.drawPath(path, ringPaint);
    }

    // ── Inner core highlight ──
    final corePaint = Paint()
      ..shader = RadialGradient(
        center: const Alignment(-0.2, -0.2),
        radius: 0.5,
        colors: [
          Colors.white.withAlpha((60 + glowValue * 40).toInt()),
          Colors.white.withAlpha(0),
        ],
      ).createShader(Rect.fromCircle(center: center, radius: radius * 0.4));

    canvas.drawCircle(center, radius * 0.4, corePaint);

    // ── Specular highlight (3D effect) ──
    final specCenter = Offset(
      center.dx - radius * 0.25,
      center.dy - radius * 0.25,
    );
    final specPaint = Paint()
      ..shader = RadialGradient(
        colors: [
          Colors.white.withAlpha((40 + glowValue * 30).toInt()),
          Colors.white.withAlpha(0),
        ],
      ).createShader(Rect.fromCircle(center: specCenter, radius: radius * 0.3));

    canvas.drawCircle(specCenter, radius * 0.3, specPaint);

    // ── Edge rim light ──
    final rimPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..shader = SweepGradient(
        colors: [
          primaryColor.withAlpha(0),
          primaryColor.withAlpha((80 + glowValue * 40).toInt()),
          secondaryColor.withAlpha((60 + glowValue * 30).toInt()),
          primaryColor.withAlpha(0),
        ],
        stops: const [0.0, 0.3, 0.7, 1.0],
        transform: GradientRotation(wavePhase * 0.5),
      ).createShader(Rect.fromCircle(center: center, radius: radius - 1));

    canvas.drawCircle(center, radius - 1, rimPaint);
  }

  @override
  bool shouldRepaint(_OrbPainter oldDelegate) => true;
}
