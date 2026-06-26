import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

enum OrbState { idle, listening, thinking, speaking }
enum AvatarStyle { orb, face }

class OculaOrb extends StatefulWidget {
  final OrbState state;
  final double size;
  final AvatarStyle avatarStyle;

  const OculaOrb({
    super.key,
    this.state = OrbState.idle,
    this.size = 120,
    this.avatarStyle = AvatarStyle.face,
  });

  @override
  State<OculaOrb> createState() => _OculaOrbState();
}

class _OculaOrbState extends State<OculaOrb> with TickerProviderStateMixin {
  late final AnimationController _waveController;
  late final AnimationController _pulseController;
  late final AnimationController _glowController;
  late final AnimationController _blinkController;
  late final FocusNode _focusNode;
  Timer? _blinkTimer;
  final _rng = Random();

  double _yaw = 0.30;
  double _pitch = -0.10;

  @override
  void initState() {
    super.initState();
    _focusNode = FocusNode(debugLabel: 'ocula_orb_focus');

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

    _blinkController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 170),
    );

    _scheduleBlink();
  }

  void _scheduleBlink() {
    final delay = 3000 + _rng.nextInt(5000);
    _blinkTimer = Timer(Duration(milliseconds: delay), () {
      if (!mounted) return;
      _blinkController
          .forward()
          .then((_) => mounted ? _blinkController.reverse() : null)
          .then((_) { if (mounted) _scheduleBlink(); });
    });
  }

  @override
  void didUpdateWidget(OculaOrb oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.state != widget.state) _updateAnimationSpeed();
  }

  void _updateAnimationSpeed() {
    switch (widget.state) {
      case OrbState.idle:
        _waveController.duration = const Duration(milliseconds: 4000);
        _pulseController.duration = const Duration(milliseconds: 3000);
        _glowController.duration = const Duration(milliseconds: 2500);
      case OrbState.listening:
        _waveController.duration = const Duration(milliseconds: 900);
        _pulseController.duration = const Duration(milliseconds: 700);
        _glowController.duration = const Duration(milliseconds: 600);
      case OrbState.thinking:
        _waveController.duration = const Duration(milliseconds: 1800);
        _pulseController.duration = const Duration(milliseconds: 1400);
        _glowController.duration = const Duration(milliseconds: 1100);
      case OrbState.speaking:
        _waveController.duration = const Duration(milliseconds: 700);
        _pulseController.duration = const Duration(milliseconds: 500);
        _glowController.duration = const Duration(milliseconds: 400);
    }
  }

  Color _primaryColor() => switch (widget.state) {
    OrbState.idle     => const Color(0xFF7C3AED),
    OrbState.listening => const Color(0xFFEF4444),
    OrbState.thinking => const Color(0xFFF59E0B),
    OrbState.speaking => const Color(0xFF06B6D4),
  };

  Color _secondaryColor() => switch (widget.state) {
    OrbState.idle     => const Color(0xFF3B82F6),
    OrbState.listening => const Color(0xFFF97316),
    OrbState.thinking => const Color(0xFF10B981),
    OrbState.speaking => const Color(0xFF8B5CF6),
  };

  double _waveIntensity() => switch (widget.state) {
    OrbState.idle     => 0.03,
    OrbState.listening => 0.12,
    OrbState.thinking => 0.06,
    OrbState.speaking => 0.15,
  };

  @override
  void dispose() {
    _blinkTimer?.cancel();
    _waveController.dispose();
    _pulseController.dispose();
    _glowController.dispose();
    _blinkController.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _updateOrbit({double yawDelta = 0, double pitchDelta = 0}) {
    setState(() {
      _yaw += yawDelta;
      if (_yaw > pi) _yaw -= 2 * pi;
      if (_yaw < -pi) _yaw += 2 * pi;
      _pitch = (_pitch + pitchDelta).clamp(-pi / 3, pi / 3);
    });
  }

  KeyEventResult _onKeyEvent(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent) return KeyEventResult.ignored;
    switch (event.logicalKey) {
      case LogicalKeyboardKey.arrowLeft:
        _updateOrbit(yawDelta: -0.10);
        return KeyEventResult.handled;
      case LogicalKeyboardKey.arrowRight:
        _updateOrbit(yawDelta: 0.10);
        return KeyEventResult.handled;
      case LogicalKeyboardKey.arrowUp:
        _updateOrbit(pitchDelta: -0.08);
        return KeyEventResult.handled;
      case LogicalKeyboardKey.arrowDown:
        _updateOrbit(pitchDelta: 0.08);
        return KeyEventResult.handled;
      default:
        return KeyEventResult.ignored;
    }
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: Listenable.merge([
        _waveController,
        _pulseController,
        _glowController,
        _blinkController,
      ]),
      builder: (context, child) {
        final pulseScale = 1.0 + (_pulseController.value * 0.06);
        final glowIntensity = 0.3 + (_glowController.value * 0.5);
        final primary = _primaryColor();
        final secondary = _secondaryColor();

        return Focus(
          focusNode: _focusNode,
          autofocus: true,
          onKeyEvent: _onKeyEvent,
          child: GestureDetector(
            onPanStart: (_) => _focusNode.requestFocus(),
            onPanUpdate: (details) {
              _updateOrbit(
                yawDelta: details.delta.dx * 0.01,
                pitchDelta: details.delta.dy * 0.008,
              );
            },
            child: SizedBox(
              width: widget.size * 1.5,
              height: widget.size * 1.5,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  // Outer glow ring
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

                  // Face or orb
                  Transform.scale(
                    scale: pulseScale,
                    child: widget.avatarStyle == AvatarStyle.face
                        ? CustomPaint(
                            size: Size(widget.size, widget.size),
                            painter: _FacePainter(
                              wavePhase: _waveController.value * 2 * pi,
                              pulseValue: _pulseController.value,
                              glowValue: _glowController.value,
                              blinkValue: _blinkController.value,
                              primaryColor: primary,
                              secondaryColor: secondary,
                              state: widget.state,
                              yaw: _yaw,
                              pitch: _pitch,
                              isMini: widget.size < 90,
                            ),
                          )
                        : CustomPaint(
                            size: Size(widget.size, widget.size),
                            painter: _OrbPainter(
                              wavePhase: _waveController.value * 2 * pi,
                              waveIntensity: _waveIntensity(),
                              primaryColor: primary,
                              secondaryColor: secondary,
                              glowValue: _glowController.value,
                              yaw: _yaw,
                              pitch: _pitch,
                            ),
                          ),
                  ),

                  // State label (hidden in mini/idle mode)
                  if (!widget.avatarStyle.isFaceMini(widget.size))
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
            ),
          ),
        );
      },
    );
  }

  String _stateLabel() => switch (widget.state) {
    OrbState.idle      => '',
    OrbState.listening => 'LISTENING',
    OrbState.thinking  => 'THINKING',
    OrbState.speaking  => 'SPEAKING',
  };
}

extension _AvatarStyleExt on AvatarStyle {
  bool isFaceMini(double size) => this == AvatarStyle.face && size < 90;
}

// ─────────────────────────────────────────────────────────────────
// Face painter — holographic AI face with talking mouth animation
// ─────────────────────────────────────────────────────────────────
class _FacePainter extends CustomPainter {
  final double wavePhase;
  final double pulseValue;
  final double glowValue;
  final double blinkValue;
  final Color primaryColor;
  final Color secondaryColor;
  final OrbState state;
  final double yaw;
  final double pitch;
  final bool isMini;

  const _FacePainter({
    required this.wavePhase,
    required this.pulseValue,
    required this.glowValue,
    required this.blinkValue,
    required this.primaryColor,
    required this.secondaryColor,
    required this.state,
    required this.yaw,
    required this.pitch,
    this.isMini = false,
  });

  double get _mouthOpenness {
    if (state != OrbState.speaking) return 0.0;
    final a = sin(wavePhase * 2.3) * 0.50;
    final b = sin(wavePhase * 3.7 + 1.0) * 0.30;
    final c = sin(wavePhase * 1.1 + 2.0) * 0.20;
    return ((a + b + c + 1.0) / 2.0).clamp(0.0, 1.0);
  }

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;
    final r = size.width / 2;

    // Clip to circle
    canvas.clipPath(
      Path()..addOval(Rect.fromCircle(center: Offset(cx, cy), radius: r)),
    );

    if (isMini) {
      _drawMiniPulse(canvas, cx, cy, r);
      return;
    }

    // Parallax offsets from head rotation
    final px = sin(yaw) * r * 0.11;
    final py = -sin(pitch) * r * 0.07;

    // Face anchor (slightly above center — shows more face, less neck)
    final faceCx = cx + px * 0.45;
    final faceCy = cy - r * 0.04 + py * 0.45;

    _drawBackground(canvas, cx, cy, r, faceCx, faceCy);
    _drawHairShadow(canvas, faceCx, faceCy, r);
    _drawFaceGlow(canvas, faceCx, faceCy, r);
    _drawEyebrows(canvas, faceCx, faceCy, r, px, py);
    _drawEyes(canvas, faceCx, faceCy, r, px, py);
    _drawNose(canvas, faceCx, faceCy, r, px, py);
    _drawMouth(canvas, faceCx, faceCy, r, px, py);
    _drawRim(canvas, size, cx, cy, r);
  }

  // When tiny (56 px), just draw a glowing pulse dot
  void _drawMiniPulse(Canvas canvas, double cx, double cy, double r) {
    final paint = Paint()
      ..color = primaryColor.withAlpha((180 + glowValue * 50).toInt())
      ..maskFilter = MaskFilter.blur(BlurStyle.normal, r * 0.25);
    canvas.drawCircle(Offset(cx, cy), r * 0.45, paint);
    canvas.drawCircle(
      Offset(cx, cy),
      r * 0.22,
      Paint()..color = Colors.white.withAlpha(200),
    );
  }

  void _drawBackground(
    Canvas canvas, double cx, double cy, double r, double fcx, double fcy) {
    // Deep dark base
    canvas.drawCircle(
      Offset(cx, cy),
      r,
      Paint()..color = const Color(0xFF070714),
    );
    // Subtle ambient glow in state color
    canvas.drawCircle(
      Offset(fcx, fcy - r * 0.1),
      r * 0.8,
      Paint()
        ..shader = RadialGradient(
          colors: [
            primaryColor.withAlpha(22),
            Colors.transparent,
          ],
          radius: 1.0,
        ).createShader(Rect.fromCircle(center: Offset(fcx, fcy), radius: r * 0.8)),
    );
  }

  void _drawHairShadow(Canvas canvas, double fcx, double fcy, double r) {
    // Dark oval at top simulating hair / crown shadow
    final hairCenter = Offset(fcx, fcy - r * 0.62);
    canvas.drawOval(
      Rect.fromCenter(center: hairCenter, width: r * 1.28, height: r * 0.80),
      Paint()
        ..shader = RadialGradient(
          colors: [
            const Color(0xFF12082A).withAlpha(230),
            Colors.transparent,
          ],
        ).createShader(
          Rect.fromCenter(center: hairCenter, width: r * 1.28, height: r * 0.80),
        ),
    );
  }

  void _drawFaceGlow(Canvas canvas, double fcx, double fcy, double r) {
    // Subtle face-area highlight (cheekbones / forehead)
    canvas.drawOval(
      Rect.fromCenter(center: Offset(fcx, fcy + r * 0.05), width: r * 1.10, height: r * 1.30),
      Paint()
        ..shader = RadialGradient(
          center: const Alignment(0, -0.2),
          radius: 0.75,
          colors: [
            Colors.white.withAlpha(10),
            Colors.transparent,
          ],
        ).createShader(
          Rect.fromCircle(center: Offset(fcx, fcy), radius: r * 0.8),
        ),
    );
  }

  void _drawEyebrows(
      Canvas canvas, double fcx, double fcy, double r, double px, double py) {
    final eyeY = fcy - r * 0.195 + py;
    final spread = r * 0.265;
    final browLift = r * 0.130;
    final raise = state == OrbState.thinking ? r * 0.045 : 0.0;

    final paint = Paint()
      ..color = primaryColor.withAlpha(170)
      ..style = PaintingStyle.stroke
      ..strokeWidth = r * 0.048
      ..strokeCap = StrokeCap.round;

    for (final side in [-1.0, 1.0]) {
      final bx = fcx + side * spread + px;
      final by = eyeY - browLift - raise;
      // Arc that curves slightly upward at outer edge
      canvas.drawArc(
        Rect.fromCenter(center: Offset(bx, by), width: r * 0.42, height: r * 0.20),
        side < 0 ? pi + 0.25 : -0.25 - (pi - 0.5),
        (pi - 0.5) * side < 0 ? 1 : -1,
        false,
        paint,
      );
    }
  }

  void _drawEyes(
      Canvas canvas, double fcx, double fcy, double r, double px, double py) {
    final eyeY = fcy - r * 0.195 + py;
    final spread = r * 0.265;
    final eyeRx = r * 0.155;
    final eyeRy = max(0.5, r * 0.105 * (1.0 - blinkValue * 0.96));

    // Pupil drift: look around gently when thinking
    double pupilDx = 0.0;
    if (state == OrbState.thinking) {
      pupilDx = sin(wavePhase * 0.38) * r * 0.045;
    }

    for (final side in [-1.0, 1.0]) {
      final ex = fcx + side * spread + px;
      final ey = eyeY;
      final eyeRect = Rect.fromCenter(
        center: Offset(ex, ey), width: eyeRx * 2, height: eyeRy * 2);

      // Sclera
      canvas.drawOval(eyeRect, Paint()..color = const Color(0xFFE5E0F5));

      // Iris
      final irisR = min(eyeRx * 0.65, eyeRy * 0.92);
      final irisRect = Rect.fromCenter(
        center: Offset(ex + pupilDx, ey),
        width: irisR * 2,
        height: irisR * 2,
      );
      canvas.drawOval(irisRect, Paint()..color = primaryColor);

      // Pupil
      final pupilR = irisR * 0.50;
      canvas.drawOval(
        Rect.fromCenter(
          center: Offset(ex + pupilDx, ey),
          width: pupilR * 2,
          height: pupilR * 2,
        ),
        Paint()..color = const Color(0xFF06040F),
      );

      // Specular highlight
      canvas.drawOval(
        Rect.fromCenter(
          center: Offset(ex + pupilDx - irisR * 0.30, ey - irisR * 0.30),
          width: eyeRx * 0.28,
          height: eyeRy * 0.40,
        ),
        Paint()..color = Colors.white.withAlpha(210),
      );

      // Eye glow (state color)
      canvas.drawOval(
        eyeRect.inflate(3 + glowValue * 2),
        Paint()
          ..color = primaryColor.withAlpha((25 + glowValue * 18).toInt())
          ..maskFilter = const MaskFilter.blur(BlurStyle.outer, 5),
      );
    }
  }

  void _drawNose(
      Canvas canvas, double fcx, double fcy, double r, double px, double py) {
    final noseY = fcy + r * 0.075 + py * 0.5;
    final nosePaint = Paint()
      ..color = primaryColor.withAlpha(70);
    final gap = r * 0.065;
    canvas.drawCircle(Offset(fcx - gap + px * 0.6, noseY), r * 0.022, nosePaint);
    canvas.drawCircle(Offset(fcx + gap + px * 0.6, noseY), r * 0.022, nosePaint);
  }

  void _drawMouth(
      Canvas canvas, double fcx, double fcy, double r, double px, double py) {
    final mx = fcx + px * 0.70;
    final my = fcy + r * 0.305 + py * 0.55;
    final mw = r * 0.46;
    final open = _mouthOpenness;
    final mh = max(r * 0.022, open * r * 0.185);

    if (open > 0.05) {
      // ── Open mouth ──
      // Outer lip shape
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromCenter(center: Offset(mx, my), width: mw, height: mh),
          Radius.circular(mh * 0.5),
        ),
        Paint()..color = primaryColor.withAlpha(230),
      );
      // Dark interior
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromCenter(
            center: Offset(mx, my + mh * 0.08),
            width: mw * 0.82,
            height: mh * 0.58,
          ),
          Radius.circular(mh * 0.45),
        ),
        Paint()..color = const Color(0xFF040310),
      );
      // Teeth hint (white strip at top of opening)
      if (mh > r * 0.06) {
        canvas.drawRRect(
          RRect.fromRectAndRadius(
            Rect.fromCenter(
              center: Offset(mx, my - mh * 0.20),
              width: mw * 0.62,
              height: mh * 0.18,
            ),
            Radius.circular(r * 0.018),
          ),
          Paint()..color = Colors.white.withAlpha(180),
        );
      }
    } else {
      // ── Closed mouth — gentle smile ──
      final smilePath = Path();
      smilePath.moveTo(mx - mw / 2, my);
      smilePath.quadraticBezierTo(
        mx, my + r * 0.032, mx + mw / 2, my);
      canvas.drawPath(
        smilePath,
        Paint()
          ..color = primaryColor.withAlpha(190)
          ..style = PaintingStyle.stroke
          ..strokeWidth = r * 0.042
          ..strokeCap = StrokeCap.round,
      );
    }

    // Mouth glow
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromCenter(
          center: Offset(mx, my),
          width: mw + 6,
          height: max(mh, r * 0.04) + 6,
        ),
        Radius.circular(mh * 0.5 + 3),
      ),
      Paint()
        ..color = primaryColor.withAlpha((35 + glowValue * 30).toInt())
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 5),
    );
  }

  void _drawRim(Canvas canvas, Size size, double cx, double cy, double r) {
    canvas.drawCircle(
      Offset(cx, cy),
      r - 1.0,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..shader = SweepGradient(
          colors: [
            primaryColor.withAlpha(0),
            primaryColor.withAlpha((75 + glowValue * 45).toInt()),
            secondaryColor.withAlpha((55 + glowValue * 35).toInt()),
            primaryColor.withAlpha(0),
          ],
          stops: const [0.0, 0.3, 0.7, 1.0],
          transform: GradientRotation(wavePhase * 0.30),
        ).createShader(Rect.fromCircle(center: Offset(cx, cy), radius: r)),
    );
  }

  @override
  bool shouldRepaint(_FacePainter old) => true;
}

// ─────────────────────────────────────────────────────────────────
// Original orb painter (unchanged — kept as alternate style)
// ─────────────────────────────────────────────────────────────────
class _OrbPainter extends CustomPainter {
  final double wavePhase;
  final double waveIntensity;
  final Color primaryColor;
  final Color secondaryColor;
  final double glowValue;
  final double yaw;
  final double pitch;

  _OrbPainter({
    required this.wavePhase,
    required this.waveIntensity,
    required this.primaryColor,
    required this.secondaryColor,
    required this.glowValue,
    required this.yaw,
    required this.pitch,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2;
    final lightX = (-0.30 + sin(yaw) * 0.35).clamp(-0.65, 0.65);
    final lightY = (-0.30 - sin(pitch) * 0.35).clamp(-0.65, 0.65);

    final bgPaint = Paint()
      ..shader = RadialGradient(
        center: Alignment(lightX, lightY),
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

    const ringCount = 8;
    for (int i = 0; i < ringCount; i++) {
      final ringRadius = radius * (0.3 + (i / ringCount) * 0.65);
      final path = Path();
      const segments = 60;

      for (int j = 0; j <= segments; j++) {
        final angle = (j / segments) * 2 * pi;
        final wave1 = sin(angle * 3 + wavePhase + i * 0.7) * waveIntensity * radius;
        final wave2 = sin(angle * 5 - wavePhase * 1.3 + i * 0.4) * waveIntensity * radius * 0.5;
        final wave3 = cos(angle * 2 + wavePhase * 0.8 + i) * waveIntensity * radius * 0.3;
        final displacement = wave1 + wave2 + wave3;

        final r = ringRadius + displacement;
        final x0 = r * cos(angle);
        final y0 = r * sin(angle);

        final x1 = x0 * cos(yaw);
        final z1 = -x0 * sin(yaw);
        final y1 = y0 * cos(pitch) - z1 * sin(pitch);
        final z2 = y0 * sin(pitch) + z1 * cos(pitch);
        final perspective = 1.0 + (z2 / (radius * 3.5));

        final x = center.dx + x1 * perspective;
        final y = center.dy + y1 * perspective;

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

    final specCenter = Offset(center.dx - radius * 0.25, center.dy - radius * 0.25);
    final specPaint = Paint()
      ..shader = RadialGradient(
        colors: [
          Colors.white.withAlpha((40 + glowValue * 30).toInt()),
          Colors.white.withAlpha(0),
        ],
      ).createShader(Rect.fromCircle(center: specCenter, radius: radius * 0.3));
    canvas.drawCircle(specCenter, radius * 0.3, specPaint);

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
