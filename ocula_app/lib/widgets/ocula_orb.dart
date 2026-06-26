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
// Face painter — digitised wireframe face, white straight lines
// Think: Apple Vision Pro persona / holographic grid head.
// All features are drawn as line segments projected in 3-D via
// yaw/pitch rotation so the face appears to turn in space.
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

  // ── 3-D projection helpers ──────────────────────────────────────
  // Input: 3-D point (x, y, z) in a coordinate space where the
  // face is centred at origin and radius ~1.
  // Output: 2-D canvas offset after yaw (Y-axis) + pitch (X-axis)
  // rotation and a tiny perspective divide.
  Offset _project(double x, double y, double z, double cx, double cy, double r) {
    // Yaw (rotate around Y-axis)
    final x1 = x * cos(yaw) + z * sin(yaw);
    final z1 = -x * sin(yaw) + z * cos(yaw);
    // Pitch (rotate around X-axis)
    final y2 = y * cos(pitch) - z1 * sin(pitch);
    final z2 = y * sin(pitch) + z1 * cos(pitch);
    // Perspective divide — deeper field (0.28) gives stronger 3-D look
    final scale = 1.0 + z2 * 0.28;
    // Canvas Y is inverted vs math Y, so negate y2
    return Offset(cx + x1 * r * scale, cy - y2 * r * scale);
  }

  // Draw a polyline through a list of 3-D points.
  void _polyline(Canvas canvas, Paint paint, List<(double, double, double)> pts,
      double cx, double cy, double r) {
    if (pts.isEmpty) return;
    final path = Path();
    final first = _project(pts[0].$1, pts[0].$2, pts[0].$3, cx, cy, r);
    path.moveTo(first.dx, first.dy);
    for (int i = 1; i < pts.length; i++) {
      final p = _project(pts[i].$1, pts[i].$2, pts[i].$3, cx, cy, r);
      path.lineTo(p.dx, p.dy);
    }
    canvas.drawPath(path, paint);
  }

  // Build a 3-D ellipse arc as a list of sample points.
  // cx3/cy3/cz3 = centre; rx/ry = radii in local X/Y plane;
  // from/to = angle range in radians; steps = segments.
  List<(double, double, double)> _ellipseArc(
      double cx3, double cy3, double cz3,
      double rx, double ry,
      double from, double to,
      {int steps = 24}) {
    final pts = <(double, double, double)>[];
    for (int i = 0; i <= steps; i++) {
      final t = from + (to - from) * i / steps;
      pts.add((cx3 + rx * cos(t), cy3 + ry * sin(t), cz3));
    }
    return pts;
  }

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;
    final r = size.width / 2;

    canvas.clipPath(
      Path()..addOval(Rect.fromCircle(center: Offset(cx, cy), radius: r)),
    );

    // Background
    canvas.drawCircle(Offset(cx, cy), r, Paint()..color = const Color(0xFF040410));

    // Subtle inner glow matching state colour
    canvas.drawCircle(
      Offset(cx, cy),
      r * 0.85,
      Paint()
        ..shader = RadialGradient(colors: [
          primaryColor.withAlpha((18 + glowValue * 14).toInt()),
          Colors.transparent,
        ]).createShader(Rect.fromCircle(center: Offset(cx, cy), radius: r * 0.85)),
    );

    if (isMini) {
      // Mini: just a pulsing cross-hair dot
      final mp = Paint()
        ..color = primaryColor.withAlpha((180 + glowValue * 60).toInt())
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, r * 0.2);
      canvas.drawCircle(Offset(cx, cy), r * 0.40, mp);
      canvas.drawCircle(Offset(cx, cy), r * 0.18,
          Paint()..color = Colors.white.withAlpha(220));
      return;
    }

    // Face lives in a normalised [-1,1] coordinate system.
    // The head oval sits between y = -0.88 (top) and y = 0.88 (bottom).
    // z = 0 is the mid-plane; features have a small positive z (front).

    final lineColor = Colors.white.withAlpha((180 + glowValue * 55).toInt());
    final dimColor  = primaryColor.withAlpha((100 + glowValue * 60).toInt());
    final bright    = Paint()
      ..color = lineColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = size.width * 0.018
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    final dim = Paint()
      ..color = dimColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = size.width * 0.012
      ..strokeCap = StrokeCap.round;

    // ── Grid / topology lines on the head oval ──────────────────
    // Horizontal latitude rings (chin → forehead).
    const latitudes = [-0.75, -0.45, -0.10, 0.25, 0.58, 0.82];
    for (final lat in latitudes) {
      // At latitude y, the head oval has x-radius that varies elliptically.
      // Head: rx ≈ 0.70, ry ≈ 0.90 (taller than wide).
      final xR = 0.70 * sqrt(max(0.0, 1.0 - (lat / 0.90) * (lat / 0.90)));
      // z follows a hemisphere: z = sqrt(max(0, 1 - xR²/0.70² - y²/0.90²)) * 0.40
      // Approximate: z_front ≈ xR * 0.55
      final zFront = xR * 0.55;
      _polyline(canvas, dim,
          _ellipseArc(0, lat, zFront * 0.5, xR, 0, 0, 2 * pi, steps: 32),
          cx, cy, r * 0.92);
    }

    // Vertical meridian lines (left temple → right temple).
    const meridians = [-0.55, -0.28, 0.0, 0.28, 0.55];
    for (final mx3 in meridians) {
      // For each x position, sweep y from bottom to top along the oval surface.
      final pts = <(double, double, double)>[];
      const steps = 20;
      for (int i = 0; i <= steps; i++) {
        final y3 = -0.90 + 1.80 * i / steps;
        final xR2 = (mx3.abs() / 0.70);
        final yR2 = (y3.abs() / 0.90);
        if (xR2 * xR2 + yR2 * yR2 > 1.01) continue;
        final zFront2 = 0.40 * sqrt(max(0.0, 1.0 - xR2 * xR2 - yR2 * yR2));
        pts.add((mx3, y3, zFront2));
      }
      _polyline(canvas, dim, pts, cx, cy, r * 0.92);
    }

    // ── Eyebrow blink offset ─────────────────────────────────────
    final browRaise = state == OrbState.thinking ? 0.055 : 0.0;

    // ── Eyebrows (straight lines, slight tilt) ───────────────────
    for (final side in [-1.0, 1.0]) {
      final bx = side * 0.28;
      final by = 0.22 + browRaise;
      _polyline(canvas, bright, [
        (bx - side * 0.16, by - 0.025, 0.30),
        (bx + side * 0.06, by + 0.025, 0.30),
      ], cx, cy, r * 0.92);
    }

    // ── Eyes ────────────────────────────────────────────────────
    final eyeOpenY = max(0.004, 0.085 * (1.0 - blinkValue * 0.97));

    // Pupil gaze drift when thinking
    double gazeDx = 0.0;
    if (state == OrbState.thinking) {
      gazeDx = sin(wavePhase * 0.4) * 0.035;
    }

    for (final side in [-1.0, 1.0]) {
      final ex = side * 0.265;
      final ey = 0.125;
      final ez = 0.34;

      // Outer eye outline — almond shape (top arc + bottom arc).
      // Top arc (bright)
      _polyline(canvas, bright,
          _ellipseArc(ex, ey, ez, 0.155, eyeOpenY, pi, 2 * pi),
          cx, cy, r * 0.92);
      // Bottom arc (slightly dimmer)
      _polyline(canvas, dim,
          _ellipseArc(ex, ey, ez, 0.155, eyeOpenY, 0, pi),
          cx, cy, r * 0.92);

      // Iris ring
      final irisR = 0.085;
      _polyline(canvas, bright,
          _ellipseArc(ex + gazeDx, ey, ez + 0.01, irisR, min(irisR, eyeOpenY * 0.92), 0, 2 * pi),
          cx, cy, r * 0.92);

      // Pupil dot (two crossing lines — looks like a crosshair)
      final pd = _project(ex + gazeDx, ey, ez + 0.02, cx, cy, r * 0.92);
      final pd2 = _project(ex + gazeDx + 0.022, ey, ez + 0.02, cx, cy, r * 0.92);
      final pupilR2d = (pd2 - pd).distance;
      canvas.drawCircle(pd, pupilR2d * 0.55,
          Paint()..color = primaryColor.withAlpha((200 + glowValue * 55).toInt())
            ..maskFilter = MaskFilter.blur(BlurStyle.normal, pupilR2d * 0.4));
      canvas.drawCircle(pd, pupilR2d * 0.22,
          Paint()..color = Colors.white.withAlpha(230));
    }

    // ── Nose bridge (two vertical lines + tip arc) ───────────────
    _polyline(canvas, dim, [
      (-0.055, 0.05, 0.36),
      (-0.045, -0.12, 0.38),
    ], cx, cy, r * 0.92);
    _polyline(canvas, dim, [
      (0.055, 0.05, 0.36),
      (0.045, -0.12, 0.38),
    ], cx, cy, r * 0.92);
    // Nose tip arc
    _polyline(canvas, dim,
        _ellipseArc(0, -0.15, 0.40, 0.08, 0.04, 0, pi, steps: 12),
        cx, cy, r * 0.92);

    // ── Mouth ────────────────────────────────────────────────────
    final open = _mouthOpenness;
    final mouthY = -0.335;
    final mouthW = 0.230;
    final openH  = open * 0.130;

    if (open > 0.04) {
      // Upper lip
      _polyline(canvas, bright, [
        (-mouthW, mouthY + openH * 0.5, 0.34),
        (-mouthW * 0.5, mouthY + openH * 0.6, 0.36),
        (0.0, mouthY + openH * 0.5, 0.37),
        (mouthW * 0.5, mouthY + openH * 0.6, 0.36),
        (mouthW, mouthY + openH * 0.5, 0.34),
      ], cx, cy, r * 0.92);
      // Lower lip
      _polyline(canvas, bright, [
        (-mouthW, mouthY + openH * 0.5, 0.34),
        (-mouthW * 0.5, mouthY - openH * 0.8, 0.36),
        (0.0, mouthY - openH, 0.37),
        (mouthW * 0.5, mouthY - openH * 0.8, 0.36),
        (mouthW, mouthY + openH * 0.5, 0.34),
      ], cx, cy, r * 0.92);
      // Teeth lines (horizontal bars inside opening)
      if (open > 0.25) {
        final tY = mouthY - openH * 0.1;
        _polyline(canvas, dim, [
          (-mouthW * 0.7, tY, 0.365),
          (mouthW * 0.7, tY, 0.365),
        ], cx, cy, r * 0.92);
      }
    } else {
      // Closed — gentle curved smile
      _polyline(canvas, bright, [
        (-mouthW, mouthY, 0.34),
        (-mouthW * 0.4, mouthY - 0.022, 0.36),
        (0.0, mouthY - 0.030, 0.37),
        (mouthW * 0.4, mouthY - 0.022, 0.36),
        (mouthW, mouthY, 0.34),
      ], cx, cy, r * 0.92);
    }

    // ── Chin / jaw line ──────────────────────────────────────────
    _polyline(canvas, dim, [
      (-0.50, -0.50, 0.20),
      (-0.35, -0.72, 0.28),
      (0.0,   -0.82, 0.32),
      (0.35,  -0.72, 0.28),
      (0.50,  -0.50, 0.20),
    ], cx, cy, r * 0.92);

    // ── Cheekbones ───────────────────────────────────────────────
    for (final side in [-1.0, 1.0]) {
      _polyline(canvas, dim, [
        (side * 0.62, -0.20, 0.18),
        (side * 0.52, 0.05, 0.24),
        (side * 0.36, 0.12, 0.32),
      ], cx, cy, r * 0.92);
    }

    // ── Rim glow ─────────────────────────────────────────────────
    canvas.drawCircle(
      Offset(cx, cy),
      r - 1.0,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..shader = SweepGradient(
          colors: [
            primaryColor.withAlpha(0),
            primaryColor.withAlpha((80 + glowValue * 50).toInt()),
            secondaryColor.withAlpha((60 + glowValue * 40).toInt()),
            primaryColor.withAlpha(0),
          ],
          stops: const [0.0, 0.3, 0.7, 1.0],
          transform: GradientRotation(wavePhase * 0.3),
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
