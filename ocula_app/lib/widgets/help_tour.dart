import 'package:flutter/material.dart';

/// A single step in the help tour — points at a widget via its [GlobalKey].
class HelpStep {
  final GlobalKey targetKey;
  final String title;
  final String description;

  const HelpStep({
    required this.targetKey,
    required this.title,
    required this.description,
  });
}

/// Spotlight overlay that walks the user through key UI elements.
class HelpTour extends StatefulWidget {
  final List<HelpStep> steps;
  final VoidCallback onComplete;

  const HelpTour({super.key, required this.steps, required this.onComplete});

  @override
  State<HelpTour> createState() => _HelpTourState();
}

class _HelpTourState extends State<HelpTour> {
  int _step = 0;

  Rect? _getTargetRect() {
    final step = widget.steps[_step];
    final ctx = step.targetKey.currentContext;
    if (ctx == null) return null;
    final box = ctx.findRenderObject() as RenderBox?;
    if (box == null) return null;
    final offset = box.localToGlobal(Offset.zero);
    return offset & box.size;
  }

  void _next() {
    if (_step < widget.steps.length - 1) {
      setState(() => _step++);
    } else {
      widget.onComplete();
    }
  }

  void _skip() => widget.onComplete();

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;
    final targetRect = _getTargetRect();
    final step = widget.steps[_step];
    final isLast = _step == widget.steps.length - 1;
    final colors = Theme.of(context).colorScheme;

    return Material(
      color: Colors.transparent,
      child: Stack(
        children: [
          // Dark overlay with spotlight cutout
          if (targetRect != null)
            CustomPaint(
              size: screenSize,
              painter: _SpotlightPainter(
                targetRect: targetRect.inflate(8),
                color: colors.primary,
              ),
            )
          else
            Container(color: Colors.black.withAlpha(160)),

          // Tapping outside dismisses
          GestureDetector(
            onTap: _skip,
            behavior: HitTestBehavior.opaque,
            child: const SizedBox.expand(),
          ),

          // Tooltip card
          if (targetRect != null)
            _TooltipCard(
              screenSize: screenSize,
              targetRect: targetRect.inflate(8),
              step: step,
              stepIndex: _step,
              stepCount: widget.steps.length,
              isLast: isLast,
              onNext: _next,
              onSkip: _skip,
              colors: colors,
            ),
        ],
      ),
    );
  }
}

// ── Spotlight Painter ──

class _SpotlightPainter extends CustomPainter {
  final Rect targetRect;
  final Color color;

  const _SpotlightPainter({required this.targetRect, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final fullScreen = Rect.fromLTWH(0, 0, size.width, size.height);
    final rRect = RRect.fromRectAndRadius(targetRect, const Radius.circular(12));

    final overlay = Path()..addRect(fullScreen);
    final hole = Path()..addRRect(rRect);
    final cutout = Path.combine(PathOperation.difference, overlay, hole);

    canvas.drawPath(
      cutout,
      Paint()..color = Colors.black.withAlpha(180),
    );

    // Purple border around the highlighted area
    canvas.drawRRect(
      rRect,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5,
    );
  }

  @override
  bool shouldRepaint(_SpotlightPainter old) => old.targetRect != targetRect;
}

// ── Tooltip Card ──

class _TooltipCard extends StatelessWidget {
  final Size screenSize;
  final Rect targetRect;
  final HelpStep step;
  final int stepIndex;
  final int stepCount;
  final bool isLast;
  final VoidCallback onNext;
  final VoidCallback onSkip;
  final ColorScheme colors;

  const _TooltipCard({
    required this.screenSize,
    required this.targetRect,
    required this.step,
    required this.stepIndex,
    required this.stepCount,
    required this.isLast,
    required this.onNext,
    required this.onSkip,
    required this.colors,
  });

  @override
  Widget build(BuildContext context) {
    const cardWidth = 280.0;
    const cardPad = 16.0;
    const estimatedCardHeight = 160.0;
    const gap = 12.0;

    // Prefer placing card below target; fall back to above
    final spaceBelow = screenSize.height - targetRect.bottom;
    final placeBelow = spaceBelow >= estimatedCardHeight + gap;

    final top = placeBelow
        ? targetRect.bottom + gap
        : targetRect.top - estimatedCardHeight - gap;

    // Horizontal: centre on target, clamp to screen
    double left = targetRect.center.dx - cardWidth / 2;
    left = left.clamp(cardPad, screenSize.width - cardWidth - cardPad);

    return Positioned(
      top: top.clamp(cardPad, screenSize.height - estimatedCardHeight - cardPad),
      left: left,
      width: cardWidth,
      child: GestureDetector(
        onTap: () {}, // prevent tap-through to the dismiss handler
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: colors.surface,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: colors.primary.withAlpha(80)),
            boxShadow: [
              BoxShadow(
                color: colors.primary.withAlpha(40),
                blurRadius: 20,
                spreadRadius: 2,
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              // Step counter
              Text(
                '${stepIndex + 1} / $stepCount',
                style: TextStyle(
                  fontSize: 11,
                  color: colors.primary,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                step.title,
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.bold,
                  color: colors.onSurface,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                step.description,
                style: TextStyle(
                  fontSize: 13,
                  height: 1.4,
                  color: colors.onSurface.withAlpha(160),
                ),
              ),
              const SizedBox(height: 14),
              Row(
                children: [
                  TextButton(
                    onPressed: onSkip,
                    style: TextButton.styleFrom(
                      visualDensity: VisualDensity.compact,
                      padding: EdgeInsets.zero,
                      foregroundColor: colors.onSurface.withAlpha(120),
                    ),
                    child: const Text('Skip'),
                  ),
                  const Spacer(),
                  FilledButton(
                    onPressed: onNext,
                    style: FilledButton.styleFrom(
                      visualDensity: VisualDensity.compact,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 8,
                      ),
                    ),
                    child: Text(isLast ? 'Done' : 'Next'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
