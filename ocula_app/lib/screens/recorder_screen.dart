import 'dart:async';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/recorder_service.dart';
import '../services/export_service.dart';
import '../services/ai_manager.dart';

/// Full-screen meeting / lecture / notes recorder with AI summarization.
class RecorderScreen extends StatefulWidget {
  const RecorderScreen({super.key});

  @override
  State<RecorderScreen> createState() => _RecorderScreenState();
}

class _RecorderScreenState extends State<RecorderScreen> {
  late final RecorderService _recorder;
  final ExportService _exportService = ExportService();

  RecorderMode _mode = RecorderMode.meeting;
  bool _isRecording = false;
  bool _isSummarizing = false;
  String _transcript = '';
  String _summary = '';

  /// Label of the AI tier used for the last summarization.
  String? _summaryTierLabel;

  /// Timer for refreshing the elapsed-time display.
  Timer? _durationTimer;
  Duration _elapsed = Duration.zero;

  StreamSubscription<String>? _transcriptSub;

  @override
  void initState() {
    super.initState();
    _recorder = RecorderService();
  }

  @override
  void dispose() {
    _transcriptSub?.cancel();
    _durationTimer?.cancel();
    _recorder.dispose();
    super.dispose();
  }

  // ── Control ──

  Future<void> _startRecording() async {
    final result = await _recorder.start();

    if (result == RecorderStartResult.permissionPermanentlyDenied) {
      if (mounted) _showPermissionDialog(permanent: true);
      return;
    }
    if (result == RecorderStartResult.permissionDenied) {
      if (mounted) _showPermissionDialog(permanent: false);
      return;
    }
    if (result == RecorderStartResult.unavailable) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              'Speech recognition is unavailable on this device.',
            ),
          ),
        );
      }
      return;
    }

    _transcriptSub = _recorder.transcriptStream.listen((t) {
      if (mounted) setState(() => _transcript = t);
    });

    _durationTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() => _elapsed = _recorder.elapsed);
    });

    setState(() {
      _isRecording = true;
      _summary = '';
      _summaryTierLabel = null;
      _transcript = '';
      _elapsed = Duration.zero;
    });
  }

  void _showPermissionDialog({required bool permanent}) {
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Microphone Access Required'),
        content: Text(
          permanent
              ? 'Microphone permission was denied. '
                'Please open Settings and enable it for Ocula to record.'
              : 'Ocula needs microphone access to record. '
                'Please allow it when prompted.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          if (permanent)
            FilledButton(
              onPressed: () {
                Navigator.pop(ctx);
                openAppSettings();
              },
              child: const Text('Open Settings'),
            ),
        ],
      ),
    );
  }

  Future<void> _stopRecording() async {
    await _recorder.stop();
    _transcriptSub?.cancel();
    _durationTimer?.cancel();
    setState(() {
      _isRecording = false;
      _transcript = _recorder.fullTranscript;
      _elapsed = _recorder.elapsed;
    });
  }

  Future<void> _summarize() async {
    if (_recorder.fullTranscript.trim().isEmpty) return;
    setState(() {
      _isSummarizing = true;
      _summary = '';
      _summaryTierLabel = null;
    });
    try {
      final result = await _recorder.summarize(_mode);
      final tier = AIManager().activeTier;
      if (mounted) {
        setState(() {
          _summary = result;
          _summaryTierLabel = tier != null ? _tierLabel(tier) : null;
          _isSummarizing = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _summary = 'Error generating summary: $e';
          _isSummarizing = false;
        });
      }
    }
  }

  Future<void> _export(BuildContext context) async {
    if (_summary.isEmpty) return;
    final modeLabel = _modeLabel(_mode);
    final box = context.findRenderObject() as RenderBox?;
    final origin = box != null ? box.localToGlobal(Offset.zero) & box.size : null;
    await _exportService.exportAndShare(
      _summary,
      title: '$modeLabel Summary',
      origin: origin,
    );
  }

  // ── Helpers ──

  String _modeLabel(RecorderMode m) {
    switch (m) {
      case RecorderMode.meeting:
        return 'Meeting';
      case RecorderMode.lecture:
        return 'Lecture';
      case RecorderMode.notes:
        return 'Voice Notes';
    }
  }

  String _tierLabel(AITier tier) {
    switch (tier) {
      case AITier.pro:
        return 'Ocula Pro';
      case AITier.plus:
        return 'Ocula Plus';
      case AITier.free:
        return 'Ocula Lite';
      case AITier.enterprise:
        return 'Ocula Enterprise';
    }
  }

  String _formatDuration(Duration d) {
    final h = d.inHours;
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return h > 0 ? '$h:$m:$s' : '$m:$s';
  }

  // ── Build ──

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final hasTranscript = _transcript.trim().isNotEmpty;
    final hasSummary = _summary.isNotEmpty;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Recorder'),
        actions: [
          if (hasSummary)
            Builder(
              builder: (ctx) => IconButton(
                icon: const Icon(Icons.ios_share_outlined),
                tooltip: 'Export summary',
                onPressed: () => _export(ctx),
              ),
            ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            // ── Mode selector (hidden while recording) ──
            if (!_isRecording)
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
                child: SegmentedButton<RecorderMode>(
                  segments: const [
                    ButtonSegment(
                      value: RecorderMode.meeting,
                      icon: Icon(Icons.groups_outlined, size: 18),
                      label: Text('Meeting'),
                    ),
                    ButtonSegment(
                      value: RecorderMode.lecture,
                      icon: Icon(Icons.school_outlined, size: 18),
                      label: Text('Lecture'),
                    ),
                    ButtonSegment(
                      value: RecorderMode.notes,
                      icon: Icon(Icons.mic_none_rounded, size: 18),
                      label: Text('Notes'),
                    ),
                  ],
                  selected: {_mode},
                  onSelectionChanged: (s) =>
                      setState(() => _mode = s.first),
                  style: ButtonStyle(
                    visualDensity: VisualDensity.compact,
                  ),
                ),
              ),

            // ── Recording status bar ──
            if (_isRecording)
              Container(
                margin: const EdgeInsets.fromLTRB(16, 12, 16, 0),
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 10,
                ),
                decoration: BoxDecoration(
                  color: colors.errorContainer.withAlpha(60),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: colors.error.withAlpha(80),
                  ),
                ),
                child: Row(
                  children: [
                    _PulsingDot(color: colors.error),
                    const SizedBox(width: 10),
                    Text(
                      'Recording — ${_modeLabel(_mode)}',
                      style: TextStyle(
                        color: colors.error,
                        fontWeight: FontWeight.w600,
                        fontSize: 13,
                      ),
                    ),
                    const Spacer(),
                    Text(
                      _formatDuration(_elapsed),
                      style: TextStyle(
                        color: colors.error,
                        fontWeight: FontWeight.w600,
                        fontSize: 13,
                        fontFeatures: const [FontFeature.tabularFigures()],
                      ),
                    ),
                  ],
                ),
              ),

            const SizedBox(height: 12),

            // ── Scrollable content ──
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Live transcript card
                    if (hasTranscript || _isRecording) ...[
                      _SectionHeader(
                        label: 'Transcript',
                        trailing: hasTranscript && !_isRecording
                            ? Text(
                                '${_transcript.split(' ').where((w) => w.isNotEmpty).length} words',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: colors.onSurface.withAlpha(140),
                                ),
                              )
                            : null,
                      ),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: colors.surfaceContainerHighest,
                          borderRadius: BorderRadius.circular(12),
                          border: _isRecording
                              ? Border.all(
                                  color: colors.error.withAlpha(60),
                                )
                              : null,
                        ),
                        child: hasTranscript
                            ? SelectableText(
                                _transcript,
                                style: TextStyle(
                                  fontSize: 14,
                                  height: 1.6,
                                  color: colors.onSurface,
                                ),
                              )
                            : Text(
                                'Listening…',
                                style: TextStyle(
                                  fontSize: 14,
                                  fontStyle: FontStyle.italic,
                                  color: colors.onSurface.withAlpha(100),
                                ),
                              ),
                      ),
                      const SizedBox(height: 16),
                    ],

                    // Empty state
                    if (!hasTranscript && !_isRecording) ...[
                      const SizedBox(height: 40),
                      Center(
                        child: Column(
                          children: [
                            Icon(
                              Icons.record_voice_over_outlined,
                              size: 64,
                              color: colors.onSurface.withAlpha(60),
                            ),
                            const SizedBox(height: 16),
                            Text(
                              'Tap Start to begin recording',
                              style: TextStyle(
                                fontSize: 16,
                                color: colors.onSurface.withAlpha(120),
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Choose a mode above, then tap Start.\n'
                              'Ocula transcribes and summarises everything on-device.',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                fontSize: 13,
                                height: 1.5,
                                color: colors.onSurface.withAlpha(80),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 40),
                    ],

                    // Summary card
                    if (hasSummary || _isSummarizing) ...[
                      _SectionHeader(
                        label: 'Summary',
                        trailing: _summaryTierLabel != null
                            ? _TierBadge(label: _summaryTierLabel!)
                            : null,
                      ),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: colors.primaryContainer.withAlpha(40),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: colors.primary.withAlpha(50),
                          ),
                        ),
                        child: _isSummarizing
                            ? const Padding(
                                padding: EdgeInsets.symmetric(vertical: 24),
                                child: Center(
                                  child: CircularProgressIndicator(),
                                ),
                              )
                            : SelectableText(
                                _summary,
                                style: TextStyle(
                                  fontSize: 14,
                                  height: 1.6,
                                  color: colors.onSurface,
                                ),
                              ),
                      ),
                      const SizedBox(height: 16),
                    ],
                  ],
                ),
              ),
            ),

            // ── Bottom action bar ──
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
              child: Column(
                children: [
                  // Summarize button — visible when stopped + has transcript
                  if (!_isRecording && hasTranscript && !_isSummarizing)
                    SizedBox(
                      width: double.infinity,
                      child: OutlinedButton.icon(
                        icon: const Icon(Icons.auto_awesome, size: 18),
                        label: Text(
                          hasSummary
                              ? 'Re-generate Summary'
                              : 'Generate Summary',
                        ),
                        onPressed: _summarize,
                        style: OutlinedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 14),
                        ),
                      ),
                    ),
                  if (!_isRecording && hasTranscript && !_isSummarizing)
                    const SizedBox(height: 10),

                  // Start / Stop button
                  SizedBox(
                    width: double.infinity,
                    child: _isRecording
                        ? FilledButton.icon(
                            icon: const Icon(Icons.stop_circle_outlined),
                            label: const Text('Stop Recording'),
                            onPressed: _stopRecording,
                            style: FilledButton.styleFrom(
                              backgroundColor: colors.error,
                              foregroundColor: colors.onError,
                              padding:
                                  const EdgeInsets.symmetric(vertical: 16),
                            ),
                          )
                        : FilledButton.icon(
                            icon: const Icon(Icons.fiber_manual_record),
                            label: Text(
                              hasTranscript
                                  ? 'Record Again'
                                  : 'Start Recording',
                            ),
                            onPressed: _startRecording,
                            style: FilledButton.styleFrom(
                              padding:
                                  const EdgeInsets.symmetric(vertical: 16),
                            ),
                          ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Supporting widgets ──

class _SectionHeader extends StatelessWidget {
  final String label;
  final Widget? trailing;
  const _SectionHeader({required this.label, this.trailing});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: Theme.of(context).colorScheme.onSurface.withAlpha(160),
              letterSpacing: 0.5,
            ),
          ),
          if (trailing != null) ...[
            const Spacer(),
            trailing!,
          ],
        ],
      ),
    );
  }
}

class _TierBadge extends StatelessWidget {
  final String label;
  const _TierBadge({required this.label});

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: colors.primaryContainer.withAlpha(80),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: colors.primary.withAlpha(60)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.auto_awesome, size: 11, color: colors.primary),
          const SizedBox(width: 4),
          Text(
            label,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: colors.primary,
            ),
          ),
        ],
      ),
    );
  }
}

/// Pulsing red dot — used in the recording status bar.
class _PulsingDot extends StatefulWidget {
  final Color color;
  const _PulsingDot({required this.color});

  @override
  State<_PulsingDot> createState() => _PulsingDotState();
}

class _PulsingDotState extends State<_PulsingDot>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);
    _anim = Tween<double>(begin: 0.4, end: 1.0).animate(
      CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _anim,
      child: Container(
        width: 10,
        height: 10,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: widget.color,
        ),
      ),
    );
  }
}
