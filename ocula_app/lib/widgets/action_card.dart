import 'package:flutter/material.dart';
import '../services/action_service.dart';

/// Confirmation card shown in the chat when Ocula detects an action intent.
/// Handles contact resolution, user editing, and execution inline.
class ActionCard extends StatefulWidget {
  final ActionRequest request;

  const ActionCard({super.key, required this.request});

  @override
  State<ActionCard> createState() => _ActionCardState();
}

class _ActionCardState extends State<ActionCard> {
  late ActionRequest _req;
  bool _resolving = true;
  bool _executing = false;
  bool _done = false;
  String? _resultText;
  bool _editing = false;

  // Editing controllers
  late TextEditingController _titleCtrl;
  late TextEditingController _contactCtrl;
  late TextEditingController _bodyCtrl;
  late TextEditingController _whenCtrl;

  @override
  void initState() {
    super.initState();
    _req = widget.request;
    _titleCtrl = TextEditingController(text: _req.title ?? '');
    _contactCtrl = TextEditingController(text: _req.contactName ?? '');
    _bodyCtrl = TextEditingController(text: _req.body ?? '');
    _whenCtrl = TextEditingController(text: _req.when != null ? _fmtDateTime(_req.when!) : '');
    _resolveContact();
  }

  @override
  void dispose() {
    _titleCtrl.dispose();
    _contactCtrl.dispose();
    _bodyCtrl.dispose();
    _whenCtrl.dispose();
    super.dispose();
  }

  Future<void> _resolveContact() async {
    if (_req.type == ActionType.callContact ||
        _req.type == ActionType.textContact ||
        _req.type == ActionType.draftEmail) {
      await ActionService.resolveContact(_req);
    }
    if (mounted) setState(() => _resolving = false);
  }

  Future<void> _confirm() async {
    if (_editing) _applyEdits();
    setState(() => _executing = true);
    final result = await ActionService.execute(_req);
    if (mounted) {
      setState(() {
        _executing = false;
        _done = true;
        _resultText = result;
      });
    }
  }

  void _applyEdits() {
    _req = _req.copyWith(
      title: _titleCtrl.text.trim().isNotEmpty ? _titleCtrl.text.trim() : _req.title,
      contactName: _contactCtrl.text.trim().isNotEmpty ? _contactCtrl.text.trim() : _req.contactName,
      body: _bodyCtrl.text.trim().isNotEmpty ? _bodyCtrl.text.trim() : _req.body,
    );
    setState(() => _editing = false);
  }

  IconData get _icon {
    switch (_req.type) {
      case ActionType.createEvent:   return Icons.calendar_today_outlined;
      case ActionType.setReminder:   return Icons.notifications_none_outlined;
      case ActionType.callContact:   return Icons.phone_outlined;
      case ActionType.textContact:   return Icons.chat_bubble_outline;
      case ActionType.draftEmail:    return Icons.mail_outline;
    }
  }

  String get _label {
    switch (_req.type) {
      case ActionType.createEvent:   return 'Create Event';
      case ActionType.setReminder:   return 'Set Reminder';
      case ActionType.callContact:   return 'Call';
      case ActionType.textContact:   return 'Send Message';
      case ActionType.draftEmail:    return 'Draft Email';
    }
  }

  String get _confirmLabel {
    switch (_req.type) {
      case ActionType.createEvent:   return 'Add to Calendar';
      case ActionType.setReminder:   return 'Set Reminder';
      case ActionType.callContact:   return 'Call Now';
      case ActionType.textContact:   return 'Open Messages';
      case ActionType.draftEmail:    return 'Open Mail';
    }
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Container(
      margin: const EdgeInsets.only(top: 2, bottom: 4),
      decoration: BoxDecoration(
        color: colors.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: colors.primary.withAlpha(60)),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header bar
            Container(
              color: colors.primaryContainer.withAlpha(80),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              child: Row(
                children: [
                  Icon(_icon, size: 18, color: colors.primary),
                  const SizedBox(width: 8),
                  Text(
                    _label,
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 13,
                      color: colors.primary,
                    ),
                  ),
                ],
              ),
            ),

            Padding(
              padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
              child: _done ? _buildDone(colors) : _buildBody(colors),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDone(ColorScheme colors) {
    return Row(
      children: [
        Icon(Icons.check_circle_outline, size: 18, color: Colors.greenAccent),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            _resultText ?? 'Done.',
            style: TextStyle(fontSize: 13, color: colors.onSurface),
          ),
        ),
      ],
    );
  }

  Widget _buildBody(ColorScheme colors) {
    if (_editing) return _buildEditMode(colors);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (_resolving)
          Row(
            children: [
              SizedBox(
                width: 14, height: 14,
                child: CircularProgressIndicator(strokeWidth: 2, color: colors.primary),
              ),
              const SizedBox(width: 8),
              Text('Looking up contact…', style: TextStyle(fontSize: 12, color: colors.onSurface.withAlpha(160))),
            ],
          )
        else ...[
          Text(
            ActionService.summary(_req),
            style: TextStyle(fontSize: 14, color: colors.onSurface, height: 1.4),
          ),
          // Show contact detail if available
          if (_req.contactPhone != null || _req.contactEmail != null) ...[
            const SizedBox(height: 4),
            Text(
              _req.contactPhone ?? _req.contactEmail ?? '',
              style: TextStyle(fontSize: 12, color: colors.onSurface.withAlpha(150)),
            ),
          ],
          if (_req.body != null && _req.body!.isNotEmpty &&
              (_req.type == ActionType.textContact || _req.type == ActionType.draftEmail)) ...[
            const SizedBox(height: 4),
            Text(
              '"${_req.body!.length > 60 ? '${_req.body!.substring(0, 60)}…' : _req.body!}"',
              style: TextStyle(
                fontSize: 12,
                color: colors.onSurface.withAlpha(160),
                fontStyle: FontStyle.italic,
              ),
            ),
          ],
        ],

        const SizedBox(height: 14),

        // Action buttons
        if (_executing)
          Center(
            child: SizedBox(
              width: 20, height: 20,
              child: CircularProgressIndicator(strokeWidth: 2, color: colors.primary),
            ),
          )
        else
          Row(
            children: [
              // Cancel — outline style
              Expanded(
                child: OutlinedButton(
                  onPressed: () => setState(() => _done = true),
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    side: BorderSide(color: colors.outline.withAlpha(80)),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  ),
                  child: Text('Cancel', style: TextStyle(fontSize: 13, color: colors.onSurface.withAlpha(180))),
                ),
              ),
              const SizedBox(width: 8),
              // Edit
              if (!_resolving)
                IconButton(
                  icon: Icon(Icons.edit_outlined, size: 18, color: colors.onSurface.withAlpha(160)),
                  tooltip: 'Edit',
                  visualDensity: VisualDensity.compact,
                  padding: EdgeInsets.zero,
                  onPressed: () => setState(() => _editing = true),
                ),
              const SizedBox(width: 4),
              // Confirm — filled
              if (!_resolving)
                Expanded(
                  child: FilledButton(
                    onPressed: _confirm,
                    style: FilledButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 8),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                    ),
                    child: Text(_confirmLabel, style: const TextStyle(fontSize: 13)),
                  ),
                ),
            ],
          ),
      ],
    );
  }

  Widget _buildEditMode(ColorScheme colors) {
    final showTitle = _req.type == ActionType.createEvent ||
        _req.type == ActionType.setReminder ||
        _req.type == ActionType.draftEmail;
    final showContact = _req.type == ActionType.callContact ||
        _req.type == ActionType.textContact ||
        _req.type == ActionType.draftEmail;
    final showBody = _req.type == ActionType.textContact ||
        _req.type == ActionType.draftEmail;
    final showWhen = _req.type == ActionType.createEvent ||
        _req.type == ActionType.setReminder;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (showTitle) ...[
          _editField(_titleCtrl, showTitle ? 'Title' : 'Subject', colors),
          const SizedBox(height: 8),
        ],
        if (showContact) ...[
          _editField(_contactCtrl, 'Contact name', colors),
          const SizedBox(height: 8),
        ],
        if (showWhen) ...[
          _editField(_whenCtrl, 'When (e.g. tomorrow at 3pm)', colors),
          const SizedBox(height: 8),
        ],
        if (showBody) ...[
          _editField(_bodyCtrl, 'Message', colors, maxLines: 3),
          const SizedBox(height: 8),
        ],
        const SizedBox(height: 4),
        Row(
          children: [
            Expanded(
              child: OutlinedButton(
                onPressed: () => setState(() => _editing = false),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  side: BorderSide(color: colors.outline.withAlpha(80)),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                ),
                child: const Text('Cancel edit', style: TextStyle(fontSize: 13)),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: FilledButton(
                onPressed: _confirm,
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                ),
                child: Text(_confirmLabel, style: const TextStyle(fontSize: 13)),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _editField(TextEditingController ctrl, String hint, ColorScheme colors, {int maxLines = 1}) {
    return TextField(
      controller: ctrl,
      maxLines: maxLines,
      style: TextStyle(fontSize: 13, color: colors.onSurface),
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: TextStyle(fontSize: 12, color: colors.onSurface.withAlpha(120)),
        filled: true,
        fillColor: colors.surface,
        contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: colors.outline.withAlpha(60)),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: colors.outline.withAlpha(60)),
        ),
      ),
    );
  }

  static String _fmtDateTime(DateTime dt) {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final diff = DateTime(dt.year, dt.month, dt.day).difference(today).inDays;
    String date = diff == 0 ? 'today' : diff == 1 ? 'tomorrow' : _weekdayName(dt.weekday);
    final h = dt.hour, m = dt.minute;
    final ampm = h < 12 ? 'AM' : 'PM';
    final h12 = h == 0 ? 12 : (h > 12 ? h - 12 : h);
    final mStr = m == 0 ? '' : ':${m.toString().padLeft(2, '0')}';
    return '$date at $h12$mStr $ampm';
  }

  static String _weekdayName(int wd) =>
      const ['', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][wd];
}
