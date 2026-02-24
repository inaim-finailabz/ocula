import 'package:flutter/material.dart';
import '../services/ocula_db.dart';
import 'session_detail_screen.dart';

/// Bottom-sheet that lists all saved chat sessions from SQLite.
class SessionsScreen extends StatefulWidget {
  /// Called when the user taps "New Chat" from inside a session detail.
  final VoidCallback? onStartNewChat;

  const SessionsScreen({super.key, this.onStartNewChat});

  @override
  State<SessionsScreen> createState() => _SessionsScreenState();
}

class _SessionsScreenState extends State<SessionsScreen> {
  late final Future<List<Map<String, dynamic>>> _sessionsFuture;

  @override
  void initState() {
    super.initState();
    _sessionsFuture = OculaDB().sessionsIndex();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return DraggableScrollableSheet(
      initialChildSize: 0.7,
      minChildSize: 0.4,
      maxChildSize: 0.95,
      expand: false,
      builder: (context, scrollController) {
        return Column(
          children: [
            // ── Handle + header ──
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 8, 4),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Center(
                          child: Container(
                            width: 36,
                            height: 4,
                            margin: const EdgeInsets.only(bottom: 14),
                            decoration: BoxDecoration(
                              color: colors.onSurface.withAlpha(50),
                              borderRadius: BorderRadius.circular(2),
                            ),
                          ),
                        ),
                        Text(
                          'Chat History',
                          style: Theme.of(context)
                              .textTheme
                              .titleMedium
                              ?.copyWith(fontWeight: FontWeight.bold),
                        ),
                      ],
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close),
                    onPressed: () => Navigator.of(context).pop(),
                  ),
                ],
              ),
            ),
            const Divider(height: 1),

            // ── Session list ──
            Expanded(
              child: FutureBuilder<List<Map<String, dynamic>>>(
                future: _sessionsFuture,
                builder: (context, snap) {
                  if (snap.connectionState != ConnectionState.done) {
                    return const Center(child: CircularProgressIndicator());
                  }
                  final sessions = snap.data ?? [];
                  if (sessions.isEmpty) {
                    return _EmptyState(colors: colors);
                  }
                  return ListView.separated(
                    controller: scrollController,
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    itemCount: sessions.length,
                    separatorBuilder: (context2, idx) =>
                        const Divider(height: 1, indent: 64),
                    itemBuilder: (context, i) {
                      final s = sessions[i];
                      return _SessionTile(
                        session: s,
                        colors: colors,
                        onTap: () {
                          Navigator.of(context).push(
                            MaterialPageRoute(
                              builder: (_) => SessionDetailScreen(
                                sessionId: s['session_id'] as String,
                                startedAt: s['started_at'] as String,
                                onStartNewChat: () {
                                  Navigator.of(context).pop(); // close detail
                                  Navigator.of(context).pop(); // close sheet
                                  widget.onStartNewChat?.call();
                                },
                              ),
                            ),
                          );
                        },
                      );
                    },
                  );
                },
              ),
            ),
          ],
        );
      },
    );
  }
}

// ── Session list tile ──

class _SessionTile extends StatelessWidget {
  final Map<String, dynamic> session;
  final ColorScheme colors;
  final VoidCallback onTap;

  const _SessionTile({
    required this.session,
    required this.colors,
    required this.onTap,
  });

  String _formatDate(String iso) {
    try {
      final dt = DateTime.parse(iso).toLocal();
      final now = DateTime.now();
      final diff = now.difference(dt);
      if (diff.inDays == 0) return 'Today';
      if (diff.inDays == 1) return 'Yesterday';
      if (diff.inDays < 7) {
        const days = [
          'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'
        ];
        return days[dt.weekday - 1];
      }
      return '${dt.day} ${_month(dt.month)} ${dt.year}';
    } catch (_) {
      return iso;
    }
  }

  String _formatTime(String iso) {
    try {
      final dt = DateTime.parse(iso).toLocal();
      final h = dt.hour.toString().padLeft(2, '0');
      final m = dt.minute.toString().padLeft(2, '0');
      return '$h:$m';
    } catch (_) {
      return '';
    }
  }

  String _month(int m) {
    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    return months[m - 1];
  }

  @override
  Widget build(BuildContext context) {
    final firstQuery = (session['first_query'] as String?) ?? 'Untitled';
    final startedAt = (session['started_at'] as String?) ?? '';
    final turnCount = session['turn_count'] as int? ?? 0;
    final preview = firstQuery.length > 72
        ? '${firstQuery.substring(0, 72)}…'
        : firstQuery;

    return ListTile(
      onTap: onTap,
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      leading: CircleAvatar(
        backgroundColor: colors.primaryContainer,
        child: Icon(Icons.chat_bubble_outline,
            size: 18, color: colors.onPrimaryContainer),
      ),
      title: Text(
        preview,
        maxLines: 2,
        overflow: TextOverflow.ellipsis,
        style: const TextStyle(fontSize: 14),
      ),
      subtitle: Padding(
        padding: const EdgeInsets.only(top: 2),
        child: Text(
          '${_formatDate(startedAt)} · ${_formatTime(startedAt)} · '
          '$turnCount ${turnCount == 1 ? "turn" : "turns"}',
          style: TextStyle(
            fontSize: 12,
            color: colors.onSurface.withAlpha(130),
          ),
        ),
      ),
      trailing: Icon(Icons.chevron_right,
          size: 18, color: colors.onSurface.withAlpha(80)),
    );
  }
}

// ── Empty state ──

class _EmptyState extends StatelessWidget {
  final ColorScheme colors;
  const _EmptyState({required this.colors});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.history_outlined,
              size: 56, color: colors.onSurface.withAlpha(60)),
          const SizedBox(height: 16),
          Text(
            'No saved sessions yet',
            style: TextStyle(
              fontSize: 16,
              color: colors.onSurface.withAlpha(120),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Start a conversation — each session is\nautomatically saved here.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 13,
              height: 1.5,
              color: colors.onSurface.withAlpha(80),
            ),
          ),
        ],
      ),
    );
  }
}
