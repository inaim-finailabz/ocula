import 'package:flutter/material.dart';
import '../services/ocula_db.dart';

/// Read-only view of all turns in a past chat session.
class SessionDetailScreen extends StatelessWidget {
  final String sessionId;
  final String startedAt;
  final VoidCallback? onStartNewChat;

  const SessionDetailScreen({
    super.key,
    required this.sessionId,
    required this.startedAt,
    this.onStartNewChat,
  });

  String _formatTitle(String iso) {
    try {
      final dt = DateTime.parse(iso).toLocal();
      final now = DateTime.now();
      final diff = now.difference(dt);
      final time =
          '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
      if (diff.inDays == 0) return 'Today at $time';
      if (diff.inDays == 1) return 'Yesterday at $time';
      const months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
      ];
      return '${dt.day} ${months[dt.month - 1]} ${dt.year}';
    } catch (_) {
      return 'Session';
    }
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Text(_formatTitle(startedAt)),
        actions: [
          if (onStartNewChat != null)
            TextButton.icon(
              onPressed: onStartNewChat,
              icon: const Icon(Icons.add_comment_outlined, size: 18),
              label: const Text('New Chat'),
            ),
        ],
      ),
      body: FutureBuilder<List<Map<String, dynamic>>>(
        future: OculaDB().sessionTurns(sessionId),
        builder: (context, snap) {
          if (snap.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }

          final turns = snap.data ?? [];

          if (turns.isEmpty) {
            return Center(
              child: Text(
                'No messages in this session.',
                style: TextStyle(
                    color: colors.onSurface.withAlpha(120), fontSize: 15),
              ),
            );
          }

          return ListView.builder(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 120),
            itemCount: turns.length,
            itemBuilder: (context, i) {
              final turn = turns[i];
              final query = (turn['query'] as String?) ?? '';
              final response = (turn['response'] as String?) ?? '';
              final ts = (turn['created_at'] as String?) ?? '';

              return Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // ── User bubble ──
                  Align(
                    alignment: Alignment.centerRight,
                    child: _ChatBubble(
                      text: query,
                      isUser: true,
                      colors: colors,
                    ),
                  ),
                  const SizedBox(height: 8),

                  // ── AI bubble ──
                  Align(
                    alignment: Alignment.centerLeft,
                    child: _ChatBubble(
                      text: response,
                      isUser: false,
                      colors: colors,
                      timestamp: ts,
                    ),
                  ),

                  if (i < turns.length - 1) const SizedBox(height: 20),
                ],
              );
            },
          );
        },
      ),
      // Floating "New Chat" button at bottom
      floatingActionButton: onStartNewChat != null
          ? FloatingActionButton.extended(
              onPressed: onStartNewChat,
              icon: const Icon(Icons.add_comment_outlined),
              label: const Text('New Chat'),
            )
          : null,
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}

// ── Chat bubble widget ──

class _ChatBubble extends StatelessWidget {
  final String text;
  final bool isUser;
  final ColorScheme colors;
  final String? timestamp;

  const _ChatBubble({
    required this.text,
    required this.isUser,
    required this.colors,
    this.timestamp,
  });

  String _shortTime(String iso) {
    try {
      final dt = DateTime.parse(iso).toLocal();
      return '${dt.hour.toString().padLeft(2, '0')}:'
          '${dt.minute.toString().padLeft(2, '0')}';
    } catch (_) {
      return '';
    }
  }

  @override
  Widget build(BuildContext context) {
    final bg = isUser ? colors.primary : colors.surfaceContainerHighest;
    final fg = isUser ? colors.onPrimary : colors.onSurface;

    return ConstrainedBox(
      constraints: BoxConstraints(
        maxWidth: MediaQuery.of(context).size.width * 0.78,
      ),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(16),
            topRight: const Radius.circular(16),
            bottomLeft: Radius.circular(isUser ? 16 : 4),
            bottomRight: Radius.circular(isUser ? 4 : 16),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SelectableText(
              text,
              style: TextStyle(fontSize: 14, height: 1.5, color: fg),
            ),
            if (!isUser && timestamp != null && timestamp!.isNotEmpty) ...[
              const SizedBox(height: 4),
              Text(
                _shortTime(timestamp!),
                style: TextStyle(
                  fontSize: 11,
                  color: fg.withAlpha(100),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
