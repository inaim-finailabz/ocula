import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:url_launcher/url_launcher.dart';
import '../widgets/model_management.dart';
import '../screens/enterprise_settings.dart';
import '../screens/rag_settings.dart';
import '../services/speech_service.dart';
import '../services/network_permission.dart';
import '../services/app_language.dart';
import '../services/ai_manager.dart';
import '../services/model_manager.dart';
import '../services/local_data.dart';
import '../services/indexer.dart';
import '../services/notification_service.dart';
import '../services/ocula_db.dart';
import '../services/feedback_service.dart';
import '../services/env_config.dart';

/// Settings screen — voice customisation + privacy controls.
class SettingsScreen extends StatefulWidget {
  final SpeechService speech;

  const SettingsScreen({super.key, required this.speech});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late double _rate;
  late double _pitch;
  late double _volume;
  String? _selectedVoiceName;
  String? _selectedLanguage;

  List<Map<String, String>> _voices = [];
  List<String> _languages = [];
  bool _loading = true;

  final _network = NetworkPermission();
  final _feedback = FeedbackService();
  late InternetAccess _internetAccess;

  final _appLang = AppLanguage();
  late String _assistantLang;

  @override
  void initState() {
    super.initState();
    _rate = widget.speech.rate;
    _pitch = widget.speech.pitch;
    _volume = widget.speech.volume;
    _selectedVoiceName = widget.speech.voice?['name'];
    _selectedLanguage = widget.speech.language;
    _internetAccess = _network.access;
    _assistantLang = _appLang.assistantLanguage;
    _loadData();
  }

  Future<void> _loadData() async {
    final voices = await widget.speech.getVoices();
    final languages = await widget.speech.getLanguages();
    await _network.load();
    await _appLang.load();
    setState(() {
      _voices = voices;
      _languages = languages;
      _internetAccess = _network.access;
      _assistantLang = _appLang.assistantLanguage;
      _loading = false;
    });
  }

  /// Voices filtered by selected language.
  List<Map<String, String>> get _filteredVoices {
    if (_selectedLanguage == null) return _voices;
    return _voices.where((v) {
      final locale = v['locale'] ?? '';
      return locale.startsWith(_selectedLanguage!.split('-').first);
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              padding: const EdgeInsets.all(16),
              children: [
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // VOICE SETTINGS
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'Language'),
                const SizedBox(height: 8),
                _languageDropdown(colors),
                const SizedBox(height: 24),

                Row(
                  children: [
                    const Text(
                      'Voice',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const Spacer(),
                    TextButton.icon(
                      onPressed: () => widget.speech.preview(),
                      icon: const Icon(Icons.play_circle_outline, size: 18),
                      label: const Text('Preview'),
                      style: TextButton.styleFrom(
                        visualDensity: VisualDensity.compact,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                _voiceList(colors),
                const SizedBox(height: 16),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // CUSTOM VOICE UPLOAD
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                _customVoiceSection(colors),
                const SizedBox(height: 24),

                _SectionHeader(title: 'Speed', trailing: _rateLabel(_rate)),
                Slider(
                  value: _rate,
                  min: 0.1,
                  max: 1.0,
                  divisions: 9,
                  onChanged: (v) {
                    setState(() => _rate = v);
                    widget.speech.setRate(v);
                  },
                ),
                const SizedBox(height: 16),

                _SectionHeader(title: 'Pitch', trailing: _pitchLabel(_pitch)),
                Slider(
                  value: _pitch,
                  min: 0.5,
                  max: 2.0,
                  divisions: 15,
                  onChanged: (v) {
                    setState(() => _pitch = v);
                    widget.speech.setPitch(v);
                  },
                ),
                const SizedBox(height: 16),

                _SectionHeader(
                  title: 'Volume',
                  trailing: '${(_volume * 100).round()}%',
                ),
                Slider(
                  value: _volume,
                  min: 0.0,
                  max: 1.0,
                  divisions: 10,
                  onChanged: (v) {
                    setState(() => _volume = v);
                    widget.speech.setVolume(v);
                  },
                ),

                const SizedBox(height: 32),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // AI MODELS
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const ModelManagement(),
                const SizedBox(height: 32),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // ENTERPRISE SETTINGS
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'Enterprise Backend'),
                const EnterpriseSettings(),
                const SizedBox(height: 32),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // YOUR DATA — what's been indexed
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'Your Data'),
                const SizedBox(height: 4),
                Text(
                  'What Ocula has indexed and can search for you.',
                  style: TextStyle(
                    fontSize: 13,
                    color: colors.onSurface.withAlpha(120),
                  ),
                ),
                const SizedBox(height: 12),
                const _IndexStatsCard(),
                const SizedBox(height: 32),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // SEARCH TUNING
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'Search Tuning'),
                const SizedBox(height: 4),
                Text(
                  'Control how Ocula searches your data and generates responses.',
                  style: TextStyle(
                    fontSize: 13,
                    color: colors.onSurface.withAlpha(120),
                  ),
                ),
                const SizedBox(height: 12),
                const RagSettings(),
                const SizedBox(height: 32),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // INTERNET ACCESS
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'Internet Access'),
                const SizedBox(height: 4),
                Text(
                  'Ocula works fully offline by default. Enable internet access '
                  'to let it search the web when you ask.',
                  style: TextStyle(
                    fontSize: 13,
                    color: colors.onSurface.withAlpha(120),
                  ),
                ),
                const SizedBox(height: 12),
                _internetTile(
                  icon: Icons.wifi_off,
                  label: 'Never',
                  subtitle: 'Fully offline. No data leaves your device.',
                  value: InternetAccess.off,
                  colors: colors,
                ),
                _internetTile(
                  icon: Icons.help_outline,
                  label: 'Ask every time',
                  subtitle: 'Ocula asks permission before each web search.',
                  value: InternetAccess.askEveryTime,
                  colors: colors,
                ),
                _internetTile(
                  icon: Icons.wifi,
                  label: 'Always allow',
                  subtitle: 'Ocula can search the web whenever needed.',
                  value: InternetAccess.always,
                  colors: colors,
                ),

                const SizedBox(height: 32),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // EMAIL (IMAP) SETTINGS
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'Email Integration'),
                const SizedBox(height: 4),
                Text(
                  'Connect your email to let Ocula search your inbox. '
                  'Credentials stay on-device — emails are fetched directly via IMAP.',
                  style: TextStyle(
                    fontSize: 13,
                    color: colors.onSurface.withAlpha(120),
                  ),
                ),
                const SizedBox(height: 12),
                _EmailConfigTile(),

                const SizedBox(height: 32),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // NOTIFICATIONS
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'Notifications'),
                const SizedBox(height: 4),
                Text(
                  'Calendar reminders and daily schedule briefings. '
                  'All notifications are local — nothing leaves your device.',
                  style: TextStyle(
                    fontSize: 13,
                    color: colors.onSurface.withAlpha(120),
                  ),
                ),
                const SizedBox(height: 12),
                const _NotificationSettings(),

                const SizedBox(height: 32),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // ASSISTANT LANGUAGE
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'Assistant Language'),
                const SizedBox(height: 4),
                Text(
                  'The language Ocula speaks and responds in.',
                  style: TextStyle(
                    fontSize: 13,
                    color: colors.onSurface.withAlpha(120),
                  ),
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: AppLanguage.assistantLanguages.map((lang) {
                    final isSelected = lang == _assistantLang;
                    return ChoiceChip(
                      label: Text(lang),
                      selected: isSelected,
                      onSelected: (_) {
                        setState(() => _assistantLang = lang);
                        _appLang.setAssistantLanguage(lang);
                        // Also update TTS language to match
                        final ttsCode = _langToTtsCode(lang);
                        if (ttsCode != null) {
                          widget.speech.setLanguage(ttsCode);
                          setState(() => _selectedLanguage = ttsCode);
                        }
                      },
                    );
                  }).toList(),
                ),

                const SizedBox(height: 32),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // ABOUT OCULA
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'About Ocula'),
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: colors.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          ShaderMask(
                            shaderCallback: (bounds) => const LinearGradient(
                              colors: [Color(0xFF6C5CE7), Color(0xFF00CEC9)],
                            ).createShader(bounds),
                            child: const Text(
                              'Ocula',
                              style: TextStyle(
                                fontSize: 22,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 2,
                            ),
                            decoration: BoxDecoration(
                              color: colors.primary.withAlpha(30),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Text(
                              'v1.0.0',
                              style: TextStyle(
                                fontSize: 11,
                                color: colors.primary,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'By Finai Labz',
                        style: TextStyle(
                          fontSize: 14,
                          color: colors.primary,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 14),
                      Text(
                        'See. Hear. Reason.',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                          color: colors.onSurface.withAlpha(220),
                          fontStyle: FontStyle.italic,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        'Ocula is a private, on-device AI assistant that sees, hears, '
                        'and reasons — without ever touching the cloud. Your emails, '
                        'files, photos, and contacts never leave your device.',
                        style: TextStyle(
                          fontSize: 13,
                          height: 1.5,
                          color: colors.onSurface.withAlpha(160),
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 20),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // KEY FEATURES
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'Key Features'),
                const SizedBox(height: 8),
                _featureTile(
                  icon: Icons.visibility,
                  title: 'Vision AI',
                  subtitle:
                      'Point your camera and Ocula identifies, reads, and analyses what it sees.',
                  colors: colors,
                ),
                _featureTile(
                  icon: Icons.mic,
                  title: 'Voice Assistant',
                  subtitle:
                      'Talk naturally — Ocula listens, thinks, and speaks back.',
                  colors: colors,
                ),
                _featureTile(
                  icon: Icons.lock,
                  title: '100% Private',
                  subtitle:
                      'All AI models run on-device. Zero data sent to any server.',
                  colors: colors,
                ),
                _featureTile(
                  icon: Icons.wifi_off,
                  title: 'Works Offline',
                  subtitle:
                      'No internet required. Full functionality in airplane mode.',
                  colors: colors,
                ),
                _featureTile(
                  icon: Icons.picture_as_pdf,
                  title: 'PDF Export',
                  subtitle: 'Turn any AI analysis into a shareable PDF report.',
                  colors: colors,
                ),

                const SizedBox(height: 20),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // SUPPORT & CONTACT
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'Support'),
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: colors.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Need help or have feedback? We\'d love to hear from you.',
                        style: TextStyle(
                          fontSize: 13,
                          height: 1.4,
                          color: colors.onSurface.withAlpha(160),
                        ),
                      ),
                      const SizedBox(height: 14),
                      SizedBox(
                        width: double.infinity,
                        child: OutlinedButton.icon(
                          onPressed: _showFeedbackDialog,
                          icon: const Icon(Icons.feedback_outlined, size: 18),
                          label: const Text('Send Feedback'),
                        ),
                      ),
                      const SizedBox(height: 10),
                      _linkRow(
                        icon: Icons.email_outlined,
                        label: 'support@finailabz.com',
                        url: 'mailto:support@finailabz.com',
                        colors: colors,
                      ),
                      const SizedBox(height: 10),
                      _linkRow(
                        icon: Icons.language,
                        label: 'finailabz.com',
                        url: 'https://finailabz.com',
                        colors: colors,
                      ),
                      const SizedBox(height: 10),
                      _linkRow(
                        icon: Icons.article_outlined,
                        label: 'FAQ & Knowledge Base',
                        url: 'https://finailabz.com/support',
                        colors: colors,
                      ),
                      const SizedBox(height: 10),
                      _linkRow(
                        icon: Icons.privacy_tip_outlined,
                        label: 'Privacy Policy',
                        url: 'https://finailabz.com/privacy',
                        colors: colors,
                      ),
                      const SizedBox(height: 10),
                      _linkRow(
                        icon: Icons.description_outlined,
                        label: 'Terms of Service',
                        url: 'https://finailabz.com/terms',
                        colors: colors,
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 20),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // SOCIAL / FOLLOW US
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'Follow Us'),
                const SizedBox(height: 8),
                Row(
                  children: [
                    _socialChip(
                      'X / Twitter',
                      'https://x.com/finailabz',
                      colors,
                    ),
                    const SizedBox(width: 8),
                    _socialChip(
                      'LinkedIn',
                      'https://linkedin.com/company/finailabz',
                      colors,
                    ),
                    const SizedBox(width: 8),
                    _socialChip(
                      'GitHub',
                      'https://github.com/finailabz',
                      colors,
                    ),
                  ],
                ),

                const SizedBox(height: 20),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // ABOUT FINAI LABZ
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'About Finai Labz'),
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: colors.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Finai Labz builds AI tools that put privacy and user '
                        'ownership first. We believe the future of artificial '
                        'intelligence is on-device, offline, and in your hands.',
                        style: TextStyle(
                          fontSize: 13,
                          height: 1.5,
                          color: colors.onSurface.withAlpha(160),
                        ),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'Our mission: make powerful AI accessible to everyone — '
                        'without sacrificing privacy, requiring subscriptions to '
                        'cloud services, or sending your data to third parties.',
                        style: TextStyle(
                          fontSize: 13,
                          height: 1.5,
                          color: colors.onSurface.withAlpha(140),
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 20),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // LEGAL
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                Center(
                  child: Text(
                    '\u00A9 2026 Finai Labz. All rights reserved.',
                    style: TextStyle(
                      fontSize: 11,
                      color: colors.onSurface.withAlpha(80),
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // ADVANCED TOOLS
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                _buildAdvancedSection(),

                const SizedBox(height: 24),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // RESET
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                Center(
                  child: TextButton(
                    onPressed: _resetDefaults,
                    child: Text(
                      'Reset to Defaults',
                      style: TextStyle(color: colors.error),
                    ),
                  ),
                ),
                const SizedBox(height: 24),
              ],
            ),
    );
  }

  Widget _buildAdvancedSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.engineering, color: Colors.orange),
                const SizedBox(width: 10),
                Text(
                  'Advanced',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 15),
            Text(
              'Tools for advanced users and troubleshooting',
              style: Theme.of(
                context,
              ).textTheme.bodySmall?.copyWith(color: Colors.grey[600]),
            ),
            const SizedBox(height: 15),
            _buildAdvancedTile(
              'Clear Memory',
              'Unload AI models from RAM',
              Icons.memory,
              () => _showClearMemoryDialog(),
            ),
            _buildAdvancedTile(
              'Clear Model Files',
              'Delete downloaded models to free space',
              Icons.delete_sweep,
              () => _showClearFilesDialog(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAdvancedTile(
    String title,
    String subtitle,
    IconData icon,
    VoidCallback onTap,
  ) {
    return ListTile(
      leading: Icon(icon, color: Colors.red[400]),
      title: Text(title),
      subtitle: Text(subtitle),
      trailing: const Icon(Icons.arrow_forward_ios, size: 16),
      onTap: onTap,
    );
  }

  void _showClearMemoryDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear Memory'),
        content: const Text(
          'This will unload all AI models from memory. '
          'You may need to wait for models to reload when using AI features again.\n\n'
          'Use this if the app is using too much memory.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(context);
              await _clearMemory();
            },
            child: const Text('Clear', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  void _showClearFilesDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear Model Files'),
        content: const Text(
          'This will delete all downloaded AI models from storage. '
          'The app will need to re-download models when you use AI features.\n\n'
          'Use this to free up storage space.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(context);
              await _clearModelFiles();
            },
            child: const Text('Delete', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  Future<void> _clearMemory() async {
    try {
      // Show loading
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (context) => const AlertDialog(
          content: Row(
            children: [
              CircularProgressIndicator(),
              SizedBox(width: 20),
              Text('Clearing memory...'),
            ],
          ),
        ),
      );

      await AIManager().clearMemory();

      Navigator.pop(context); // Close loading dialog

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Memory cleared successfully'),
          backgroundColor: Colors.green,
        ),
      );
    } catch (e) {
      Navigator.pop(context); // Close loading dialog
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to clear memory: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Future<void> _clearModelFiles() async {
    try {
      // Show loading
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (context) => const AlertDialog(
          content: Row(
            children: [
              CircularProgressIndicator(),
              SizedBox(width: 20),
              Text('Deleting model files...'),
            ],
          ),
        ),
      );

      final success = await OculaModelManager().clearModelFiles();

      Navigator.pop(context); // Close loading dialog

      if (success) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Model files deleted successfully'),
            backgroundColor: Colors.green,
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Failed to delete model files'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } catch (e) {
      Navigator.pop(context); // Close loading dialog
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red),
      );
    }
  }

  // Voice defaults — Natural preset applied automatically on init.
  // Custom voice upload section below lets users personalize further.

  // ── Language Dropdown ──

  Widget _languageDropdown(ColorScheme colors) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: colors.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: _languages.contains(_selectedLanguage)
              ? _selectedLanguage
              : null,
          hint: const Text('Select language'),
          isExpanded: true,
          items: _languages.map((lang) {
            return DropdownMenuItem(value: lang, child: Text(lang));
          }).toList(),
          onChanged: (lang) {
            if (lang == null) return;
            setState(() {
              _selectedLanguage = lang;
              _selectedVoiceName = null;
            });
            widget.speech.setLanguage(lang);
          },
        ),
      ),
    );
  }

  // ── Voice List ──

  Widget _voiceList(ColorScheme colors) {
    if (_filteredVoices.isEmpty) {
      return Padding(
        padding: const EdgeInsets.all(12),
        child: Text(
          'No voices available for this language.',
          style: TextStyle(color: colors.onSurface.withAlpha(120)),
        ),
      );
    }

    return Container(
      constraints: const BoxConstraints(maxHeight: 200),
      decoration: BoxDecoration(
        color: colors.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: ListView.builder(
        shrinkWrap: true,
        itemCount: _filteredVoices.length,
        itemBuilder: (context, index) {
          final v = _filteredVoices[index];
          final name = v['name'] ?? 'Unknown';
          final locale = v['locale'] ?? '';
          final isSelected = name == _selectedVoiceName;

          return ListTile(
            dense: true,
            title: Text(
              _formatVoiceName(name),
              style: TextStyle(
                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
            subtitle: Text(
              locale,
              style: TextStyle(
                fontSize: 12,
                color: colors.onSurface.withAlpha(100),
              ),
            ),
            trailing: isSelected
                ? Icon(Icons.check_circle, color: colors.primary, size: 20)
                : null,
            onTap: () {
              setState(() => _selectedVoiceName = name);
              widget.speech.setVoice(v);
            },
          );
        },
      ),
    );
  }

  // ── Custom Voice Upload ──

  Widget _customVoiceSection(ColorScheme colors) {
    final hasVoice = widget.speech.hasCustomVoice;
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: colors.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
        border: hasVoice
            ? Border.all(color: Colors.greenAccent.withAlpha(80))
            : null,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                hasVoice ? Icons.graphic_eq : Icons.upload_file,
                size: 20,
                color: hasVoice ? Colors.greenAccent : colors.onSurface,
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  hasVoice ? 'Custom voice uploaded' : 'Upload your voice',
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                    color: hasVoice ? Colors.greenAccent : colors.onSurface,
                  ),
                ),
              ),
              if (hasVoice)
                IconButton(
                  icon: Icon(
                    Icons.delete_outline,
                    size: 20,
                    color: colors.error,
                  ),
                  tooltip: 'Remove custom voice',
                  onPressed: () async {
                    await widget.speech.removeCustomVoice();
                    setState(() {});
                  },
                ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            hasVoice
                ? 'Your voice sample is stored locally for future voice cloning.'
                : 'Upload a voice recording (.wav, .m4a, .mp3) to personalize how Ocula speaks. '
                      'Your voice sample stays on-device.',
            style: TextStyle(
              fontSize: 12,
              color: colors.onSurface.withAlpha(120),
              height: 1.3,
            ),
          ),
          if (!hasVoice) ...[
            const SizedBox(height: 10),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: _pickCustomVoice,
                icon: const Icon(Icons.mic, size: 18),
                label: const Text('Choose voice file'),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Future<void> _pickCustomVoice() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['wav', 'm4a', 'mp3', 'aac', 'ogg', 'caf'],
      );
      if (result != null && result.files.single.path != null) {
        final file = File(result.files.single.path!);
        final saved = await widget.speech.saveCustomVoice(file);
        if (saved != null && mounted) {
          setState(() {});
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Voice sample saved successfully')),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Could not upload voice: $e')));
      }
    }
  }

  // ── Internet Access Tiles ──

  Widget _internetTile({
    required IconData icon,
    required String label,
    required String subtitle,
    required InternetAccess value,
    required ColorScheme colors,
  }) {
    final isSelected = _internetAccess == value;
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Material(
        color: isSelected
            ? colors.primary.withAlpha(30)
            : colors.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: () {
            setState(() => _internetAccess = value);
            _network.setAccess(value);
          },
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            child: Row(
              children: [
                Icon(
                  icon,
                  size: 22,
                  color: isSelected ? colors.primary : colors.onSurface,
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        label,
                        style: TextStyle(
                          fontWeight: isSelected
                              ? FontWeight.bold
                              : FontWeight.w500,
                          color: isSelected ? colors.primary : colors.onSurface,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        subtitle,
                        style: TextStyle(
                          fontSize: 12,
                          color: colors.onSurface.withAlpha(100),
                        ),
                      ),
                    ],
                  ),
                ),
                if (isSelected)
                  Icon(Icons.check_circle, color: colors.primary, size: 20),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ── Feature Tile ──

  Widget _featureTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required ColorScheme colors,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: colors.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: colors.primary.withAlpha(25),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, size: 20, color: colors.primary),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 14,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    style: TextStyle(
                      fontSize: 12,
                      color: colors.onSurface.withAlpha(120),
                      height: 1.3,
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

  // ── Link Row (tappable URL) ──

  Widget _linkRow({
    required IconData icon,
    required String label,
    required String url,
    required ColorScheme colors,
  }) {
    return InkWell(
      borderRadius: BorderRadius.circular(8),
      onTap: () => _openUrl(url),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          children: [
            Icon(icon, size: 18, color: colors.primary),
            const SizedBox(width: 10),
            Text(
              label,
              style: TextStyle(
                fontSize: 13,
                color: colors.primary,
                fontWeight: FontWeight.w500,
              ),
            ),
            const Spacer(),
            Icon(
              Icons.open_in_new,
              size: 14,
              color: colors.onSurface.withAlpha(60),
            ),
          ],
        ),
      ),
    );
  }

  // ── Social Chip ──

  Widget _socialChip(String label, String url, ColorScheme colors) {
    return ActionChip(
      label: Text(label, style: const TextStyle(fontSize: 12)),
      avatar: Icon(Icons.open_in_new, size: 14, color: colors.primary),
      onPressed: () => _openUrl(url),
    );
  }

  Future<void> _openUrl(String url) async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  Future<void> _showFeedbackDialog() async {
    final messageController = TextEditingController();
    var category = 'general';
    var sending = false;

    await showDialog(
      context: context,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (ctx, setDialogState) {
            return AlertDialog(
              title: const Text('Send Feedback'),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  DropdownButtonFormField<String>(
                    initialValue: category,
                    decoration: const InputDecoration(
                      labelText: 'Category',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                    items: const [
                      DropdownMenuItem(
                        value: 'general',
                        child: Text('General'),
                      ),
                      DropdownMenuItem(value: 'bug', child: Text('Bug')),
                      DropdownMenuItem(
                        value: 'feature',
                        child: Text('Feature'),
                      ),
                    ],
                    onChanged: sending
                        ? null
                        : (v) {
                            if (v == null) return;
                            setDialogState(() => category = v);
                          },
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: messageController,
                    enabled: !sending,
                    minLines: 4,
                    maxLines: 6,
                    decoration: const InputDecoration(
                      hintText: 'What should we improve?',
                      border: OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Sent to: ${EnvConfig.feedbackApiUrl}',
                    style: const TextStyle(fontSize: 11),
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: sending ? null : () => Navigator.pop(ctx),
                  child: const Text('Cancel'),
                ),
                FilledButton(
                  onPressed: sending
                      ? null
                      : () async {
                          final message = messageController.text.trim();
                          if (message.isEmpty) return;

                          setDialogState(() => sending = true);
                          try {
                            await _feedback.send(
                              message: message,
                              category: category,
                            );
                            if (ctx.mounted) Navigator.pop(ctx);
                            if (!mounted) return;
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text('Feedback sent. Thank you.'),
                              ),
                            );
                          } catch (e) {
                            setDialogState(() => sending = false);
                            if (!mounted) return;
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: Text('Could not send feedback: $e'),
                              ),
                            );
                          }
                        },
                  child: sending
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text('Send'),
                ),
              ],
            );
          },
        );
      },
    );
    messageController.dispose();
  }

  // ── Helpers ──

  String _rateLabel(double rate) {
    if (rate <= 0.3) return 'Slow';
    if (rate <= 0.6) return 'Normal';
    if (rate <= 0.8) return 'Fast';
    return 'Very Fast';
  }

  String _pitchLabel(double pitch) {
    if (pitch < 0.8) return 'Low';
    if (pitch <= 1.2) return 'Normal';
    if (pitch <= 1.6) return 'High';
    return 'Very High';
  }

  String _formatVoiceName(String name) {
    final parts = name.split('.');
    final last = parts.last;
    return last[0].toUpperCase() + last.substring(1);
  }

  String? _langToTtsCode(String lang) {
    const map = {
      'English': 'en-US',
      'French': 'fr-FR',
      'Spanish': 'es-ES',
      'German': 'de-DE',
      'Arabic': 'ar-SA',
      'Chinese': 'zh-CN',
      'Japanese': 'ja-JP',
      'Korean': 'ko-KR',
      'Portuguese': 'pt-BR',
      'Hindi': 'hi-IN',
      'Russian': 'ru-RU',
      'Italian': 'it-IT',
      'Dutch': 'nl-NL',
      'Turkish': 'tr-TR',
    };
    return map[lang];
  }

  void _resetDefaults() {
    setState(() {
      _rate = 0.5;
      _pitch = 1.0;
      _volume = 1.0;
    });
    widget.speech.setRate(0.5);
    widget.speech.setPitch(1.0);
    widget.speech.setVolume(1.0);
  }
}

// ── Section Header ──

/// Email IMAP configuration tile with presets for common providers.
class _EmailConfigTile extends StatefulWidget {
  @override
  State<_EmailConfigTile> createState() => _EmailConfigTileState();
}

class _EmailConfigTileState extends State<_EmailConfigTile> {
  final _local = LocalData();
  final _hostController = TextEditingController();
  final _portController = TextEditingController(text: '993');
  final _userController = TextEditingController();
  final _passController = TextEditingController();
  bool _configured = false;
  bool _testing = false;

  static const _presets = {
    'iCloud Mail': ('imap.mail.me.com', 993),
    'Gmail': ('imap.gmail.com', 993),
    'Outlook': ('outlook.office365.com', 993),
    'Yahoo': ('imap.mail.yahoo.com', 993),
  };

  @override
  void initState() {
    super.initState();
    _checkConfigured();
  }

  Future<void> _checkConfigured() async {
    final ok = await _local.isEmailConfigured;
    if (mounted) setState(() => _configured = ok);
  }

  @override
  void dispose() {
    _hostController.dispose();
    _portController.dispose();
    _userController.dispose();
    _passController.dispose();
    super.dispose();
  }

  void _applyPreset(String name) {
    final preset = _presets[name]!;
    _hostController.text = preset.$1;
    _portController.text = preset.$2.toString();
  }

  Future<void> _save() async {
    final host = _hostController.text.trim();
    final port = int.tryParse(_portController.text.trim()) ?? 993;
    final user = _userController.text.trim();
    final pass = _passController.text;

    if (host.isEmpty || user.isEmpty || pass.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please fill in all fields')),
      );
      return;
    }

    setState(() => _testing = true);

    await _local.saveEmailConfig(
      host: host,
      port: port,
      user: user,
      password: pass,
    );

    // Test fetch
    final emails = await _local.recentEmails(limit: 5);
    setState(() => _testing = false);

    if (emails.isNotEmpty) {
      // Trigger re-index to pick up emails
      Indexer().runFullIndex();
      if (mounted) {
        setState(() => _configured = true);
        Navigator.pop(context); // Close dialog
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Connected! Found ${emails.length} emails. Indexing...',
            ),
          ),
        );
      }
    } else {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Could not fetch emails. Check credentials.'),
          ),
        );
      }
    }
  }

  void _showConfigDialog() {
    showDialog(
      context: context,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (ctx, setDialogState) {
            final colors = Theme.of(ctx).colorScheme;
            return AlertDialog(
              title: const Text('Email (IMAP) Setup'),
              content: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Provider presets
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: _presets.keys.map((name) {
                        return ActionChip(
                          label: Text(
                            name,
                            style: const TextStyle(fontSize: 12),
                          ),
                          onPressed: () {
                            _applyPreset(name);
                            setDialogState(() {});
                          },
                        );
                      }).toList(),
                    ),
                    const SizedBox(height: 16),
                    TextField(
                      controller: _hostController,
                      decoration: const InputDecoration(
                        labelText: 'IMAP Server',
                        hintText: 'imap.mail.me.com',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: _portController,
                      decoration: const InputDecoration(
                        labelText: 'Port',
                        hintText: '993',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                      keyboardType: TextInputType.number,
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: _userController,
                      decoration: const InputDecoration(
                        labelText: 'Email',
                        hintText: 'you@icloud.com',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                      keyboardType: TextInputType.emailAddress,
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: _passController,
                      decoration: const InputDecoration(
                        labelText: 'Password / App Password',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                      obscureText: true,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'For iCloud/Gmail, use an App-Specific Password. '
                      'Credentials are stored only on this device.',
                      style: TextStyle(
                        fontSize: 11,
                        color: colors.onSurface.withAlpha(100),
                      ),
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(ctx),
                  child: const Text('Cancel'),
                ),
                FilledButton(
                  onPressed: _testing ? null : _save,
                  child: _testing
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text('Connect & Test'),
                ),
              ],
            );
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Material(
      color: colors.surfaceContainerHighest,
      borderRadius: BorderRadius.circular(12),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: _showConfigDialog,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          child: Row(
            children: [
              Icon(
                _configured ? Icons.check_circle : Icons.email_outlined,
                size: 22,
                color: _configured ? Colors.greenAccent : colors.onSurface,
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _configured ? 'Email connected' : 'Connect email account',
                      style: TextStyle(
                        fontWeight: FontWeight.w500,
                        color: _configured
                            ? Colors.greenAccent
                            : colors.onSurface,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _configured
                          ? 'Tap to change IMAP settings'
                          : 'iCloud, Gmail, Outlook, Yahoo',
                      style: TextStyle(
                        fontSize: 12,
                        color: colors.onSurface.withAlpha(100),
                      ),
                    ),
                  ],
                ),
              ),
              Icon(Icons.chevron_right, color: colors.onSurface.withAlpha(60)),
            ],
          ),
        ),
      ),
    );
  }
}

/// Shows what Ocula has indexed — contacts, photos, calendar, files, emails.
class _IndexStatsCard extends StatefulWidget {
  const _IndexStatsCard();

  @override
  State<_IndexStatsCard> createState() => _IndexStatsCardState();
}

class _IndexStatsCardState extends State<_IndexStatsCard> {
  Map<String, int>? _sources;
  int _totalChunks = 0;
  int _knowledgeTriples = 0;
  bool _reindexing = false;

  @override
  void initState() {
    super.initState();
    _loadStats();
  }

  Future<void> _loadStats() async {
    final db = OculaDB();
    final sources = await db.sourceBreakdown();
    final stats = await db.stats();
    if (mounted) {
      setState(() {
        _sources = sources;
        _totalChunks = stats['rag_chunks'] ?? 0;
        _knowledgeTriples = stats['knowledge_triples'] ?? 0;
      });
    }
  }

  Future<void> _reindex() async {
    setState(() => _reindexing = true);
    await Indexer().runFullIndex();
    await _loadStats();
    if (mounted) setState(() => _reindexing = false);
  }

  static const _sourceConfig = <String, (IconData, String)>{
    'contact': (Icons.people, 'Contacts'),
    'photo': (Icons.photo_library, 'Photos'),
    'calendar': (Icons.event, 'Calendar Events'),
    'file': (Icons.insert_drive_file, 'Files'),
    'email': (Icons.email, 'Emails'),
    'chat': (Icons.chat_bubble, 'Conversations'),
  };

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    if (_sources == null) {
      return const Center(child: CircularProgressIndicator());
    }

    final entries = _sourceConfig.entries.where((e) {
      return (_sources![e.key] ?? 0) > 0;
    }).toList();

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: colors.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Source rows
          if (entries.isEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: Text(
                'No data indexed yet. Ocula will index your phone assets automatically.',
                style: TextStyle(
                  fontSize: 13,
                  color: colors.onSurface.withAlpha(120),
                ),
              ),
            )
          else
            for (final entry in entries)
              _statRow(
                icon: entry.value.$1,
                label: entry.value.$2,
                count: _sources![entry.key] ?? 0,
                colors: colors,
              ),

          if (_totalChunks > 0) ...[
            const Divider(height: 20),
            Row(
              children: [
                Text(
                  '$_totalChunks indexed items',
                  style: TextStyle(
                    fontSize: 12,
                    color: colors.onSurface.withAlpha(100),
                  ),
                ),
                if (_knowledgeTriples > 0)
                  Text(
                    '  ·  $_knowledgeTriples knowledge facts',
                    style: TextStyle(
                      fontSize: 12,
                      color: colors.onSurface.withAlpha(100),
                    ),
                  ),
              ],
            ),
          ],
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: _reindexing ? null : _reindex,
              icon: _reindexing
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.refresh, size: 18),
              label: Text(_reindexing ? 'Re-indexing...' : 'Re-index Now'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _statRow({
    required IconData icon,
    required String label,
    required int count,
    required ColorScheme colors,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Icon(icon, size: 18, color: colors.primary),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              label,
              style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
            decoration: BoxDecoration(
              color: colors.primary.withAlpha(25),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              '$count',
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.bold,
                color: colors.primary,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String title;
  final String? trailing;

  const _SectionHeader({required this.title, this.trailing});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(
          title,
          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
        if (trailing != null) ...[
          const Spacer(),
          Text(
            trailing!,
            style: TextStyle(
              fontSize: 14,
              color: Theme.of(context).colorScheme.primary,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ],
    );
  }
}

/// Notification settings — calendar reminders + daily briefing toggles.
class _NotificationSettings extends StatefulWidget {
  const _NotificationSettings();

  @override
  State<_NotificationSettings> createState() => _NotificationSettingsState();
}

class _NotificationSettingsState extends State<_NotificationSettings> {
  final _notif = NotificationService();
  bool _enabled = true;
  bool _briefingEnabled = true;
  int _reminderMinutes = 30;
  int _briefingHour = 8;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final enabled = await _notif.isEnabled;
    final briefing = await _notif.isBriefingEnabled;
    final minutes = await _notif.reminderMinutes;
    final hour = await _notif.briefingHour;
    if (mounted) {
      setState(() {
        _enabled = enabled;
        _briefingEnabled = briefing;
        _reminderMinutes = minutes;
        _briefingHour = hour;
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) return const SizedBox.shrink();

    final colors = Theme.of(context).colorScheme;

    return Column(
      children: [
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          title: const Text('Calendar Reminders'),
          subtitle: Text('Notify $_reminderMinutes min before events'),
          value: _enabled,
          activeColor: colors.primary,
          onChanged: (v) {
            setState(() => _enabled = v);
            _notif.setEnabled(v);
          },
        ),
        if (_enabled)
          Padding(
            padding: const EdgeInsets.only(left: 16),
            child: Row(
              children: [
                const Text('Remind me', style: TextStyle(fontSize: 13)),
                const SizedBox(width: 8),
                DropdownButton<int>(
                  value: _reminderMinutes,
                  underline: const SizedBox.shrink(),
                  style: TextStyle(fontSize: 13, color: colors.primary),
                  items: const [
                    DropdownMenuItem(value: 10, child: Text('10 min')),
                    DropdownMenuItem(value: 15, child: Text('15 min')),
                    DropdownMenuItem(value: 30, child: Text('30 min')),
                    DropdownMenuItem(value: 60, child: Text('1 hour')),
                  ],
                  onChanged: (v) {
                    if (v != null) {
                      setState(() => _reminderMinutes = v);
                      _notif.setReminderMinutes(v);
                    }
                  },
                ),
                const Text(' before events', style: TextStyle(fontSize: 13)),
              ],
            ),
          ),
        const Divider(height: 1),
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          title: const Text('Daily Briefing'),
          subtitle: Text('Schedule overview at ${_formatHour(_briefingHour)}'),
          value: _briefingEnabled,
          activeColor: colors.primary,
          onChanged: (v) {
            setState(() => _briefingEnabled = v);
            _notif.setBriefingEnabled(v);
          },
        ),
        if (_briefingEnabled)
          Padding(
            padding: const EdgeInsets.only(left: 16),
            child: Row(
              children: [
                const Text('Briefing time', style: TextStyle(fontSize: 13)),
                const SizedBox(width: 8),
                DropdownButton<int>(
                  value: _briefingHour,
                  underline: const SizedBox.shrink(),
                  style: TextStyle(fontSize: 13, color: colors.primary),
                  items: List.generate(
                    24,
                    (i) =>
                        DropdownMenuItem(value: i, child: Text(_formatHour(i))),
                  ),
                  onChanged: (v) {
                    if (v != null) {
                      setState(() => _briefingHour = v);
                      _notif.setBriefingHour(v);
                    }
                  },
                ),
              ],
            ),
          ),
      ],
    );
  }

  String _formatHour(int hour) {
    final h = hour > 12 ? hour - 12 : (hour == 0 ? 12 : hour);
    final amPm = hour >= 12 ? 'PM' : 'AM';
    return '$h:00 $amPm';
  }
}
