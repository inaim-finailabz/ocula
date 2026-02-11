import 'package:flutter/material.dart';
import '../services/speech_service.dart';
import '../services/network_permission.dart';
import '../services/app_language.dart';

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
        actions: [
          TextButton.icon(
            onPressed: () => widget.speech.preview(),
            icon: const Icon(Icons.play_arrow, size: 20),
            label: const Text('Preview'),
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              padding: const EdgeInsets.all(16),
              children: [
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // RECOMMENDED PRESETS
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'Recommended'),
                const SizedBox(height: 8),
                _presetRow(colors),
                const SizedBox(height: 24),

                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                // VOICE SETTINGS
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'Language'),
                const SizedBox(height: 8),
                _languageDropdown(colors),
                const SizedBox(height: 24),

                const _SectionHeader(title: 'Voice'),
                const SizedBox(height: 8),
                _voiceList(colors),
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
                // ABOUT
                // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                const _SectionHeader(title: 'About'),
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
                          const Text(
                            'Ocula',
                            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                          ),
                          const SizedBox(width: 8),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                            decoration: BoxDecoration(
                              color: colors.primary.withAlpha(30),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Text(
                              'v1.0.0',
                              style: TextStyle(fontSize: 11, color: colors.primary),
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
                      const SizedBox(height: 12),
                      Text(
                        'See. Hear. Reason.',
                        style: TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w500,
                          color: colors.onSurface.withAlpha(200),
                          fontStyle: FontStyle.italic,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Ocula is your private AI assistant that lives entirely on '
                        'your device. No cloud. No data transfer. Your emails, files, '
                        'photos, and contacts stay yours \u2014 always.',
                        style: TextStyle(
                          fontSize: 13,
                          height: 1.4,
                          color: colors.onSurface.withAlpha(150),
                        ),
                      ),
                      const SizedBox(height: 12),
                      Divider(color: colors.onSurface.withAlpha(30)),
                      const SizedBox(height: 8),
                      Text(
                        'Finai Labz builds AI tools that put privacy and '
                        'user ownership first. We believe the future of AI is '
                        'on-device, offline, and in your hands.',
                        style: TextStyle(
                          fontSize: 12,
                          height: 1.4,
                          color: colors.onSurface.withAlpha(120),
                        ),
                      ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Icon(Icons.language, size: 16, color: colors.primary),
                          const SizedBox(width: 6),
                          Text(
                            'finailabz.com',
                            style: TextStyle(
                              fontSize: 13,
                              color: colors.primary,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 32),

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

  // ── Recommended Presets ──

  static const _presets = [
    _VoicePreset(
      name: 'Calm',
      icon: Icons.self_improvement,
      rate: 0.4,
      pitch: 0.9,
      volume: 0.8,
    ),
    _VoicePreset(
      name: 'Natural',
      icon: Icons.record_voice_over,
      rate: 0.5,
      pitch: 1.0,
      volume: 1.0,
    ),
    _VoicePreset(
      name: 'Energetic',
      icon: Icons.bolt,
      rate: 0.65,
      pitch: 1.2,
      volume: 1.0,
    ),
    _VoicePreset(
      name: 'Fast Read',
      icon: Icons.speed,
      rate: 0.8,
      pitch: 1.0,
      volume: 1.0,
    ),
  ];

  Widget _presetRow(ColorScheme colors) {
    return SizedBox(
      height: 90,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: _presets.length,
        separatorBuilder: (_, __) => const SizedBox(width: 10),
        itemBuilder: (context, i) {
          final p = _presets[i];
          final isActive =
              (_rate - p.rate).abs() < 0.05 &&
              (_pitch - p.pitch).abs() < 0.05 &&
              (_volume - p.volume).abs() < 0.05;

          return GestureDetector(
            onTap: () => _applyPreset(p),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              width: 85,
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: isActive
                    ? colors.primary.withAlpha(40)
                    : colors.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(14),
                border: isActive
                    ? Border.all(color: colors.primary, width: 2)
                    : null,
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(p.icon, size: 28, color: isActive ? colors.primary : colors.onSurface),
                  const SizedBox(height: 6),
                  Text(
                    p.name,
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: isActive ? FontWeight.bold : FontWeight.normal,
                      color: isActive ? colors.primary : colors.onSurface,
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  void _applyPreset(_VoicePreset p) {
    setState(() {
      _rate = p.rate;
      _pitch = p.pitch;
      _volume = p.volume;
    });
    widget.speech.setRate(p.rate);
    widget.speech.setPitch(p.pitch);
    widget.speech.setVolume(p.volume);
  }

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
              style: TextStyle(fontSize: 12, color: colors.onSurface.withAlpha(100)),
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
                Icon(icon, size: 22, color: isSelected ? colors.primary : colors.onSurface),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        label,
                        style: TextStyle(
                          fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
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

// ── Voice Preset Model ──

class _VoicePreset {
  final String name;
  final IconData icon;
  final double rate;
  final double pitch;
  final double volume;

  const _VoicePreset({
    required this.name,
    required this.icon,
    required this.rate,
    required this.pitch,
    required this.volume,
  });
}

// ── Section Header ──

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
