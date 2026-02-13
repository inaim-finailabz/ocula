import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/ai_manager.dart';
import '../services/model_manager.dart';

/// Enterprise settings — gated behind a valid enterprise API key.
///
/// Only enterprise customers with a valid key can access the full
/// configuration panel. Everyone else sees a locked prompt.
class EnterpriseSettings extends StatefulWidget {
  const EnterpriseSettings({super.key});

  @override
  State<EnterpriseSettings> createState() => _EnterpriseSettingsState();
}

class _EnterpriseSettingsState extends State<EnterpriseSettings> {
  // ── Gate state ──
  bool _isUnlocked = false;
  bool _validating = false;
  String? _keyError;

  // ── Config state (only used when unlocked) ──
  bool _isEnterpriseEnabled = false;
  String _modelPath = '';
  String _modelUrl = '';
  String _apiKey = '';
  bool _useLocalModel = true;
  bool _loading = true;

  late TextEditingController _enterpriseKeyController;
  late TextEditingController _serverUrlController;

  @override
  void initState() {
    super.initState();
    _enterpriseKeyController = TextEditingController();
    _serverUrlController = TextEditingController();
    _loadEnterpriseSettings();
  }

  @override
  void dispose() {
    _enterpriseKeyController.dispose();
    _serverUrlController.dispose();
    super.dispose();
  }

  Future<void> _loadEnterpriseSettings() async {
    final prefs = await SharedPreferences.getInstance();
    final savedKey = prefs.getString('enterprise_api_key') ?? '';
    final serverUrl = await OculaModelManager().getModelServerUrl();

    final unlocked = _isValidKey(savedKey);
    setState(() {
      _isUnlocked = unlocked;
      _apiKey = savedKey;
      _enterpriseKeyController.text = savedKey;
      _isEnterpriseEnabled = prefs.getBool('enterprise_enabled') ?? false;
      _modelPath = prefs.getString('enterprise_model_path') ?? '';
      _modelUrl = prefs.getString('enterprise_model_url') ?? '';
      _useLocalModel = prefs.getBool('enterprise_use_local') ?? true;
      _serverUrlController.text = serverUrl;
      _loading = false;
    });
  }

  /// Validate the enterprise key format.
  /// Delegates to the shared validator in [AIManager].
  bool _isValidKey(String key) => AIManager.isValidEnterpriseKey(key);

  /// Validate and unlock enterprise settings.
  Future<void> _validateKey() async {
    final key = _enterpriseKeyController.text.trim();
    setState(() { _validating = true; _keyError = null; });

    // Simulate brief validation delay (in production this would hit a server)
    await Future.delayed(const Duration(milliseconds: 400));

    if (!_isValidKey(key)) {
      setState(() {
        _validating = false;
        _keyError = 'Invalid enterprise key. Keys start with "ocula-ent-" '
            'and must be at least 32 characters.';
      });
      return;
    }

    // Key is valid — persist and unlock
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('enterprise_api_key', key);
    setState(() {
      _apiKey = key;
      _isUnlocked = true;
      _validating = false;
    });
    debugPrint('[Enterprise] Key validated — enterprise settings unlocked');
  }

  /// Revoke access and clear enterprise settings.
  Future<void> _revokeAccess() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('enterprise_enabled', false);
    await prefs.remove('enterprise_api_key');
    await prefs.remove('enterprise_model_path');
    await prefs.remove('enterprise_model_url');
    await prefs.remove('enterprise_use_local');

    // If currently on enterprise tier, switch back to free
    if (AIManager().activeTier == AITier.enterprise) {
      try { await AIManager().switchEngine(AITier.free); } catch (_) {}
    }

    setState(() {
      _isUnlocked = false;
      _isEnterpriseEnabled = false;
      _apiKey = '';
      _modelPath = '';
      _modelUrl = '';
      _useLocalModel = true;
      _enterpriseKeyController.clear();
    });
    debugPrint('[Enterprise] Access revoked — settings cleared');

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Enterprise access revoked'),
          backgroundColor: Colors.orange,
        ),
      );
    }
  }

  Future<void> _saveEnterpriseSettings() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('enterprise_enabled', _isEnterpriseEnabled);
    await prefs.setString('enterprise_model_path', _modelPath);
    await prefs.setString('enterprise_model_url', _modelUrl);
    await prefs.setString('enterprise_api_key', _apiKey);
    await prefs.setBool('enterprise_use_local', _useLocalModel);

    // Save model server URL
    final url = _serverUrlController.text.trim();
    await OculaModelManager().setModelServerUrl(
      url.isEmpty ? OculaModelManager.defaultModelServerUrl : url,
    );

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Enterprise settings saved'),
          backgroundColor: Colors.green,
        ),
      );
    }
  }

  Future<void> _testEnterpriseModel() async {
    if (!_isEnterpriseEnabled) {
      _showError('Enterprise mode is not enabled');
      return;
    }

    if (_useLocalModel && _modelPath.isEmpty) {
      _showError('Model path is required for local models');
      return;
    }

    if (!_useLocalModel && _modelUrl.isEmpty) {
      _showError('Model URL is required for remote models');
      return;
    }

    try {
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (context) => const AlertDialog(
          content: Row(
            children: [
              CircularProgressIndicator(),
              SizedBox(width: 20),
              Text('Testing enterprise model...'),
            ],
          ),
        ),
      );

      await AIManager().switchEngine(AITier.enterprise);

      Navigator.pop(context);

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Enterprise model loaded successfully'),
          backgroundColor: Colors.green,
        ),
      );
    } catch (e) {
      Navigator.pop(context);
      _showError('Failed to load enterprise model: $e');
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
  }

  // ════════════════════════════════════════════════════════════════════
  // BUILD
  // ════════════════════════════════════════════════════════════════════

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }

    final colors = Theme.of(context).colorScheme;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Header ──
            Row(
              children: [
                Icon(
                  _isUnlocked ? Icons.lock_open : Icons.lock,
                  color: _isUnlocked ? Colors.green : colors.onSurface.withAlpha(120),
                ),
                const SizedBox(width: 10),
                Text(
                  'Enterprise Configuration',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                if (_isUnlocked) ...[
                  const Spacer(),
                  TextButton.icon(
                    onPressed: _revokeAccess,
                    icon: const Icon(Icons.logout, size: 16),
                    label: const Text('Revoke'),
                    style: TextButton.styleFrom(foregroundColor: Colors.red),
                  ),
                ],
              ],
            ),
            const SizedBox(height: 10),
            Text(
              _isUnlocked
                  ? 'Enterprise access active. Configure your custom AI models below.'
                  : 'Enter your enterprise API key to unlock custom model configuration.',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Colors.grey[600],
              ),
            ),
            const SizedBox(height: 20),

            // ── API Key Gate ──
            if (!_isUnlocked) ...[
              TextField(
                controller: _enterpriseKeyController,
                decoration: InputDecoration(
                  hintText: 'ocula-ent-...',
                  border: const OutlineInputBorder(),
                  prefixIcon: const Icon(Icons.vpn_key),
                  labelText: 'Enterprise API Key',
                  errorText: _keyError,
                  suffixIcon: _validating
                      ? const Padding(
                          padding: EdgeInsets.all(12),
                          child: SizedBox(
                            width: 20, height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          ),
                        )
                      : null,
                ),
                obscureText: true,
                onSubmitted: (_) => _validateKey(),
              ),
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: _validating ? null : _validateKey,
                  icon: const Icon(Icons.lock_open, size: 18),
                  label: Text(_validating ? 'Validating...' : 'Unlock Enterprise'),
                ),
              ),
            ],

            // ── Full config (only when unlocked) ──
            if (_isUnlocked) ...[
              // ── Model Server URL ──
              Text(
                'Model Server',
                style: Theme.of(context).textTheme.titleSmall,
              ),
              const SizedBox(height: 8),
              TextField(
                controller: _serverUrlController,
                decoration: InputDecoration(
                  hintText: OculaModelManager.defaultModelServerUrl,
                  border: const OutlineInputBorder(),
                  prefixIcon: const Icon(Icons.dns_outlined),
                  labelText: 'Server URL',
                  suffixIcon: IconButton(
                    icon: const Icon(Icons.restore, size: 20),
                    tooltip: 'Reset to default',
                    onPressed: () {
                      _serverUrlController.text = OculaModelManager.defaultModelServerUrl;
                    },
                  ),
                ),
                keyboardType: TextInputType.url,
              ),
              const SizedBox(height: 4),
              Text(
                'Where AI models are downloaded from',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Colors.grey[600],
                ),
              ),
              const SizedBox(height: 20),

              // Enable Enterprise Mode
              SwitchListTile(
                title: const Text('Enable Enterprise Mode'),
                subtitle: const Text('Use custom models instead of default Ocula models'),
                value: _isEnterpriseEnabled,
                onChanged: (value) {
                  setState(() => _isEnterpriseEnabled = value);
                },
              ),

              if (_isEnterpriseEnabled) ...[
                const SizedBox(height: 16),

                // Model Type Selection
                Text(
                  'Model Deployment',
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                const SizedBox(height: 8),

                RadioListTile<bool>(
                  title: const Text('Local Model'),
                  subtitle: const Text('GGUF model file on device'),
                  value: true,
                  groupValue: _useLocalModel,
                  onChanged: (value) {
                    if (value != null) setState(() => _useLocalModel = value);
                  },
                ),

                RadioListTile<bool>(
                  title: const Text('Remote API'),
                  subtitle: const Text('Enterprise API endpoint'),
                  value: false,
                  groupValue: _useLocalModel,
                  onChanged: (value) {
                    if (value != null) setState(() => _useLocalModel = value);
                  },
                ),

                const SizedBox(height: 16),

                if (_useLocalModel) ...[
                  Text(
                    'Local Model Path',
                    style: Theme.of(context).textTheme.titleSmall,
                  ),
                  const SizedBox(height: 8),
                  TextField(
                    controller: TextEditingController(text: _modelPath),
                    decoration: const InputDecoration(
                      hintText: '/path/to/model.gguf',
                      border: OutlineInputBorder(),
                      prefixIcon: Icon(Icons.folder),
                    ),
                    onChanged: (value) => _modelPath = value,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Path to the GGUF model file on the device',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Colors.grey[600],
                    ),
                  ),
                ] else ...[
                  Text(
                    'API Configuration',
                    style: Theme.of(context).textTheme.titleSmall,
                  ),
                  const SizedBox(height: 8),
                  TextField(
                    controller: TextEditingController(text: _modelUrl),
                    decoration: const InputDecoration(
                      hintText: 'https://api.company.com/v1/chat',
                      border: OutlineInputBorder(),
                      prefixIcon: Icon(Icons.link),
                      labelText: 'API Endpoint',
                    ),
                    onChanged: (value) => _modelUrl = value,
                  ),
                ],

                const SizedBox(height: 20),

                // Action Buttons
                Row(
                  children: [
                    ElevatedButton(
                      onPressed: _saveEnterpriseSettings,
                      child: const Text('Save Settings'),
                    ),
                    const SizedBox(width: 12),
                    OutlinedButton(
                      onPressed: _testEnterpriseModel,
                      child: const Text('Test Model'),
                    ),
                  ],
                ),
              ],
            ],
          ],
        ),
      ),
    );
  }
}
