import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/ai_manager.dart';
import '../services/model_manager.dart';

/// Enterprise settings for configuring custom AI models on-device.
/// Allows enterprise clients to deploy their own models directly.
class EnterpriseSettings extends StatefulWidget {
  const EnterpriseSettings({super.key});

  @override
  State<EnterpriseSettings> createState() => _EnterpriseSettingsState();
}

class _EnterpriseSettingsState extends State<EnterpriseSettings> {
  bool _isEnterpriseEnabled = false;
  String _modelPath = '';
  String _modelUrl = '';
  String _apiKey = '';
  bool _useLocalModel = true;
  bool _loading = true;

  late TextEditingController _serverUrlController;

  @override
  void initState() {
    super.initState();
    _serverUrlController = TextEditingController();
    _loadEnterpriseSettings();
  }

  @override
  void dispose() {
    _serverUrlController.dispose();
    super.dispose();
  }

  Future<void> _loadEnterpriseSettings() async {
    final prefs = await SharedPreferences.getInstance();
    final serverUrl = await OculaModelManager().getModelServerUrl();
    setState(() {
      _isEnterpriseEnabled = prefs.getBool('enterprise_enabled') ?? false;
      _modelPath = prefs.getString('enterprise_model_path') ?? '';
      _modelUrl = prefs.getString('enterprise_model_url') ?? '';
      _apiKey = prefs.getString('enterprise_api_key') ?? '';
      _useLocalModel = prefs.getBool('enterprise_use_local') ?? true;
      _serverUrlController.text = serverUrl;
      _loading = false;
    });
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

    if (!_useLocalModel && (_modelUrl.isEmpty || _apiKey.isEmpty)) {
      _showError('Model URL and API key are required for remote models');
      return;
    }

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
              Text('Testing enterprise model...'),
            ],
          ),
        ),
      );

      // Test the enterprise model
      await AIManager().switchEngine(AITier.enterprise);
      
      Navigator.pop(context); // Close loading dialog
      
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Enterprise model loaded successfully'),
          backgroundColor: Colors.green,
        ),
      );
    } catch (e) {
      Navigator.pop(context); // Close loading dialog
      _showError('Failed to load enterprise model: $e');
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
      ),
    );
  }

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
            Row(
              children: [
                Icon(Icons.business, color: colors.primary),
                const SizedBox(width: 10),
                Text(
                  'Enterprise Configuration',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Text(
              'Configure custom AI models for enterprise deployment',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Colors.grey[600],
              ),
            ),
            const SizedBox(height: 20),

            // ── Model Server URL (always visible) ──
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
              'Where AI modes are downloaded from',
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
                  if (value != null) {
                    setState(() => _useLocalModel = value);
                  }
                },
              ),
              
              RadioListTile<bool>(
                title: const Text('Remote API'),
                subtitle: const Text('Enterprise API endpoint'),
                value: false,
                groupValue: _useLocalModel,
                onChanged: (value) {
                  if (value != null) {
                    setState(() => _useLocalModel = value);
                  }
                },
              ),
              
              const SizedBox(height: 16),
              
              if (_useLocalModel) ...[
                // Local Model Configuration
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
                // Remote API Configuration
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
                const SizedBox(height: 12),
                TextField(
                  controller: TextEditingController(text: _apiKey),
                  decoration: const InputDecoration(
                    hintText: 'sk-...',
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.key),
                    labelText: 'API Key',
                  ),
                  obscureText: true,
                  onChanged: (value) => _apiKey = value,
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
        ),
      ),
    );
  }
}
