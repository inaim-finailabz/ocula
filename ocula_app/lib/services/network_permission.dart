import 'package:shared_preferences/shared_preferences.dart';

/// Controls when Ocula is allowed to access the internet.
///
/// By default: NO internet. Everything stays on-device.
/// The user can grant:
///   - [InternetAccess.off]       — Never (default). Fully offline.
///   - [InternetAccess.askEveryTime] — Ask before each web request.
///   - [InternetAccess.always]    — Permanent access until revoked.
enum InternetAccess { off, askEveryTime, always }

class NetworkPermission {
  static final NetworkPermission _instance = NetworkPermission._();
  factory NetworkPermission() => _instance;
  NetworkPermission._();

  static const _key = 'internet_access';

  InternetAccess _access = InternetAccess.off;
  bool _loaded = false;

  // Temp grant for a single query session
  bool _tempGranted = false;

  InternetAccess get access => _access;

  /// Whether internet is currently allowed (permanent or temp).
  bool get isAllowed => _access == InternetAccess.always || _tempGranted;

  /// Whether we need to ask the user before going online.
  bool get needsPrompt => _access == InternetAccess.askEveryTime && !_tempGranted;

  Future<void> load() async {
    if (_loaded) return;
    final prefs = await SharedPreferences.getInstance();
    final idx = prefs.getInt(_key) ?? 0;
    _access = InternetAccess.values[idx];
    _loaded = true;
  }

  Future<void> setAccess(InternetAccess value) async {
    _access = value;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_key, value.index);
  }

  /// Grant temporary access for the current query.
  void grantTemp() {
    _tempGranted = true;
  }

  /// Revoke temporary access (call after the query completes).
  void revokeTemp() {
    _tempGranted = false;
  }
}
