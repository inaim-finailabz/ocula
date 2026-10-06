import 'dart:io';

import 'package:integration_test/integration_test_driver_extended.dart';

/// Saves screenshots requested by integration tests to build/screenshots/.
///
/// In-app capture comes back blank on iOS simulators, so the driver (which
/// runs on the host) grabs the real screen with `simctl` instead. Set
/// OCULA_SIM_UDID to target a specific simulator (default: booted).
Future<void> main() => integrationDriver(
  onScreenshot: (name, bytes, [args]) async {
    final path = 'build/screenshots/$name.png';
    await File(path).create(recursive: true);
    final udid = Platform.environment['OCULA_SIM_UDID'] ?? 'booted';
    final r = await Process.run('xcrun', [
      'simctl',
      'io',
      udid,
      'screenshot',
      path,
    ]);
    if (r.exitCode != 0) await File(path).writeAsBytes(bytes);
    return true;
  },
);
