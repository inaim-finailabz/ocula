/// Build-time environment configuration.
///
/// Values are injected via `--dart-define-from-file=.env.dev` or `.env.prod`.
/// Usage:
///   flutter run --dart-define-from-file=.env.dev
///   flutter build ios --dart-define-from-file=.env.prod
///
/// Access: `EnvConfig.modelServerUrl`, `EnvConfig.isDev`, etc.
class EnvConfig {
  const EnvConfig._();

  /// 'dev' or 'prod'
  static const env = String.fromEnvironment('ENV', defaultValue: 'prod');

  /// Model download server URL.
  static const modelServerUrl = String.fromEnvironment(
    'MODEL_SERVER_URL',
    defaultValue: 'https://backend-ocula.finailabz.com',
  );

  static bool get isDev => env == 'dev';
  static bool get isProd => env == 'prod';
}
