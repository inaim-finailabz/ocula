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

  /// Feedback API endpoint.
  /// Example: https://backend-ocula.finailabz.com/feedback
  static const feedbackApiUrl = String.fromEnvironment(
    'FEEDBACK_API_URL',
    defaultValue: 'https://backend-ocula.finailabz.com/feedback',
  );

  /// Optional API key sent as `X-Feedback-Key`.
  static const feedbackApiKey = String.fromEnvironment(
    'FEEDBACK_API_KEY',
    defaultValue: '',
  );

  /// Optional Bearer token for feedback endpoint auth.
  static const feedbackBearerToken = String.fromEnvironment(
    'FEEDBACK_BEARER_TOKEN',
    defaultValue: '',
  );

  /// Enterprise key verification endpoint.
  static const enterpriseVerifyApiUrl = String.fromEnvironment(
    'ENTERPRISE_VERIFY_API_URL',
    defaultValue:
        'https://backend-ocula.finailabz.com/client/enterprise/verify-api-key',
  );

  static bool get isDev => env == 'dev';
  static bool get isProd => env == 'prod';
}
