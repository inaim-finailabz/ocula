import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Manages the app's UI language and the assistant's response language.
///
/// Two separate settings:
/// - [appLocale]: UI labels, buttons, hints
/// - [assistantLanguage]: Language Ocula responds in (injected into prompts)
class AppLanguage extends ChangeNotifier {
  static final AppLanguage _instance = AppLanguage._();
  factory AppLanguage() => _instance;
  AppLanguage._();

  static const _appLocaleKey = 'app_locale';
  static const _assistantLangKey = 'assistant_language';

  Locale _appLocale = const Locale('en');
  String _assistantLanguage = 'English';
  bool _loaded = false;

  Locale get appLocale => _appLocale;
  String get assistantLanguage => _assistantLanguage;

  /// Supported UI locales.
  static const supportedLocales = [
    Locale('en'),
    Locale('fr'),
    Locale('es'),
    Locale('de'),
    Locale('ar'),
    Locale('zh'),
    Locale('ja'),
    Locale('ko'),
    Locale('pt'),
    Locale('hi'),
    Locale('ru'),
    Locale('it'),
    Locale('nl'),
    Locale('tr'),
  ];

  /// Human-readable names for assistant response languages.
  static const assistantLanguages = [
    'English',
    'French',
    'Spanish',
    'German',
    'Arabic',
    'Chinese',
    'Japanese',
    'Korean',
    'Portuguese',
    'Hindi',
    'Russian',
    'Italian',
    'Dutch',
    'Turkish',
  ];

  /// Display names for UI locales.
  static String localeDisplayName(Locale locale) {
    switch (locale.languageCode) {
      case 'en': return 'English';
      case 'fr': return 'Fran\u00e7ais';
      case 'es': return 'Espa\u00f1ol';
      case 'de': return 'Deutsch';
      case 'ar': return '\u0627\u0644\u0639\u0631\u0628\u064a\u0629';
      case 'zh': return '\u4e2d\u6587';
      case 'ja': return '\u65e5\u672c\u8a9e';
      case 'ko': return '\ud55c\uad6d\uc5b4';
      case 'pt': return 'Portugu\u00eas';
      case 'hi': return '\u0939\u093f\u0928\u094d\u0926\u0940';
      case 'ru': return '\u0420\u0443\u0441\u0441\u043a\u0438\u0439';
      case 'it': return 'Italiano';
      case 'nl': return 'Nederlands';
      case 'tr': return 'T\u00fcrk\u00e7e';
      default: return locale.languageCode;
    }
  }

  Future<void> load() async {
    if (_loaded) return;
    final prefs = await SharedPreferences.getInstance();
    final code = prefs.getString(_appLocaleKey) ?? 'en';
    _appLocale = Locale(code);
    _assistantLanguage = prefs.getString(_assistantLangKey) ?? 'English';
    _loaded = true;
  }

  Future<void> setAppLocale(Locale locale) async {
    _appLocale = locale;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_appLocaleKey, locale.languageCode);
  }

  Future<void> setAssistantLanguage(String lang) async {
    _assistantLanguage = lang;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_assistantLangKey, lang);
  }

  /// Prompt prefix telling the LLM what language to respond in.
  String get promptPrefix {
    if (_assistantLanguage == 'English') return '';
    return 'Respond in $_assistantLanguage. ';
  }
}
