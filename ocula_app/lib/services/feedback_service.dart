import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';

import 'env_config.dart';

class FeedbackService {
  Future<void> send({
    required String message,
    String category = 'general',
    int? rating,
    String source = 'ocula_ai_assistant_client',
    String? sessionId,
    String? conversationId,
    String? userEmail,
    String? userName,
    List<String>? tags,
    Map<String, dynamic>? extra,
  }) async {
    final trimmed = message.trim();
    if (trimmed.isEmpty) {
      throw ArgumentError('Feedback message cannot be empty.');
    }

    final uri = Uri.parse(EnvConfig.feedbackApiUrl);
    final payload = <String, dynamic>{
      'message': trimmed,
      'rating': rating,
      'category': category,
      'source': source,
      if (sessionId != null && sessionId.isNotEmpty) 'session_id': sessionId,
      if (conversationId != null && conversationId.isNotEmpty)
        'conversation_id': conversationId,
      if (userEmail != null && userEmail.isNotEmpty) 'user_email': userEmail,
      if (userName != null && userName.isNotEmpty) 'user_name': userName,
      if (tags?.isNotEmpty ?? false) 'tags': tags,
      'metadata': {
        'timestamp': DateTime.now().toUtc().toIso8601String(),
        'platform': defaultTargetPlatform.name,
        'env': EnvConfig.env,
      },
      ...?extra,
    };

    final client = HttpClient()..connectionTimeout = const Duration(seconds: 8);
    try {
      final req = await client.postUrl(uri);
      req.headers.contentType = ContentType.json;
      req.headers.set(HttpHeaders.acceptHeader, 'application/json');
      if (EnvConfig.feedbackBearerToken.isNotEmpty) {
        req.headers.set(
          HttpHeaders.authorizationHeader,
          'Bearer ${EnvConfig.feedbackBearerToken}',
        );
      } else if (EnvConfig.feedbackApiKey.isNotEmpty) {
        req.headers.set('X-Feedback-Key', EnvConfig.feedbackApiKey);
      }
      req.write(jsonEncode(payload));

      final res = await req.close();
      final code = res.statusCode;
      if (code < 200 || code >= 300) {
        final body = await utf8.decoder.bind(res).join();
        throw HttpException(
          'Feedback API error ($code): ${body.isEmpty ? 'empty body' : body}',
          uri: uri,
        );
      }
    } finally {
      client.close(force: true);
    }
  }
}
