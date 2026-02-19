import 'package:flutter/material.dart';
import '../services/ai_manager.dart';

/// Wraps any Pro/Plus feature. If the user hasn't unlocked the tier,
/// shows a paywall bottom sheet instead of executing the action.
class PaywallGate extends StatelessWidget {
  final AITier requiredTier;
  final Widget child;
  final VoidCallback onAllowed;

  const PaywallGate({
    super.key,
    required this.requiredTier,
    required this.child,
    required this.onAllowed,
  });

  static const _tierLabels = {
    AITier.free: 'Free',
    AITier.plus: 'Plus',
    AITier.pro: 'Pro',
  };

  static const _tierPrices = {
    AITier.plus: '\$2.99 one-time',
    AITier.pro: '\$4.99/month',
  };

  static const _tierFeatures = {
    AITier.plus: [
      'Object counting & spatial detail',
      'Receipt total reading',
      'PDF export',
    ],
    AITier.pro: [
      'Document & handwriting OCR',
      'Complex diagram analysis',
      'Batch processing',
      'Advanced reasoning ("Why" / "How")',
    ],
  };

  /// Check if the user has purchased this tier.
  /// Replace with your actual purchase verification logic.
  bool _isUnlocked() {
    return true;
  }

  void _showPaywall(BuildContext context) {
    final label = _tierLabels[requiredTier]!;
    final price = _tierPrices[requiredTier] ?? 'Free';
    final features = _tierFeatures[requiredTier] ?? [];

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) => Padding(
        padding: const EdgeInsets.fromLTRB(24, 32, 24, 48),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Tier badge
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: requiredTier == AITier.pro
                    ? Colors.deepPurple
                    : Colors.teal,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                '$label — $price',
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
            ),
            const SizedBox(height: 20),

            // Feature list
            ...features.map((f) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    children: [
                      const Icon(Icons.check_circle, color: Colors.green, size: 20),
                      const SizedBox(width: 8),
                      Expanded(child: Text(f)),
                    ],
                  ),
                )),
            const SizedBox(height: 24),

            // Purchase button
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: () async {
                  // TODO: Trigger StoreKit / Google Play purchase flow
                  // On success:
                  Navigator.pop(ctx);
                  final ai = AIManager();
                  await ai.switchEngine(requiredTier);
                  onAllowed();
                },
                child: Text('Unlock $label'),
              ),
            ),

            // Restore purchases
            TextButton(
              onPressed: () {
                // TODO: Restore purchases via StoreKit / Google Play
              },
              child: const Text('Restore Purchases'),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        if (requiredTier == AITier.free || _isUnlocked()) {
          onAllowed();
        } else {
          _showPaywall(context);
        }
      },
      child: child,
    );
  }
}
