# Mobile AI Model Strategy 2026

## 1. Tiers & Models

| Tier | Model | Size | Use Case | Load Trigger |
|------|-------|------|----------|--------------|
| **Free** | SmolVLM-256M | 256M | Instant object ID ("Is this a dog?") | On App Start (always loaded) |
| **Plus ($2.99)** | Moondream 3 (0.5B) | 0.5B | Object detail, counting, receipt totals | On Shutter Press |
| **Pro ($4.99/mo)** | Qwen2-VL-2B | 2B | Document/receipt OCR, charts, handwriting | On "Document Mode" switch |
| **Pro (on-demand)** | LLaVA-v1.6-7B | 7B | Complex reasoning, "Why/How" questions | On user request only |

## 2. Model-to-Feature Mapping

| Feature | Model | Rationale |
|---------|-------|-----------|
| Real-time Camera Preview | SmolVLM-256M | Tiny enough to run in background without affecting UI |
| Object Detail / "What is this?" | Moondream 3 | Better spatial reasoning than SmolVLM |
| Document/Receipt Analysis | Qwen2-VL-2B | Superior text reading accuracy, needs more RAM |
| Complex Reasoning / Chat | LLaVA-v1.6 | Full reasoning for "Why" and "How" questions |

## 3. RAM Management

- **Max RAM Target:** 1.5GB
- **Model Format:** 4-bit Quantized GGUF
- **Action:** `dispose()` previous model before loading new `.gguf`
- **Garbage Collection:** Use Method Channels for native memory clearing

### Smooth Switching Techniques

**A. Visual Placeholder Trick**
- Keep SmolVLM output visible as a "Summary" while switching
- Show "Deep Analysis in Progress..." shimmer effect
- Background-load the heavier model (hides 1-2s load time)

**B. Proactive Memory Management ("Flush")**
- Before loading new model: `activeModel.dispose();`
- Trigger native memory clearing via Method Channels
- Ensure OS reclaims memory before allocating for next model

**C. On-Demand Download**
- Ship app with SmolVLM-256M only (keeps install size small)
- Download Moondream/Qwen/LLaVA `.gguf` files on first "Pro" access
- Use `dio` package with progress bar for download UX
- Store models in phone's local storage

## 4. Monetization Gates

| Feature | Required Tier |
|---------|--------------|
| Basic object identification | Free |
| Object counting & detail | Plus (one-time $2.99) |
| PDF Export | Plus |
| Document/Receipt OCR | Pro ($4.99/mo) |
| Batch Processing | Pro |
| Complex image reasoning | Pro |
| Offline Deep Search | Lifetime Pass Only |

## 5. Model Distillation (Long-term Goal)

**Strategy:** Train one "master model" that replaces the multi-model stack.

1. Use LLaVA (large) on Mac to label thousands of niche-specific images
2. Fine-tune SmolVLM or Moondream with those labels
3. The small model becomes as capable as the large one for your specific domain
4. Eventually delete the heavy models entirely

**Result:** Single lightweight model that handles all tiers = lower RAM, simpler code, better UX.
