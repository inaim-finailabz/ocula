# Bundled AI Models

This directory contains the pre-bundled AI models that ship with the app:

## Free Tier Model (Bundled)
- `smolvlm-256m-q4.gguf` (180MB) - Ships with app for instant AI features
- No download required, loads directly at startup

## Pro Models (Downloaded on Demand)
- `moondream2-q4.gguf` (350MB) - Downloaded when user upgrades to Plus
- `qwen2.5-vl-3b-q4.gguf` (2GB) - Downloaded when user upgrades to Pro
- `mmproj-qwen2.5-vl-f16.gguf` (600MB) - Vision projector for Pro tier

## Production Deployment
To bundle the free model with the app:
1. Download `smolvlm-256m-q4.gguf` from HuggingFace
2. Place it in this directory
3. The app will automatically use the bundled version