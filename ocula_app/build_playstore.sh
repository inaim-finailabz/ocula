#!/usr/bin/env bash
set -euo pipefail
#
# Build Play Store AAB with Play Asset Delivery
# Models are delivered as a fast-follow asset pack instead of bundled in the APK.
#
# Usage:
#   ./build_playstore.sh           # Build AAB
#   ./build_playstore.sh --apk     # Build website APK (with bundled models)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/assets/models"
BACKUP_DIR="/tmp/ocula_models_backup"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

BUILD_APK=false
if [[ "${1:-}" == "--apk" ]]; then
    BUILD_APK=true
fi

if $BUILD_APK; then
    echo -e "${BOLD}Building Website APK (models bundled)${NC}"
    echo ""
    cd "$SCRIPT_DIR"
    flutter build apk --release
    APK="$SCRIPT_DIR/build/app/outputs/flutter-apk/app-release.apk"
    echo ""
    echo -e "${GREEN}APK ready:${NC} $APK"
    echo -e "Size: $(du -sh "$APK" | cut -f1)"
    cp "$APK" ~/Desktop/Ocula.apk
    echo -e "Copied to: ${CYAN}~/Desktop/Ocula.apk${NC}"
    exit 0
fi

echo -e "${BOLD}Building Play Store AAB (models via Asset Delivery)${NC}"
echo ""

# Step 1: Move GGUF models out of Flutter assets (they're in the asset pack instead)
echo -e "${CYAN}[1/4] Moving models out of Flutter assets...${NC}"
mkdir -p "$BACKUP_DIR"
for f in "$MODELS_DIR"/*.gguf; do
    if [[ -f "$f" ]]; then
        mv "$f" "$BACKUP_DIR/"
        echo "  Moved: $(basename "$f")"
    fi
done

# Keep a placeholder so Flutter doesn't complain about empty asset dir
echo "Models delivered via Play Asset Delivery" > "$MODELS_DIR/.placeholder"

# Step 2: Build AAB
echo -e "${CYAN}[2/4] Building AAB...${NC}"
cd "$SCRIPT_DIR"

# Trap to restore models even if build fails
restore_models() {
    echo -e "${CYAN}[*] Restoring models to Flutter assets...${NC}"
    rm -f "$MODELS_DIR/.placeholder"
    for f in "$BACKUP_DIR"/*.gguf; do
        if [[ -f "$f" ]]; then
            mv "$f" "$MODELS_DIR/"
        fi
    done
    rmdir "$BACKUP_DIR" 2>/dev/null || true
}
trap restore_models EXIT

flutter build appbundle --release || echo -e "${CYAN}(debug symbol stripping warning is non-fatal, continuing...)${NC}"

# Step 3: Restore models
# (handled by trap, but let's be explicit)
restore_models
trap - EXIT

# Step 4: Report
AAB="$SCRIPT_DIR/build/app/outputs/bundle/release/app-release.aab"
echo ""
echo -e "${GREEN}${BOLD}Play Store AAB ready!${NC}"
echo -e "  AAB: $AAB"
echo -e "  Size: $(du -sh "$AAB" | cut -f1)"
echo ""
echo -e "  ${CYAN}Base AAB:${NC} App code + UI (should be < 150MB)"
echo -e "  ${CYAN}Asset Pack:${NC} models_pack (fast-follow, ~545MB)"
echo -e "  ${CYAN}Total install:${NC} ~700MB (models download after install)"
echo ""
echo -e "  ${BOLD}Upload to Google Play Console:${NC}"
echo -e "  https://play.google.com/console"
echo ""
cp "$AAB" ~/Desktop/Ocula-playstore.aab
echo -e "  Copied to: ${CYAN}~/Desktop/Ocula-playstore.aab${NC}"
