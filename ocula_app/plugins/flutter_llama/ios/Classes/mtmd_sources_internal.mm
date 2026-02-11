/*
 * mtmd_sources_internal.mm — Compilation unit for the internal mtmd
 * library sources: clip.cpp, mtmd.cpp, mtmd-audio.cpp, and all model files.
 *
 * These files share MTMD_INTERNAL_HEADER and can coexist in one TU
 * but must NOT be in the same TU as mtmd-helper.cpp.
 *
 * We #undef DEBUG because Xcode debug builds define it as a macro
 * which conflicts with local variables in the mtmd source.
 */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"

#ifdef DEBUG
#  undef DEBUG
#endif

// ---- mtmd core (internal) -------------------------------------------------
#include "../../llama.cpp/tools/mtmd/clip.cpp"
#include "../../llama.cpp/tools/mtmd/mtmd-audio.cpp"
#include "../../llama.cpp/tools/mtmd/mtmd.cpp"

// ---- mtmd model-specific sources ------------------------------------------
#include "../../llama.cpp/tools/mtmd/models/cogvlm.cpp"
#include "../../llama.cpp/tools/mtmd/models/conformer.cpp"
#include "../../llama.cpp/tools/mtmd/models/glm4v.cpp"
#include "../../llama.cpp/tools/mtmd/models/internvl.cpp"
#include "../../llama.cpp/tools/mtmd/models/kimivl.cpp"
#include "../../llama.cpp/tools/mtmd/models/llama4.cpp"
#include "../../llama.cpp/tools/mtmd/models/llava.cpp"
#include "../../llama.cpp/tools/mtmd/models/minicpmv.cpp"
#include "../../llama.cpp/tools/mtmd/models/mobilenetv5.cpp"
#include "../../llama.cpp/tools/mtmd/models/pixtral.cpp"
#include "../../llama.cpp/tools/mtmd/models/qwen2vl.cpp"
#include "../../llama.cpp/tools/mtmd/models/qwen3vl.cpp"
#include "../../llama.cpp/tools/mtmd/models/siglip.cpp"
#include "../../llama.cpp/tools/mtmd/models/whisper-enc.cpp"
#include "../../llama.cpp/tools/mtmd/models/youtuvl.cpp"

#pragma clang diagnostic pop
