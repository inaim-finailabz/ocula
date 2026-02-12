/*
 * mtmd_sources.mm — Compilation unit for mtmd-helper.cpp only.
 *
 * mtmd-helper.cpp has a compile-time guard that prevents it from being
 * compiled in the same TU as clip.h / mtmd-audio.h (MTMD_INTERNAL_HEADER).
 * The other mtmd .cpp files are compiled in mtmd_sources_internal.mm.
 *
 * We also #undef DEBUG because Xcode defines it in debug builds and it
 * conflicts with a local variable in the mtmd code.
 */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"

// mtmd-helper MUST be compiled alone — it checks that MTMD_INTERNAL_HEADER
// is NOT defined, but clip.h/mtmd-audio.h define it.
#ifdef DEBUG
#  undef DEBUG
#endif

#include "../../llama.cpp/tools/mtmd/mtmd-helper.cpp"

#pragma clang diagnostic pop

