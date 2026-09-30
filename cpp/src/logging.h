#pragma once
#include <cstdio>

// Runtime logging for the live app: silent by default so double-clicking
// vtuber_live shows nothing on the terminal. Pass --verbose (or -v) to
// print the full diagnostic log to stderr; from a terminal this still works
// because the app re-attaches to the parent console (Windows GUI subsystem).
inline bool g_verbose = false;

// Quiet mode (non-verbose) redirects stdout/stderr to /dev/null because the
// MediaPipe library prints glog diagnostics (INFO/WARNING lines) to stderr
// and ignores GLOG_minloglevel. g_real_err keeps a handle on the original
// stderr for critical user-facing errors (ELOG below), which must stay
// visible even in quiet mode.
inline FILE* g_real_err = nullptr;

// Error log: writes to the original stderr even when quiet mode has
// redirected fd 2 to /dev/null. Use for fatal/startup errors the user
// must see (model load failure, etc).
#define ELOG(...)                                                     \
    do {                                                              \
        fprintf(g_real_err ? g_real_err : stderr, __VA_ARGS__);      \
        fflush(g_real_err ? g_real_err : stderr);                     \
    } while (0)

#define VLOG(...)                              \
    do {                                       \
        if (g_verbose) fprintf(stderr, __VA_ARGS__); \
    } while (0)
