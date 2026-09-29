#pragma once
#include <cstdio>

// Runtime logging for the live app: silent by default so double-clicking
// vtuber_live.exe shows nothing on the terminal. Pass --verbose (or -v) to
// print the full diagnostic log to stderr; from a terminal this still works
// because the app re-attaches to the parent console (Windows GUI subsystem).
inline bool g_verbose = false;

#define VLOG(...)                              \
    do {                                       \
        if (g_verbose) fprintf(stderr, __VA_ARGS__); \
    } while (0)
