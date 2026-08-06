#include <EGL/egl.h>
#include <EGL/eglext.h>

EGLDisplay eglGetDisplay(EGLNativeDisplayType d) { return EGL_NO_DISPLAY; }
EGLDisplay eglGetPlatformDisplay(EGLenum p, void *d, const EGLAttrib *a) { return EGL_NO_DISPLAY; }
EGLBoolean eglInitialize(EGLDisplay d, EGLint *maj, EGLint *min) { return EGL_FALSE; }
EGLBoolean eglChooseConfig(EGLDisplay d, const EGLint *a, EGLConfig *c, EGLint s, EGLint *n) { return EGL_FALSE; }
EGLContext eglCreateContext(EGLDisplay d, EGLConfig c, EGLContext s, const EGLint *a) { return EGL_NO_CONTEXT; }
EGLSurface eglCreatePbufferSurface(EGLDisplay d, EGLConfig c, const EGLint *a) { return EGL_NO_SURFACE; }
EGLSurface eglCreateWindowSurface(EGLDisplay d, EGLConfig c, EGLNativeWindowType w, const EGLint *a) { return EGL_NO_SURFACE; }
EGLBoolean eglMakeCurrent(EGLDisplay d, EGLSurface r, EGLSurface w, EGLContext c) { return EGL_FALSE; }
EGLBoolean eglSwapBuffers(EGLDisplay d, EGLSurface s) { return EGL_FALSE; }
EGLBoolean eglDestroyContext(EGLDisplay d, EGLContext c) { return EGL_TRUE; }
EGLBoolean eglDestroySurface(EGLDisplay d, EGLSurface s) { return EGL_TRUE; }
EGLBoolean eglTerminate(EGLDisplay d) { return EGL_TRUE; }
EGLBoolean eglBindAPI(EGLenum a) { return EGL_TRUE; }
const char *eglQueryString(EGLDisplay d, EGLint n) { return ""; }
EGLBoolean eglGetConfigAttrib(EGLDisplay d, EGLConfig c, EGLint a, EGLint *v) { return EGL_FALSE; }
__eglMustCastToProperFunctionPointerType eglGetProcAddress(const char *n) { return 0; }
