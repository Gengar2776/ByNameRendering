
#include "VRImGuiMenu.h"

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <cmath>
#include <vector>

#include "../imGUI/imgui.h"
#include "../imGUI/imgui_impl_opengl3.h"

#include "BNMIncludes.hpp"
#include <BNM/Class.hpp>
#include <BNM/Method.hpp>
#include <BNM/UnityStructures.hpp>
#include "../BNMResolve.hpp"

#include "XRInput.hpp"

#define LOG(...) __android_log_print(ANDROID_LOG_DEBUG, "VRMenu", __VA_ARGS__)
#define ERR(...) __android_log_print(ANDROID_LOG_ERROR, "VRMenu", __VA_ARGS__)

using namespace BNM::Structures::Unity;

// ─── Constants ────────────────────────────────────────────────────────────────

static constexpr int   FBO_W  = 512;
static constexpr int   FBO_H  = 512;
static constexpr float MENU_X = 0.0f;
static constexpr float MENU_Y = 0.1f;
static constexpr float MENU_Z = 0.2f;
static constexpr float MENU_S = 0.3f;

// ─── Our own isolated EGL context ─────────────────────────────────────────────

static EGLDisplay g_eglDisplay = EGL_NO_DISPLAY;
static EGLContext g_eglContext = EGL_NO_CONTEXT;
static EGLSurface g_eglSurface = EGL_NO_SURFACE; // pbuffer, not a window

// ─── GLES objects ─────────────────────────────────────────────────────────────

static GLuint g_fbo = 0;
static GLuint g_tex = 0;  // color attachment — only used internally

// ─── CPU pixel buffer ─────────────────────────────────────────────────────────

static std::vector<uint8_t> g_pixels; // FBO_W * FBO_H * 4 bytes

// ─── Unity state ──────────────────────────────────────────────────────────────

static Texture2D* g_unityTex   = nullptr;
static Transform* g_rightHand  = nullptr;
static Transform* g_leftHand   = nullptr;
static Transform* g_quad       = nullptr;
static bool       g_unityReady = false;

// ─── ImGui state ──────────────────────────────────────────────────────────────

static bool g_imguiReady = false;

// ─── BNM method cache ─────────────────────────────────────────────────────────

static BNM::Method<void>  g_LoadRawTextureData; // Texture2D.LoadRawTextureData(byte[], int)
static BNM::Method<void>  g_Apply;              // Texture2D.Apply()
static bool               g_methodsResolved = false;

// ─── Input state ──────────────────────────────────────────────────────────────

static bool g_wasTriggered = false;

// ─── Math helpers ─────────────────────────────────────────────────────────────

static float   Dot  (const Vector3& a, const Vector3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static Vector3 Sub  (const Vector3& a, const Vector3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static Vector3 Add  (const Vector3& a, const Vector3& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static Vector3 MulF (const Vector3& v, float s)          { return {v.x*s,   v.y*s,   v.z*s};   }


//
//  We create a completely separate EGL context using a 1x1 pbuffer surface.
//  This has nothing to do with Unity's Vulkan context — it's our own isolated
//  GLES3 sandbox. We make it current only while we need to render/readback,
//  then release it so Unity's thread is undisturbed.
//
static bool InitEGL() {
    g_eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (g_eglDisplay == EGL_NO_DISPLAY) { ERR("eglGetDisplay failed"); return false; }

    EGLint major, minor;
    if (!eglInitialize(g_eglDisplay, &major, &minor)) { ERR("eglInitialize failed"); return false; }


    const EGLint configAttribs[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
            EGL_RED_SIZE,        8,
            EGL_GREEN_SIZE,      8,
            EGL_BLUE_SIZE,       8,
            EGL_ALPHA_SIZE,      8,
            EGL_NONE
    };

    EGLConfig config;
    EGLint numConfigs;
    if (!eglChooseConfig(g_eglDisplay, configAttribs, &config, 1, &numConfigs) || numConfigs == 0) {
        ERR("eglChooseConfig "); return false;
    }


    const EGLint surfaceAttribs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
    g_eglSurface = eglCreatePbufferSurface(g_eglDisplay, config, surfaceAttribs);
    if (g_eglSurface == EGL_NO_SURFACE) { ERR("eglCreatePbufferSurface failed"); return false; }

    const EGLint contextAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    g_eglContext = eglCreateContext(g_eglDisplay, config, EGL_NO_CONTEXT, contextAttribs);
    if (g_eglContext == EGL_NO_CONTEXT) { ERR("eglCreateContext failed"); return false; }

    LOG("EGL created");
    return true;
}


template<typename Fn>
static bool WithOurContext(Fn fn) {
    if (!eglMakeCurrent(g_eglDisplay, g_eglSurface, g_eglSurface, g_eglContext)) {
        ERR("eglMakeCurrent failed: 0x%x", eglGetError());
        return false;
    }
    fn();
    eglMakeCurrent(g_eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    return true;
}



static bool InitFBO() {
    glGenTextures(1, &g_tex);
    glBindTexture(GL_TEXTURE_2D, g_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, FBO_W, FBO_H, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers(1, &g_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, g_tex, 0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (status != GL_FRAMEBUFFER_COMPLETE) {
        ERR("FBO incomplete: 0x%x", status);
        return false;
    }

    g_pixels.resize(FBO_W * FBO_H * 4, 0);
    LOG("FBO ready");
    return true;
}



static void InitImGui() {
    ImGui::CreateContext();
    ImGuiIO& io    = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)FBO_W, (float)FBO_H);
    io.IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGui_ImplOpenGL3_Init("#version 300 es");
    g_imguiReady = true;
    LOG("ImGui ready");
}



static bool ResolveMethods() {
    if (g_methodsResolved) return true;

    auto tex2DClass = BNM::Class("UnityEngine", "Texture2D",
                                 BNM::Image("UnityEngine.CoreModule.dll"));


    g_LoadRawTextureData = tex2DClass.GetMethod("LoadRawTextureData", 2);

    g_Apply = tex2DClass.GetMethod("Apply", 0);

    if (!g_LoadRawTextureData.IsValid() || !g_Apply.IsValid()) {
        ERR("Failed to resolve Texture2D methods");
        return false;
    }

    g_methodsResolved = true;
    return true;
}



static Transform* FindTransformByNames(const char** names, int count) {
    for (int i = 0; i < count; i++) {
        auto obj = GameObject::Find(names[i]);
        if (obj) { auto t = obj->GetTransform(); if (t) return t; }
    }
    return nullptr;
}

static bool SetupUnity() {

    if (!ResolveMethods()) return false;


    if (!g_rightHand) {
        const char* names[] = { "RightHandAnchor", "RightHand Controller", "RightHand" };
        g_rightHand = FindTransformByNames(names, 3);
        if (!g_rightHand) { LOG("noRHfound"); return false; }
    }
    if (!g_leftHand) {
        const char* names[] = { "LeftHandAnchor", "LeftHand Controller", "LeftHand" };
        g_leftHand = FindTransformByNames(names, 3);
        if (!g_leftHand) { LOG("no"); return false; }
    }


    // TextureFormat 4 = RGBA32
    static BNM::Method<Texture2D*> ctor =
            BNM::Class("UnityEngine", "Texture2D",
                       BNM::Image("UnityEngine.CoreModule.dll"))
                    .GetMethod(".ctor", 2); // Texture2D(int width, int height)

    if (!ctor.IsValid()) { ERR("Texture2D ctor not found"); return false; }

    auto tex2DClass = BNM::Class("UnityEngine", "Texture2D",
                                 BNM::Image("UnityEngine.CoreModule.dll"));
    g_unityTex = (Texture2D*)BNM::IL2CPP::IL2CPP_VT_BLOB_OBJECT;
    if (!g_unityTex) { ERR("CreateNewObject failsed"); return false; }
    ctor.Call(g_unityTex, FBO_W, FBO_H);


    auto quadObj = GameObject::CreatePrimitive(PrimitiveType::Quad);
    if (!quadObj) { ERR("CreatePrimitive faileds"); return false; }

    g_quad = quadObj->GetTransform();
    if (!g_quad) { ERR("Quad transform false"); return false; }

    auto renderer = (Renderer*)quadObj->GetComponent(Renderer::GetType());
    if (renderer) {
        auto mat = renderer->GetMaterial();
        if (mat) {
            auto shader = Shader::Find("Unlit/Texture");
            if (shader) mat->SetShader(shader);

            static BNM::Method<void> SetMainTex =
                    BNM::Class("UnityEngine", "Material",
                               BNM::Image("UnityEngine.CoreModule.dll"))
                            .GetMethod("set_mainTexture", 1);
            if (SetMainTex.IsValid()) SetMainTex.Call(mat, g_unityTex);
        }
    }

    g_quad->SetParent(g_rightHand, false);
    g_quad->SetLocalPosition({ MENU_X, MENU_Y, MENU_Z });
    g_quad->SetLocalScale   ({ MENU_S, MENU_S, MENU_S });

    LOG("Unity setup complete");
    return true;
}



static void UpdateInput() {
    if (!g_quad || !g_leftHand) return;

    ImGuiIO& io = ImGui::GetIO();

    Vector3 rayOrigin  = g_leftHand->GetPosition();
    Vector3 rayDir     = g_leftHand->GetForward();
    Vector3 quadPos    = g_quad->GetPosition();
    Vector3 quadNormal = g_quad->GetForward();

    float denom = Dot(quadNormal, rayDir);
    if (std::fabs(denom) < 1e-6f)
    { io.MousePos = ImVec2(-1,-1); io.MouseDown[0] = false; return; }

    float t = Dot(Sub(quadPos, rayOrigin), quadNormal) / denom;
    if (t < 0.0f)
    { io.MousePos = ImVec2(-1,-1); io.MouseDown[0] = false; return; }

    Vector3 hitPoint   = Add(rayOrigin, MulF(rayDir, t));
    Vector3 right      = g_quad->GetRight();
    Vector3 up         = g_quad->GetUp();
    Vector3 localHit   = Sub(hitPoint, quadPos);
    Vector3 lossyScale = g_quad->GetLocalScale();

    float u = Dot(localHit, right) / lossyScale.x + 0.5f;
    float v = Dot(localHit, up)    / lossyScale.y + 0.5f;

    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f)
    { io.MousePos = ImVec2(-1,-1); io.MouseDown[0] = false; return; }

    io.MousePos = ImVec2(u * FBO_W, (1.0f - v) * FBO_H);

    bool triggered  = XRInput::GetBoolFeature(TriggerButton, Controller::Left);
    io.MouseDown[0] = triggered;

    if (triggered && !g_wasTriggered)
        XRInput::SendHapticImpulse(Controller::Left, 0.3f, 0.05f);
    g_wasTriggered = triggered;
}



static void RenderAndReadback() {
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glViewport(0, 0, FBO_W, FBO_H);
    glClearColor(0.0f, 0.0f, 0.0f, 0.85f);
    glClear(GL_COLOR_BUFFER_BIT);

    UpdateInput();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();


    ImGui::Begin("GenLauncher");
    ImGui::Text("Gengars Imgui porn menu");
    ImGui::End();


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glReadPixels(0, 0, FBO_W, FBO_H,
                 GL_RGBA, GL_UNSIGNED_BYTE, g_pixels.data());

    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    const int rowBytes = FBO_W * 4;
    std::vector<uint8_t> rowBuf(rowBytes);
    for (int y = 0; y < FBO_H / 2; y++) {
        uint8_t* top = g_pixels.data() + y              * rowBytes;
        uint8_t* bot = g_pixels.data() + (FBO_H-1 - y) * rowBytes;
        memcpy(rowBuf.data(), top,          rowBytes);
        memcpy(top,           bot,          rowBytes);
        memcpy(bot,           rowBuf.data(), rowBytes);
    }
}



static void UploadToUnity() {
    if (!g_unityTex || !g_LoadRawTextureData.IsValid() || !g_Apply.IsValid()) return;


    g_LoadRawTextureData.Call(g_unityTex,
                              g_pixels.data(),
                              (int)g_pixels.size());


    g_Apply.Call(g_unityTex);
}



void VRMenu::onupdate() {


    if (g_eglContext == EGL_NO_CONTEXT) {
        if (!InitEGL()) return;
        bool ok = false;
        WithOurContext([&]{ ok = InitFBO(); });
        if (!ok) return;
        WithOurContext([&]{ InitImGui(); });
    }


    if (!g_unityReady) {
        g_unityReady = SetupUnity();
        if (!g_unityReady) return;
    }


    WithOurContext([&]{ RenderAndReadback(); });
    UploadToUnity();
}
