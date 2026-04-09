#pragma once

#include <GLES3/gl3.h>
#include <jni.h>
#include "../imGUI/imgui.h"

namespace VRMenu {
    using DrawCallback = void(*)();
    void Init(JNIEnv* env, jobject activity);
    void Shutdown();
    void onupdate();
    GLuint GetMenuTexture();
    void RegisterDrawCallback(DrawCallback cb);
}
