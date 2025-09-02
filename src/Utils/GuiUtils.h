#pragma once

#include "raylib.h"
#include "imgui.h"
#include "rlImGui.h"

namespace Utils::Gui {

    void Setup();
    void ShutDown();

    inline void SetupStyle(const bool useDarkStyle){
        //useDarkStyle ? ImGui::StyleColorsDark() : ImGui::StyleColorsLight();
        ImGuiStyle &style = ImGui::GetStyle();
        // Sets the border sizes and rounding.
        style.WindowRounding = 0.0f;
        // style.ChildRounding = 8.0f;
        // style.FrameRounding = 6.0f;
        // style.PopupRounding = 6.0f;
        // style.ScrollbarRounding = 6.0f;
        // style.GrabRounding = 6.0f;
        // style.TabRounding = 6.0f;
    }

    inline void ImageFit(const Texture2D &texture){
        const ImVec2 windowSize = ImGui::GetContentRegionAvail();
        const auto imgW = static_cast<float>(texture.width);
        const auto imgH = static_cast<float>(texture.height);
        const float imgRatio = imgW / imgH;
        const float winRatio = windowSize.x / windowSize.y;
        float ratio;
        if (winRatio > imgRatio)
            ratio = windowSize.y / imgH;
        else
            ratio = windowSize.x / imgW;
        rlImGuiImageSize(&texture, static_cast<int>(imgW * ratio), static_cast<int>(imgH * ratio));
    }
}
