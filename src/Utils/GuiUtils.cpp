#include "GuiUtils.h"

#include "implot.h"

namespace Utils::Gui {
    void Setup() {
        rlImGuiSetup(true);
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        ImPlot::CreateContext();
        SetupStyle(true);
    }
    void ShutDown() {
        ImPlot::DestroyContext();
        rlImGuiShutdown();
    }
}
