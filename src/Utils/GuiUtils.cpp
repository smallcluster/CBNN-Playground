#include "GuiUtils.h"

#include "implot.h"

namespace Utils::Gui {
    void Setup() {
        rlImGuiSetup(true);
        ImPlot::CreateContext();
        SetupStyle(true);
    }
    void ShutDown() {
        ImPlot::DestroyContext();
        rlImGuiShutdown();
    }
}
