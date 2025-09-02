#include <cmath>

#include "fmt/format.h"
#include "raylib.h"
#include "imgui.h"
#include "implot.h"
#include "rlImGui.h"

#include "libml/neural/dataset.h"
#include "libml/neural/losses.h"
#include "libml/compute/graph.h"
#include "libml/neural/layers.h"
#include "libml/neural/mlp.h"
#include "libml/neural/optimizers.h"

#include "Utils/GuiUtils.h"

int ImGuiContentWidth() {
  const ImVec2 vMin = ImGui::GetWindowContentRegionMin();
  const ImVec2 vMax = ImGui::GetWindowContentRegionMax();
  return static_cast<int>(vMax.x - vMin.x);
}

int ImGuiContentHeight() {
  const ImVec2 vMin = ImGui::GetWindowContentRegionMin();
  const ImVec2 vMax = ImGui::GetWindowContentRegionMax();
  return static_cast<int>(vMax.y - vMin.y);
}

class Viewport2D {
public:
  Viewport2D(const int width, const int height) : _target(LoadRenderTexture(width, height)){}
  ~Viewport2D() {
    UnloadRenderTexture(_target);
  }
  void updateSize(const int width, const int height) {
    const int newWidth = width < 2 ? 2 : width;
    const int newHeight = height < 2 ? 2 : height;
    if (this->width() != newWidth || this->height() != newHeight) {
      UnloadRenderTexture(_target);
      _target = LoadRenderTexture(newWidth, newHeight);
    }
  }
  RenderTexture2D& target() { return _target;}
  Texture& texture() {return _target.texture;}
  int width() const {return _target.texture.width;}
  int height() const {return _target.texture.height;}
private:
  RenderTexture2D _target;
};


int main(int argc, char *argv[]) {

  ml::ComputeGraph g;

  ml::DataSet d({2,{
    0.1, 0.1,
    0.5, 0.5}},
    {3,{
        0.5, 0.5, 0.5,
        0.1, 0.2, 1.0}});

  ml::MLP mlp(g, {
    ml::LayerBuilder(2, ml::LayerBuilder::Type::Identity, false),
    ml::LayerBuilder(12, ml::LayerBuilder::Type::ReLu, true),
    ml::LayerBuilder(12, ml::LayerBuilder::Type::ReLu, true),
    ml::LayerBuilder(3, ml::LayerBuilder::Type::Identity, true)
  });

  std::unique_ptr<ml::Optimizer> optimizer = std::make_unique<ml::BatchOptimizer>(mlp, d, std::make_unique<ml::MSELoss>(g));

  optimizer->optimize();


  SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_ALWAYS_RUN | FLAG_WINDOW_RESIZABLE);
  InitWindow(1920, 1080, "CBNN Playground");
  SetTargetFPS(60);
  Utils::Gui::Setup();

  Viewport2D modelViewport(2,2);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground({30, 31, 34, 255});

    // Gui drawing
    rlImGuiBegin();

    ImGui::BeginMainMenuBar();
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Create")) {
      }
      if (ImGui::MenuItem("Open")) {
      }
      if (ImGui::MenuItem("Save")) {
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
      if (ImGui::MenuItem("Reset")) {
      }
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();

    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpaceOverViewport(dockspace_id);


    ImGui::Begin("Model view");
    modelViewport.updateSize(ImGuiContentWidth(), ImGuiContentHeight());
    BeginTextureMode(modelViewport.target());
      ClearBackground(WHITE);
      DrawCircle(modelViewport.width()/2, modelViewport.height()/2, std::min(modelViewport.width(), modelViewport.height())/4, RED);
    EndTextureMode();
    rlImGuiImage(&modelViewport.texture());
    ImGui::End();

    ImGui::Begin("Compute graph view");
    ImGui::End();

    ImGui::Begin("Input preview");
    ImGui::End();

    ImGui::Begin("Output preview");
    ImGui::End();

    ImGui::Begin("Model builder");
    ImGui::End();

    ImGui::Begin("Training settings");
    ImGui::End();

    ImGui::Begin("Model training infos");
    ImGui::End();

    ImGui::Begin("Debug");
    ImGui::Text(fmt::format("Frame time: {:.2}ms", GetFrameTime() * 1000.0f).c_str());
    ImGui::End();


    rlImGuiEnd();
    EndDrawing();
  }
  Utils::Gui::ShutDown();
  CloseWindow();

  return 0;
}