#include <cmath>

#include "fmt/base.h"
#include "tinyfiledialogs/tinyfiledialogs.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdio.h>
#include <string.h>
#include <array>

#include "stb/stb_image_write.h"
#include "fmt/format.h"
#include "imgui.h"
#include "implot.h"
#include "raylib.h"
#include "rlImGui.h"

#include "libml/compute/graph.h"
#include "libml/neural/dataset.h"
#include "libml/neural/layers.h"
#include "libml/neural/losses.h"
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
  Viewport2D(const int width, const int height)
      : _target(LoadRenderTexture(width, height)) {}
  ~Viewport2D() { UnloadRenderTexture(_target); }
  void updateSize(const int width, const int height) {
    const int newWidth = width < 2 ? 2 : width;
    const int newHeight = height < 2 ? 2 : height;
    if (this->width() != newWidth || this->height() != newHeight) {
      UnloadRenderTexture(_target);
      _target = LoadRenderTexture(newWidth, newHeight);
    }
  }
  RenderTexture2D &target() { return _target; }
  Texture &texture() { return _target.texture; }
  int width() const { return _target.texture.width; }
  int height() const { return _target.texture.height; }

private:
  RenderTexture2D _target;
};

struct ApplicationState {
  std::optional<Texture2D> inputImage;
  bool isInTraining;
  bool isModelReady;
  bool autoEvalDuringTraining;
  int outputWidth;
  int outputHeight;
  Texture2D outputImage;
  std::optional<Texture2D> trainingOutputImage;
};

void appStateInit(ApplicationState& s){
  s.isInTraining = false;
  s.isModelReady = false;
  s.autoEvalDuringTraining = false;
  s.outputWidth = 2;
  s.outputHeight = 2;
  Image img = GenImageColor(s.outputWidth, s.outputHeight, BLACK);
  s.outputImage = LoadTextureFromImage(img);
  UnloadImage(img);
}

void loadInputImage(const std::string& path, ApplicationState& s){
  // Clear textures memory
  if(s.inputImage.has_value())
    UnloadTexture(s.inputImage.value());
  if(s.trainingOutputImage.has_value())
    UnloadTexture(s.trainingOutputImage.value());
  // Create textures based on loaded image
  s.inputImage = LoadTexture(path.c_str());
  Image img = GenImageColor(s.inputImage->width, s.inputImage->height, BLACK);
  s.trainingOutputImage = LoadTextureFromImage(img);
  UnloadImage(img);
}

std::optional<std::string> openPngFile(){
  const char* pngPattern = "*.png";
  std::array<const char*, 1> patterns = {pngPattern};
  const char* path = tinyfd_openFileDialog("Save Image", "", patterns.size(), patterns.data(), NULL, 0);
  if(path)
    return std::string(path);
  return {};
}

std::optional<std::string> savePngFile(){
  const char* pngPattern = "*.png";
  std::array<const char*, 1> patterns = {pngPattern};
  const char* path = tinyfd_saveFileDialog("Save Image", "", patterns.size(), patterns.data(), NULL);
  if(path){
    std::string finalpath{path};
    if (!finalpath.ends_with(".png"))
      finalpath += ".png";
    return finalpath;
  }
  return {};
}

void writePngFromTexture(const std::string& path, Texture2D& t){
  const int channels = 3;
  std::vector<uint8_t> data(t.width*t.height*channels);
  Image img = LoadImageFromTexture(t);
  Color * colors = LoadImageColors(img);
  for(int i=0; i<t.width*t.height; ++i){
    Color c = colors[i];
    data[i*channels] = c.r;
    data[i*channels+1] = c.g;
    data[i*channels+2] = c.b;
  }
  UnloadImageColors(colors);
  UnloadImage(img);
  stbi_write_png(path.c_str(), t.width, t.height, channels, data.data(), t.width*sizeof(uint8_t));
}



int main(int argc, char *argv[]) {

  // TODO: remove this
  ml::ComputeGraph g;

  ml::DataSet d({2, {0, 0, 0, 0}}, {3, {1, 1, 1, 1, 1, 1}});

  ml::MLP mlp(g, {ml::LayerBuilder(2, ml::LayerBuilder::Type::Identity, false),
                  ml::LayerBuilder(3, ml::LayerBuilder::Type::Identity, true)});


  std::unique_ptr<ml::Optimizer> optimizer =
      std::make_unique<ml::BatchOptimizer>(mlp, d,
                                           std::make_unique<ml::MSELoss>(g));

  while(optimizer->optimize());

  SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_ALWAYS_RUN |
                 FLAG_WINDOW_RESIZABLE);
  InitWindow(1920, 1080, "CBNN Playground");
  SetTargetFPS(60);
  Utils::Gui::Setup();

  Viewport2D modelViewport(2, 2);
  Viewport2D computeViewport(2, 2);

  // ALWAYS AFTER RAYLIB INIT
  ApplicationState appState;
  appStateInit(appState);

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

    if (ImGui::Begin("Model view")) {
      modelViewport.updateSize(ImGuiContentWidth(), ImGuiContentHeight());
      BeginTextureMode(modelViewport.target());
      ClearBackground(WHITE);
      DrawCircle(modelViewport.width() / 2, modelViewport.height() / 2,
                 std::min(modelViewport.width(), modelViewport.height()) / 4,
                 RED);
      EndTextureMode();
      rlImGuiImage(&modelViewport.texture());
    }
    ImGui::End();

    if (ImGui::Begin("Compute graph view")) {
      computeViewport.updateSize(ImGuiContentWidth(), ImGuiContentHeight());
      BeginTextureMode(computeViewport.target());
      ClearBackground(WHITE);
      DrawCircle(computeViewport.width() / 2, computeViewport.height() / 2,
                 std::min(computeViewport.width(), computeViewport.height()) /
                     4,
                 BLUE);
      EndTextureMode();
      rlImGuiImage(&computeViewport.texture());
    }
    ImGui::End();

    if (ImGui::Begin("Training Image")) {

      if (ImGui::Button("Load", ImVec2(ImGuiContentWidth(), 0))) {
        std::optional<std::string> path = openPngFile();
        if (path.has_value())
          loadInputImage(path.value(), appState);
      }

      if(appState.inputImage.has_value()){
        if (ImGui::Button("Remove", ImVec2(ImGuiContentWidth(), 0))) {
          UnloadTexture(appState.inputImage.value());
          appState.inputImage = {};
          UnloadTexture(appState.trainingOutputImage.value());
          appState.trainingOutputImage = {};
        }
        if(ImGui::CollapsingHeader("Image infos")){
          ImGui::LabelText("Dimensions", "%dx%d px", appState.inputImage->width, appState.inputImage->height);
          ImGui::Separator();
          ImGui::LabelText("Pixels", "%d", appState.inputImage->width*appState.inputImage->height);
        }
        if(appState.inputImage.has_value())
          Utils::Gui::ImageFit(appState.inputImage.value());
      }

    }
    ImGui::End();

    if(ImGui::Begin("Training Preview")){
      if(appState.trainingOutputImage.has_value()){
        Utils::Gui::ImageFit(appState.trainingOutputImage.value());
      } else {
        ImGui::TextColored({1, 0, 0, 1}, "NO INPUT IMAGE");
      }
    }
    ImGui::End();

    if (ImGui::Begin("Model Eval")) {

      bool change = false;

      if(ImGui::Button("Resize to training input", ImVec2(ImGuiContentWidth(), 0))){

        if(!appState.inputImage.has_value()){
          if(tinyfd_messageBox("Input missing", "Missing input image, load one?", "yesno", "question", 1)){
            std::optional<std::string> path = openPngFile();
            if(path.has_value())
              loadInputImage(path.value(), appState);
          }
        }

        if(appState.inputImage.has_value()){
          appState.outputWidth = appState.inputImage->width;
          appState.outputHeight = appState.inputImage->height;
          change = true;
        }
      }

      change = change || ImGui::InputInt("width", &appState.outputWidth);
      change = change || ImGui::InputInt("Height", &appState.outputHeight);


      if(change){
        if(appState.outputWidth < 2)
          appState.outputWidth = 2;
        if(appState.outputHeight < 2)
          appState.outputHeight = 2;
        UnloadTexture(appState.outputImage);
        Image img = GenImageColor(appState.outputWidth, appState.outputHeight, BLACK);
        appState.outputImage = LoadTextureFromImage(img);
        UnloadImage(img);
      }

      if(ImGui::Button("Save as PNG", ImVec2(ImGuiContentWidth(), 0))){
        std::optional<std::string> path = savePngFile();
        if(path.has_value())
          writePngFromTexture(path.value(), appState.outputImage);
      }

      ImGui::Separator();

      if(ImGui::Button("Eval model", ImVec2(ImGuiContentWidth(), 0))){

      }

      ImGui::Checkbox("Auto eval during training", &appState.autoEvalDuringTraining);

      Utils::Gui::ImageFit(appState.outputImage);
    }
    ImGui::End();

    if (ImGui::Begin("Model builder")) {

      if(ImGui::Button("New", ImVec2(ImGuiContentWidth(), 0))){

      }
      if(ImGui::Button("Load", ImVec2(ImGuiContentWidth(), 0))){

      }
      if(ImGui::Button("Save", ImVec2(ImGuiContentWidth(), 0))){

      }
      ImGui::Separator();

    }
    ImGui::End();

    if (ImGui::Begin("Training settings")) {
    }
    ImGui::End();

    if (ImGui::Begin("Model training infos")) {
    }
    ImGui::End();

    if (ImGui::Begin("Debug")) {
      ImGui::LabelText("Frame Time", "%.2f ms", GetFrameTime() * 1000.0f);
      ImGui::Separator();
    }
    ImGui::End();

    rlImGuiEnd();
    EndDrawing();
  }
  Utils::Gui::ShutDown();
  CloseWindow();

  return 0;
}
