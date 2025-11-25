#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdio.h>
#include <string.h>

#include "fmt/base.h"
#include "fmt/format.h"
#include "imgui.h"
#include "implot.h"
#include "raylib.h"
#include "rlImGui.h"
#include "stb/stb_image_write.h"
#include "tinyfiledialogs/tinyfiledialogs.h"

#include "libml/compute/graph.h"
#include "libml/neural/dataset.h"
#include "libml/neural/layers.h"
#include "libml/neural/losses.h"
#include "libml/neural/mlp.h"
#include "libml/neural/optimizers.h"

#include "Utils/GuiUtils.h"
#include "Utils/MathUtils.h"

#define BATCH_OP 0
#define SGD_OP 1

#define MSE_LOSS 0
#define L2_LOSS 1
#define L1_LOSS 2

#define MAX_PLOT_POINTS 200

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

class Window {
public:
  Window(const int width, const int height, const char *title,
         const unsigned int flags) {
    SetConfigFlags(flags);
    InitWindow(width, height, title);
  }
  ~Window() { CloseWindow(); }
  void setTargetFps(const int fps) { SetTargetFPS(fps); }
  bool shouldClose() { return WindowShouldClose(); }
};

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
  std::optional<ml::DataSet> dataSet;
  std::vector<int> deepLayerWidths;
  std::vector<int> deepLayerActivationFuncs;
  ml::ComputeGraph g;
  std::unique_ptr<ml::MLP> mlp;
  std::unique_ptr<ml::Optimizer> optimizer;
  std::optional<Texture2D> inputImage;
  int trainingSteps = 0;
  std::vector<double> avgMSE;
  bool isInTraining = false;
  bool isModelReady = false;
  bool autoEvalDuringTraining = false;
  int outputWidth = 2;
  int outputHeight = 2;
  Texture2D outputImage;
  std::optional<Texture2D> trainingOutputImage;
  const std::vector<const char *> activationFuncChoices = {"Identity", "ReLu",
                                                           "Sigmoid"};
  int currentOptimizer = 1;
  const std::vector<const char *> optimizerChoices = {"Batch", "SGD_OP"};
  int currentLoss = 0;
  const std::vector<const char *> lossChoices = {"MSE", "L2", "L1"};

  double lastLearningRate = 0.01;
  double lastMomentum = 0.0;
  bool lastIsNesterov = false;

  ApplicationState() {
    Image img = GenImageColor(outputWidth, outputHeight, BLACK);
    outputImage = LoadTextureFromImage(img);
    UnloadImage(img);
  }
  ~ApplicationState() {
    if (inputImage.has_value()) {
      UnloadTexture(inputImage.value());
      inputImage = {};
    }

    if (trainingOutputImage.has_value()) {
      UnloadTexture(trainingOutputImage.value());
      trainingOutputImage = {};
    }

    UnloadTexture(outputImage);
  }
};

ml::DataSet genDataSetFromImage(const Image &img) {
  constexpr int numChannels = 3;
  std::vector<double> coords;
  coords.reserve(2 * img.height * img.width);
  std::vector<double> data;
  data.reserve(numChannels * img.height * img.width);
  Color *colors = LoadImageColors(img);
  for (int y = 0; y < img.height; ++y) {
    for (int x = 0; x < img.width; ++x) {
      Color c = colors[y * img.width + x];
      // Normalized space
      coords.push_back(static_cast<double>(x) / static_cast<double>(img.width));
      coords.push_back(static_cast<double>(y) /
                       static_cast<double>(img.height));
      constexpr double channelMaxVal = 255.0;
      data.push_back(static_cast<double>(c.r) / channelMaxVal);
      data.push_back(static_cast<double>(c.g) / channelMaxVal);
      data.push_back(static_cast<double>(c.b) / channelMaxVal);
    }
  }
  UnloadImageColors(colors);
  return std::move(ml::DataSet({2, coords}, {numChannels, data}));
}

ml::DataSet genDataSetFromTexture(const Texture2D &t) {
  Image img = LoadImageFromTexture(t);
  ml::DataSet d{genDataSetFromImage(img)};
  UnloadImage(img);
  return std::move(d);
}

void loadInputImage(const std::string &path, ApplicationState &s) {
  // Clear textures memory
  if (s.inputImage.has_value())
    UnloadTexture(s.inputImage.value());
  if (s.trainingOutputImage.has_value())
    UnloadTexture(s.trainingOutputImage.value());
  // Load and format image to RGBA 32bit
  Image imgFile = LoadImage(path.c_str());
  ImageFormat(&imgFile, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
  // Create dataset
  s.dataSet = genDataSetFromImage(imgFile);
  // Create texture
  s.inputImage = LoadTextureFromImage(imgFile);
  UnloadImage(imgFile);

  // Create Training preview (same dimensions as input)
  Image img = GenImageColor(s.inputImage->width, s.inputImage->height, BLACK);
  s.trainingOutputImage = LoadTextureFromImage(img);
  UnloadImage(img);
}

std::optional<std::string> openPngFile() {
  const char *pngPattern = "*.png";
  std::array<const char *, 1> patterns = {pngPattern};
  const char *path = tinyfd_openFileDialog("Save Image", "", patterns.size(),
                                           patterns.data(), NULL, 0);
  if (path)
    return std::string(path);
  return {};
}

std::optional<std::string> savePngFile() {
  const char *pngPattern = "*.png";
  std::array<const char *, 1> patterns = {pngPattern};
  const char *path = tinyfd_saveFileDialog("Save Image", "", patterns.size(),
                                           patterns.data(), NULL);
  if (path) {
    std::string finalpath{path};
    if (!finalpath.ends_with(".png"))
      finalpath += ".png";
    return finalpath;
  }
  return {};
}

void writePngFromTexture(const std::string &path, Texture2D &t) {
  const int channels = 3;
  std::vector<uint8_t> data(t.width * t.height * channels);
  Image img = LoadImageFromTexture(t);
  Color *colors = LoadImageColors(img);
  for (int i = 0; i < t.width * t.height; ++i) {
    Color c = colors[i];
    data[i * channels] = c.r;
    data[i * channels + 1] = c.g;
    data[i * channels + 2] = c.b;
  }
  UnloadImageColors(colors);
  UnloadImage(img);
  stbi_write_png(path.c_str(), t.width, t.height, channels, data.data(),
                 t.width * sizeof(uint8_t));
}

ml::LayerBuilder::Type layerIndexToType(const int i) {
  switch (i) {
  default:
  case 0:
    return ml::LayerBuilder::Type::Identity;
  case 1:
    return ml::LayerBuilder::Type::ReLu;
  case 2:
    return ml::LayerBuilder::Type::Sigmoid;
  }
}

std::unique_ptr<ml::MLP>
buildCBNR(ml::ComputeGraph &g, const std::vector<int> &deepLayerWidths,
          const std::vector<int> &deepLayerActivationFuncs) {
  std::vector<ml::LayerBuilder> layers;
  layers.push_back(
      ml::LayerBuilder(2, ml::LayerBuilder::Type::Identity, false));
  for (int i = 0; i < deepLayerWidths.size(); ++i) {
    layers.push_back(
        ml::LayerBuilder(deepLayerWidths[i],
                         layerIndexToType(deepLayerActivationFuncs[i]), true));
  }
  layers.push_back(
      ml::LayerBuilder(3, ml::LayerBuilder::Type::Identity, false));
  return std::make_unique<ml::MLP>(g, layers);
}

void evalMlPToTexture(ml::MLP &mlp, Texture2D &t) {
  std::vector<Color> colors;
  colors.reserve(t.width * t.height);
  for (int y = 0; y < t.height; ++y) {
    for (int x = 0; x < t.width; ++x) {
      // In normalized space
      mlp.setInput(static_cast<double>(x) / static_cast<double>(t.width), 0);
      mlp.setInput(static_cast<double>(y) / static_cast<double>(t.height), 1);
      mlp.eval();
      // Fetch the colors to RGBA 32bit
      constexpr double channelMaxVal = 255.0;
      Color c = {static_cast<unsigned char>(std::clamp(
                     mlp.getOutput(0) * channelMaxVal, 0.0, channelMaxVal)),
                 static_cast<unsigned char>(std::clamp(
                     mlp.getOutput(1) * channelMaxVal, 0.0, channelMaxVal)),
                 static_cast<unsigned char>(std::clamp(
                     mlp.getOutput(2) * channelMaxVal, 0.0, channelMaxVal)),
                 static_cast<unsigned char>(channelMaxVal)};
      colors.push_back(c);
    }
  }
  UpdateTexture(t, &colors[0]);
}

void createOptimizer(ApplicationState &s) {
  std::unique_ptr<ml::Loss> loss;
  if (s.currentLoss == MSE_LOSS)
    loss = std::make_unique<ml::MSELoss>(s.g);
  else if (s.currentLoss == L2_LOSS)
    loss = std::make_unique<ml::L2Loss>(s.g);
  else
    loss = std::make_unique<ml::L1Loss>(s.g);
  if (s.currentOptimizer == BATCH_OP) {
    s.optimizer = std::make_unique<ml::BatchOptimizer>(
        *s.mlp, std::move(loss), s.lastLearningRate, s.lastMomentum);
  } else if (s.currentOptimizer == SGD_OP) {
    s.optimizer = std::make_unique<ml::SGDOptimizer>(
        *s.mlp, std::move(loss), s.lastLearningRate, s.lastMomentum,
        s.lastIsNesterov);
  }
}

void askLoadInputImage(ApplicationState &s) {
  if (tinyfd_messageBox("Input missing", "Missing input image, load one?",
                        "yesno", "question", 1)) {
    std::optional<std::string> path = openPngFile();
    if (path.has_value())
      loadInputImage(path.value(), s);
  }
}

int main(int argc, char *argv[]) {

  Window window(1920, 1080, "CBNN Playground",
                FLAG_MSAA_4X_HINT | FLAG_WINDOW_ALWAYS_RUN |
                    FLAG_WINDOW_RESIZABLE);
  window.setTargetFps(60);

  Utils::Gui::Setup();

  Viewport2D modelViewport(2, 2);
  Viewport2D computeViewport(2, 2);

  // ALWAYS AFTER WINDOW INIT
  ApplicationState appState;

  while (!window.shouldClose()) {
    BeginDrawing();
    ClearBackground({30, 31, 34, 255});

    if (appState.isInTraining) {
      // run a training pass
      while (appState.optimizer->optimize())
        ;
      ++appState.trainingSteps;
      // TODO : improve API for this
      double loss = appState.optimizer->getLoss().loss;

      appState.avgMSE.push_back(loss);
      if (appState.avgMSE.size() > MAX_PLOT_POINTS)
        appState.avgMSE.erase(appState.avgMSE.begin());

      // Update the result on the output preview
      evalMlPToTexture(*appState.mlp, appState.trainingOutputImage.value());

      // Update the result to the resolution independant image
      // (upscaling/donwscaling)
      // If both image are the same, just copy the pixels over
      if (appState.outputImage.width == appState.trainingOutputImage->width &&
          appState.outputImage.height && appState.autoEvalDuringTraining) {
        Image img = LoadImageFromTexture(appState.trainingOutputImage.value());
        Color *colors = LoadImageColors(img);
        UpdateTexture(appState.outputImage, colors);
        UnloadImageColors(colors);
        UnloadImage(img);
      } else if (appState.autoEvalDuringTraining)
        evalMlPToTexture(*appState.mlp, appState.outputImage);
    }

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

      if (appState.isInTraining)
        ImGui::BeginDisabled();

      if (ImGui::Button("Load", ImVec2(ImGuiContentWidth(), 0))) {
        std::optional<std::string> path = openPngFile();
        if (path.has_value())
          loadInputImage(path.value(), appState);
      }

      if (appState.inputImage.has_value()) {
        if (ImGui::Button("Remove", ImVec2(ImGuiContentWidth(), 0))) {
          UnloadTexture(appState.inputImage.value());
          appState.inputImage = {};
          UnloadTexture(appState.trainingOutputImage.value());
          appState.trainingOutputImage = {};
        }

        if (appState.isInTraining)
          ImGui::EndDisabled();

        if (ImGui::CollapsingHeader("Image infos")) {
          ImGui::LabelText("Dimensions", "%dx%d px", appState.inputImage->width,
                           appState.inputImage->height);
          ImGui::Separator();
          ImGui::LabelText("Pixels", "%d",
                           appState.inputImage->width *
                               appState.inputImage->height);
        }
        if (appState.inputImage.has_value())
          Utils::Gui::ImageFit(appState.inputImage.value());
      } else if (appState.isInTraining)
        ImGui::EndDisabled();
    }
    ImGui::End();

    if (ImGui::Begin("Training Preview")) {
      if (appState.trainingOutputImage.has_value()) {
        Utils::Gui::ImageFit(appState.trainingOutputImage.value());
      } else {
        ImGui::TextColored({1, 0, 0, 1}, "NO INPUT IMAGE");
      }
    }
    ImGui::End();

    if (ImGui::Begin("Model Eval")) {

      bool change = false;

      if (ImGui::Button("Resize to training input",
                        ImVec2(ImGuiContentWidth(), 0))) {

        if (!appState.inputImage.has_value()) {
          askLoadInputImage(appState);
        }

        if (appState.inputImage.has_value()) {
          appState.outputWidth = appState.inputImage->width;
          appState.outputHeight = appState.inputImage->height;
          change = true;
        }
      }

      change = change || ImGui::InputInt("width", &appState.outputWidth);
      change = change || ImGui::InputInt("Height", &appState.outputHeight);

      if (change) {
        if (appState.outputWidth < 2)
          appState.outputWidth = 2;
        if (appState.outputHeight < 2)
          appState.outputHeight = 2;
        UnloadTexture(appState.outputImage);
        Image img =
            GenImageColor(appState.outputWidth, appState.outputHeight, BLACK);
        appState.outputImage = LoadTextureFromImage(img);
        UnloadImage(img);
      }

      if (ImGui::Button("Save as PNG", ImVec2(ImGuiContentWidth(), 0))) {
        std::optional<std::string> path = savePngFile();
        if (path.has_value())
          writePngFromTexture(path.value(), appState.outputImage);
      }

      ImGui::Separator();

      if (!appState.mlp)
        ImGui::BeginDisabled();
      if (ImGui::Button("Eval model", ImVec2(ImGuiContentWidth(), 0))) {
        evalMlPToTexture(*appState.mlp.get(), appState.outputImage);
      }
      if (!appState.mlp)
        ImGui::EndDisabled();

      ImGui::Checkbox("Auto eval during training",
                      &appState.autoEvalDuringTraining);

      Utils::Gui::ImageFit(appState.outputImage);
    }
    ImGui::End();

    if (ImGui::Begin("Model builder")) {

      if (appState.isInTraining)
        ImGui::BeginDisabled();

      if (ImGui::Button("Load", ImVec2(ImGuiContentWidth(), 0))) {
      }
      if (ImGui::Button("Save", ImVec2(ImGuiContentWidth(), 0))) {
      }
      ImGui::Separator();

      if (ImGui::Button("Build MLP", ImVec2(ImGuiContentWidth(), 0))) {
        appState.trainingSteps = 0;
        appState.avgMSE = {};
        // We always have to reconstruct the optimizer
        appState.optimizer.reset();
        appState.mlp.reset();
        appState.mlp = buildCBNR(appState.g, appState.deepLayerWidths,
                                 appState.deepLayerActivationFuncs);
        // Always create en optimizer by using the last specified settings
        createOptimizer(appState);
      }
      ImGui::Text(
          fmt::format("Deep layer number: {}", appState.deepLayerWidths.size())
              .c_str());
      if (ImGui::Button("Add new deep Layer")) {
        appState.deepLayerWidths.push_back(1);
        appState.deepLayerActivationFuncs.push_back(0);
      }
      std::vector<bool> toRemove(appState.deepLayerWidths.size());
      bool update = false;
      for (int i = 0; i < appState.deepLayerWidths.size(); i++) {
        if (ImGui::CollapsingHeader(fmt::format("Layer {} Size", i).c_str())) {
          if (ImGui::InputInt(fmt::format("##Layer {} Size", i).c_str(),
                              &appState.deepLayerWidths[i])) {
            if (appState.deepLayerWidths[i] < 1)
              appState.deepLayerWidths[i] = 1;
          }
          ImGui::Combo(fmt::format("Activation function ##Layer {}", i).c_str(),
                       &appState.deepLayerActivationFuncs[i],
                       appState.activationFuncChoices.data(),
                       static_cast<int>(appState.activationFuncChoices.size()));
          if (ImGui::Button(fmt::format("Remove ##Layer {}", i).c_str())) {
            update = true;
            toRemove[i] = true;
          }
        }
      }
      if (update) {
        for (int i = appState.deepLayerWidths.size() - 1; i >= 0; i--) {
          if (toRemove[i]) {
            appState.deepLayerWidths.erase(appState.deepLayerWidths.begin() +
                                           i);
            appState.deepLayerActivationFuncs.erase(
                appState.deepLayerActivationFuncs.begin() + i);
          }
        }
      }
      if (appState.isInTraining)
        ImGui::EndDisabled();
    }
    ImGui::End();

    if (ImGui::Begin("Training settings")) {

      if (appState.mlp) {

        if (appState.isInTraining)
          ImGui::BeginDisabled();

        if (ImGui::Combo("Optimizer", &appState.currentOptimizer,
                         appState.optimizerChoices.data(),
                         static_cast<int>(appState.optimizerChoices.size()))) {
          appState.optimizer.reset();
          createOptimizer(appState);
        }
        if (ImGui::Combo("Loss", &appState.currentLoss,
                         appState.lossChoices.data(),
                         static_cast<int>(appState.lossChoices.size()))) {
          appState.optimizer.reset();
          createOptimizer(appState);
        }

        if (appState.currentOptimizer == BATCH_OP) {
          if (ImGui::InputDouble("Learning rate", &appState.lastLearningRate)) {
            static_cast<ml::BatchOptimizer *>(appState.optimizer.get())
                ->learningRate = appState.lastLearningRate;
          }
          if (ImGui::InputDouble("Momentum", &appState.lastMomentum)) {
            static_cast<ml::BatchOptimizer *>(appState.optimizer.get())
                ->momentum = appState.lastMomentum;
          }
        } else if (appState.currentOptimizer == SGD_OP) {
          if (ImGui::InputDouble("Learning rate", &appState.lastLearningRate)) {
            static_cast<ml::SGDOptimizer *>(appState.optimizer.get())
                ->learningRate = appState.lastLearningRate;
          }
          if (ImGui::InputDouble("Momentum", &appState.lastMomentum)) {
            static_cast<ml::SGDOptimizer *>(appState.optimizer.get())
                ->momentum = appState.lastMomentum;
          }
          if (ImGui::Checkbox("Nesterov", &appState.lastIsNesterov)) {
            static_cast<ml::SGDOptimizer *>(appState.optimizer.get())
                ->nesterov = appState.lastIsNesterov;
          }
        }

        if (ImGui::Button("Reset", ImVec2(ImGuiContentWidth(), 0))) {
          appState.optimizer.reset();
          createOptimizer(appState);
        }

        if (appState.isInTraining) {
          ImGui::EndDisabled();
          if (ImGui::Button("Stop training", ImVec2(ImGuiContentWidth(), 0))) {
            appState.isInTraining = false;
            window.setTargetFps(60);
          }
        } else if (ImGui::Button("Start training",
                                 ImVec2(ImGuiContentWidth(), 0))) {
          if (!appState.inputImage.has_value()) {
            askLoadInputImage(appState);
          }
          if (appState.inputImage.has_value()) {
            // Set the dataset and rung the training
            appState.optimizer->setDataset(appState.dataSet.value());
            appState.isInTraining = true;
            window.setTargetFps(0);
          }
        }

      } else {
        ImGui::TextColored({1, 0, 0, 1}, "LOAD OR BUILD A MODEL FIRST");
      }
    }
    ImGui::End();

    if (ImGui::Begin("Model performance")) {
      if (!appState.avgMSE.empty()) {
        ImGui::Text(fmt::format("Latest MSE: {:.6}",
                                appState.avgMSE[appState.avgMSE.size() - 1])
                        .c_str());
        ImGui::Separator();
        ImGui::Text(fmt::format("Total training steps: {}",
                                static_cast<int>(appState.trainingSteps))
                        .c_str());
        ImGui::Separator();
        if (appState.isInTraining)
          ImPlot::SetNextAxesToFit();
        ImPlot::BeginPlot("Image MSE");
        ImPlot::PlotLine("MSE##data", appState.avgMSE.data(),
                         static_cast<int>(appState.avgMSE.size()));
        ImPlot::EndPlot();
      } else
        ImGui::TextColored({1, 0, 0, 1}, "WAITING FOR TRAINING DATA");
    }
    ImGui::End();

    if (ImGui::Begin("Debug")) {
      ImGui::LabelText("Frame Time", "%.2f ms", GetFrameTime() * 1000.0f);
      ImGui::Separator();
      ImGui::LabelText("Total compute nodes", "%d nodes", appState.g.nbNodes());
      ImGui::Separator();
      ImGui::LabelText("Total compute edges", "%d edges",
                       appState.g.getEdges().size());
      ImGui::Separator();
      if (appState.mlp) {
        ImGui::LabelText("MLP weights", "%d weights",
                         appState.mlp->nbWeights());
        ImGui::Separator();
        ImGui::LabelText("MLP compute nodes", "%d nodes",
                         appState.mlp->nbNodes());
        ImGui::Separator();
        ImGui::LabelText("MLP compute edges", "%d edges",
                         appState.mlp->getEdges().size());
        ImGui::Separator();
      }
      if (appState.optimizer) {
        ImGui::LabelText("Optimizer compute nodes", "%d nodes",
                         appState.optimizer->nbNodes());
        ImGui::Separator();
        ImGui::LabelText("Optimizer compute edges", "%d edges",
                         appState.optimizer->getEdges().size());
        ImGui::Separator();
        ImGui::LabelText("Loss compute nodes", "%d nodes",
                         appState.optimizer->getLoss().nbNodes());
        ImGui::Separator();
        ImGui::LabelText("Loss compute edges", "%d edges",
                         appState.optimizer->getLoss().getEdges().size());
        ImGui::Separator();
      }
    }
    ImGui::End();
    rlImGuiEnd();
    EndDrawing();
  }

  Utils::Gui::ShutDown();
  return 0;
}
