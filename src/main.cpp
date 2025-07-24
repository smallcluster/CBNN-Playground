#include <map>
#include <ctime>
#include <random>
#include <memory>
#include <optional>
#include <vector>

#include "spdlog/spdlog.h"
#include "raylib.h"
#include "raymath.h"
#include "imgui.h"
#include "rlImGui.h"

#include "ML/Layer.h"
#include "ML/MLP.h"
#include "Utils/GuiUtils.h"
#include "Utils/MathUtils.h"


ML::MLP buildCBNR(const std::vector<int> &layerSizes) {
    ML::MLP model;
    model.addLayer(2, ML::Neuron::Identity()); // Pixel coordinates
    for (const auto &s: layerSizes)
        model.addLayer(s, ML::Neuron::ReLu());
    model.addLayer(3, ML::Neuron::Identity()); // RGB value
    return std::move(model);
}


int main(int argc, char *argv[]) {
    int nbDeepLayers = 2;
    std::vector<int> deepLayerWidths = {6, 6};
    ML::MLP model = buildCBNR(deepLayerWidths);

    bool animatedEval = false;
    int evalProgress = 0;
    int currentX = 0;
    int currentY = 0;
    std::vector<Color> outputPixels;

    float neuronRadius = 24.0f;
    float neuronPadding = neuronRadius;
    float layerPadding = neuronRadius * 4.0f * 1.5f;

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_ALWAYS_RUN | FLAG_WINDOW_RESIZABLE);
    InitWindow(1920, 1080, "COOL CHIC VIS");
    SetTargetFPS(60);
    Utils::Gui::Setup();

    Camera2D camera = {0};
    camera.rotation = 0.0f;
    camera.zoom = 1.0f;

    std::optional<Image> imageInput;
    std::optional<Texture2D> textureInput;
    std::optional<Texture2D> textureOutput;


    while (!WindowShouldClose()) {
        camera.target = {0, 0};
        camera.offset = {static_cast<float>(GetScreenWidth()) / 2.0f, static_cast<float>(GetScreenHeight()) / 2.0f};
        // Camera zoom controls
        // Uses log scaling to provide consistent zoom speed
        camera.zoom = expf(logf(camera.zoom) + ((float) GetMouseWheelMove() * 0.1f));
        if (camera.zoom > 3.0f) camera.zoom = 3.0f;
        else if (camera.zoom < 0.1f) camera.zoom = 0.1f;

        // Image drag and drop
        if (IsFileDropped()) {
            const FilePathList droppedFiles = LoadDroppedFiles();
            // Get only the first path
            std::string path(droppedFiles.paths[0]);
            spdlog::info("Loading file {}", path);
            UnloadDroppedFiles(droppedFiles);

            // Load the image and texture
            if (imageInput.has_value()) {
                UnloadImage(imageInput.value());
            }
            imageInput = LoadImage(path.c_str());
            ImageFormat(&imageInput.value(), PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
            if (textureInput.has_value())
                UnloadTexture(textureInput.value());
            textureInput = LoadTextureFromImage(imageInput.value());

            if (textureOutput.has_value())
                UnloadTexture(textureOutput.value());
            Image imageOutput = GenImageColor(imageInput->width, imageInput->height, BLACK);
            textureOutput = LoadTextureFromImage(imageOutput);
            UnloadImage(imageOutput);
        }
        //------------- DRAWING
        BeginDrawing();
        ClearBackground({30, 31, 34, 255});

        // animated eval
        if (animatedEval) {
            auto values = model.eval({static_cast<double>(currentX), static_cast<double>(currentY)});

            // clamped output
            Utils::Math::clamp(values, 0.0, 1.0);

            const int w = imageInput.value().width;
            const int h = imageInput.value().height;
            outputPixels[w * h - 1 - evalProgress] = {
                static_cast<unsigned char>(values[0] * 255.0),
                static_cast<unsigned char>(values[1] * 255.0),
                static_cast<unsigned char>(values[2] * 255.0),
                255
            };
            UpdateTexture(textureOutput.value(), &outputPixels[0]);

            currentX++;
            if (currentX == w) {
                currentX = 0;
                currentY++;
            }

            DrawRectangle(0, GetScreenHeight() - 16, GetScreenWidth(), 16, BLACK);
            const float factor = 1.0f - static_cast<float>(evalProgress) / static_cast<float>(w * h - 1);
            const float barWidth = static_cast<float>(GetScreenWidth()) * factor;
            BeginScissorMode(0, GetScreenHeight() - 16, static_cast<int>(barWidth), 16);
            DrawRectangleGradientH(0, GetScreenHeight() - 16, GetScreenWidth(), 16, RED, GREEN);
            EndScissorMode();

            evalProgress--;
            if (evalProgress < 0) {
                SetTargetFPS(60);
                animatedEval = false;
            }
        }

        BeginMode2D(camera);
        const Vector2 modelDrawSize = model.computeDrawSize(neuronRadius, layerPadding, neuronPadding);
        model.draw(modelDrawSize * -1 / 2.0f, neuronRadius, layerPadding, neuronPadding);
        EndMode2D();

        rlImGuiBegin();

        ImGui::BeginMainMenuBar();
        ImGui::Text(fmt::format("Frame time: {:.2}ms", GetFrameTime() * 1000.0f).c_str());
        ImGui::EndMainMenuBar();

        ImGui::Begin("Model usage");
        if (imageInput.has_value()) {
            if (ImGui::Button("Eval")) {
                animatedEval = false;
                SetTargetFPS(60);
                const int w = imageInput.value().width;
                const int h = imageInput.value().height;
                outputPixels.clear();
                outputPixels.reserve(w * h);
                // Compute values
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        auto values = model.eval({static_cast<double>(x), static_cast<double>(y)});

                        // clamped output
                        Utils::Math::clamp(values, 0.0, 1.0);

                        outputPixels.push_back({
                            static_cast<unsigned char>(Clamp(static_cast<float>(values[0]) * 255, 0, 255)),
                            static_cast<unsigned char>(Clamp(static_cast<float>(values[1]) * 255, 0, 255)),
                            static_cast<unsigned char>(Clamp(static_cast<float>(values[2]) * 255, 0, 255)),
                            255
                        });
                    }
                }
                UpdateTexture(textureOutput.value(), &outputPixels[0]);
            }
            if (ImGui::Button("Animated eval")) {
                animatedEval = true;
                const int w = imageInput.value().width;
                const int h = imageInput.value().height;
                outputPixels.clear();
                outputPixels.reserve(w * h);
                for (int i = 0; i < w * h; i++)
                    outputPixels.push_back(BLACK);
                evalProgress = w * h - 1;
                currentX = 0;
                currentY = 0;
                SetTargetFPS(0);
            }
        } else {
            ImGui::TextColored({1, 0, 0, 1}, "NO INPUT IMAGE");
        }

        ImGui::End();


        ImGui::Begin("Model config");
        if (ImGui::Button("Build MLP")) {
            model = buildCBNR(deepLayerWidths);
        }
        ImGui::Text(fmt::format("Deep layer number: {}", nbDeepLayers).c_str());
        if (ImGui::Button("Add new deep Layer")) {
            deepLayerWidths.push_back(1);
            nbDeepLayers++;
        }
        std::vector<bool> toRemove(nbDeepLayers);
        bool update = false;
        for (int i = 0; i < nbDeepLayers; i++) {
            if (ImGui::CollapsingHeader(fmt::format("Layer {} Size", i).c_str())) {
                if (ImGui::InputInt(fmt::format("##Layer {} Size", i).c_str(), &deepLayerWidths[i])) {
                    if (deepLayerWidths[i] < 1)
                        deepLayerWidths[i] = 1;
                }
                if (ImGui::Button(fmt::format("Remove ##Layer {}", i).c_str())) {
                    update = true;
                    toRemove[i] = true;
                }
            }
        }
        if (update) {
            for (int i = nbDeepLayers - 1; i >= 0; i--) {
                if (toRemove[i]) {
                    deepLayerWidths.erase(deepLayerWidths.begin() + i);
                    nbDeepLayers--;
                }
            }
        }
        ImGui::End();

        ImGui::Begin("Input Image");
        if (textureInput.has_value()) {
            ImGui::Text(fmt::format("Size: {}x{}", textureInput->width, textureInput->height).c_str());
            Utils::Gui::ImageFit(textureInput.value());
        } else {
            ImGui::TextColored({0, 1, 0, 1}, "DRAG & DROP IMAGE TO LOAD IT");
        }
        ImGui::End();

        ImGui::Begin("Output Image");
        if (textureOutput.has_value()) {
            Utils::Gui::ImageFit(textureOutput.value());
        } else {
            ImGui::TextColored({1, 0, 0, 1}, "NO INPUT IMAGE");
        }
        ImGui::End();


        rlImGuiEnd();
        EndDrawing();
    }
    Utils::Gui::ShutDown();
    CloseWindow();
    return 0;
}
