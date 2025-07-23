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
#include  "implot.h"
#include "rlImGui.h"


#include "Utils/ImGuiUtils.h"

#include "ML/Layer.h"
#include "ML/MLP.h"


int main(int argc, char *argv[]) {
    unsigned int seed = time(nullptr);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> urd(-1, 1);
    std::function<double()> weightInitializer = [&urd, &rng] { return urd(rng); };


    ML::MLP model(weightInitializer);
    int inputSize = 4;
    model.addLayer(std::make_unique<ML::Layer>(inputSize));
    model.addLayer(std::make_unique<ML::Layer>(7));
    model.addLayer(std::make_unique<ML::Layer>(10));
    model.addLayer(std::make_unique<ML::Layer>(5));

    std::vector<double> defaultInput;
    defaultInput.reserve(inputSize);
    for (int i = 0; i < inputSize; ++i) {
        defaultInput.push_back(weightInitializer());
    }

    float neuronRadius = 24.0f;
    float neuronPadding = neuronRadius;
    float layerPadding = neuronRadius * 4.0f * 1.5f;

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_ALWAYS_RUN | FLAG_WINDOW_RESIZABLE);
    InitWindow(1920, 1080, "COOL CHIC VIS");

    Camera2D camera = {0};
    camera.rotation = 0.0f;
    camera.zoom = 1.0f;

    std::optional<Image> imageInput;
    std::optional<Texture2D> textureInput;
    std::optional<Image> imageOutput;
    std::optional<Texture2D> textureOutput;


    SetTargetFPS(60);
    rlImGuiSetup(false);
    ImPlot::CreateContext();
    Utils::Gui::SetupStyle(false);


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
            if (textureInput.has_value())
                UnloadTexture(textureInput.value());

            textureInput = LoadTextureFromImage(imageInput.value());

            // Reset output image to black
            if (imageOutput.has_value())
                UnloadImage(imageOutput.value());
            if (textureOutput.has_value())
                UnloadTexture(textureOutput.value());
            imageOutput = GenImageColor(imageInput->width, imageInput->height, BLACK);
            textureOutput = LoadTextureFromImage(imageOutput.value());
        }

        //------------- DRAWING
        BeginDrawing();
        ClearBackground({30, 31, 34, 255});

        BeginMode2D(camera);
        const Vector2 modelDrawSize = model.computeDrawSize(neuronRadius, layerPadding, neuronPadding);
        model.draw(modelDrawSize * -1 / 2.0f, neuronRadius, layerPadding, neuronPadding);
        EndMode2D();


        rlImGuiBegin();

        ImPlot::ShowDemoWindow();


        ImGui::BeginMainMenuBar();
        static bool check = false;
        ImGui::Checkbox("Neural network", &check);
        static bool check2 = false;
        ImGui::Checkbox("Latent img", &check2);
        ImGui::EndMainMenuBar();


        ImGui::Begin("Testing");
        if (ImGui::Button("Eval")) {
            model.eval(defaultInput);
        }
        ImGui::End();

        ImGui::Begin("Input Image");
        if (textureInput.has_value()) {
            Utils::Gui::ImageFit(textureInput.value());
        } else {
            ImGui::TextColored({1, 0, 0, 1}, "DRAG & DROP IMAGE");
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
    ImPlot::DestroyContext();
    rlImGuiShutdown();

    CloseWindow();
    return 0;
}
