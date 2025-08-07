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
#include "implot.h"
#include "rlImGui.h"

#include "ML/Layer.h"
#include "Utils/GuiUtils.h"
#include "Utils/MathUtils.h"
#include "ML/MLP.h"

#include "effolkronium/random.hpp"
using Random = effolkronium::random_static;

const ML::ActivationFunc &getActivationFunction(const int index) {
    switch (index) {
        case 1: return ML::Neuron::Sigmoid();
        case 2: return ML::Neuron::Identity();
        default: return ML::Neuron::ReLu();
    }
}


ML::MLP buildCBNR(const std::vector<int> &layerSizes,
                  const std::vector<int> &activationFunctions) {
    ML::MLP model;
    model.addLayer(2, ML::Neuron::Identity()); // Pixel coordinates
    for (int i = 0; i < layerSizes.size(); ++i)
        model.addLayer(layerSizes[i], getActivationFunction(activationFunctions[i]), true);
    model.addLayer(3, ML::Neuron::Identity()); // RGB value
    return std::move(model);
}

Color normalizedToColor(const std::vector<double> &values) {
    return {
        static_cast<unsigned char>(std::clamp(values[0], 0.0, 1.0) * 255),
        static_cast<unsigned char>(std::clamp(values[1], 0.0, 1.0) * 255),
        static_cast<unsigned char>(std::clamp(values[2], 0.0, 1.0) * 255),
        255
    };
}

std::vector<double> colorToNormalized(const Color &color) {
    return {
        static_cast<double>(color.r) / 255.0,
        static_cast<double>(color.g) / 255.0,
        static_cast<double>(color.b) / 255.0
    };
}

std::vector<Color> predictColors(const ML::MLP &model, const int width, const int height) {
    std::vector<Color> pixels;
    pixels.reserve(width * height);
    // Compute values
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const auto values = model.eval({static_cast<double>(x)/width, static_cast<double>(y)/height});
            pixels.push_back(normalizedToColor(values));
        }
    }
    return std::move(pixels);
}

int main(int argc, char *argv[]) {
    int nbDeepLayers = 2;
    std::vector<int> deepLayerWidths = {6, 6};
    std::vector<int> deepLayerActivationFuncs = {0, 0};
    const std::vector<const char *> activationFuncChoices = {"ReLu", "Sigmoid", "Identity"};

    ML::MLP model = buildCBNR(deepLayerWidths, deepLayerActivationFuncs);

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
    camera.target = {0, 0};
    camera.rotation = 0.0f;
    camera.zoom = 1.0f;

    std::optional<Vector2> panOrigin;

    std::optional<Image> imageInput;
    std::optional<Color *> colorsInput;
    std::optional<Texture2D> textureInput;
    std::optional<Texture2D> textureOutput;
    std::vector<double> avgMSE;

    bool training = false;


    const auto stepTrain = [&](const std::function<void(const std::vector<double> &trueValues)> &gradFunc,
                               const std::function<void()> &postGradFunc) {
        animatedEval = false;

        const int w = imageInput.value().width;
        const int h = imageInput.value().height;

        // Randomize pixels coords
        std::vector<std::pair<int, int> > coords;
        coords.reserve(w * h);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                coords.emplace_back(x, y);
        Random::shuffle(coords);

        // Reset predicted pixels colors
        outputPixels.clear();
        outputPixels.reserve(w * h);
        outputPixels.resize(w * h);

        const auto &[mseEval, msePDiff] = ML::MLP::MSE();
        std::vector<double> mseValues;
        mseValues.reserve(w * h);

        for (const auto &[x, y]: coords) {
            // Compute predicted values
            auto values = model.eval({static_cast<double>(x)/w, static_cast<double>(y)/h});
            // Compute gradient
            const Color c = colorsInput.value()[y * w + x];
            const std::vector<double> trueValues = colorToNormalized(c);
            gradFunc(trueValues);
            // Record pixel MSE
            mseValues.push_back(mseEval(values, trueValues));
            // Update output image
            outputPixels[y * w + x] = normalizedToColor(values);
        }
        postGradFunc();
        // Record image MSE
        const double imageMSE = std::accumulate(mseValues.begin(), mseValues.end(), 0.0) / (w * h);
        avgMSE.push_back(imageMSE);
        // Update predicted texture
        UpdateTexture(textureOutput.value(), &outputPixels[0]);
    };

    const auto stepTrainSGD = [&] {
        stepTrain([&](const std::vector<double> &trueValues) {
            model.grad(ML::MLP::MSE(), trueValues);
            model.updateWeights();
        }, [] {
        });
    };

    const auto stepTrainBatch = [&] {
        stepTrain([&](const std::vector<double> &trueValues) {
            model.grad(ML::MLP::MSE(), trueValues);
        }, [&] { model.updateWeights(); });
    };

    std::function<void()> currentStepTrain = stepTrainBatch;
    const std::vector<const char *> trainMethod = {"Batch", "SGD"};
    int selectedTrainMethod = 0;



    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground({30, 31, 34, 255});

        // Camera panning
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            panOrigin = {camera.target.x*camera.zoom + (static_cast<float>(GetMouseX()) - camera.offset.x), camera.target.y*camera.zoom + (static_cast<float>(GetMouseY()) - camera.offset.y)};
        }
        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            panOrigin = {};
        }
        if (panOrigin.has_value()) {
            camera.target.x = -(static_cast<float>(GetMouseX())-camera.offset.x-panOrigin.value().x)/camera.zoom;
            camera.target.y = -(static_cast<float>(GetMouseY())- camera.offset.y-panOrigin.value().y)/camera.zoom;
        }
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

            // Load the image, texture and colors
            if (colorsInput.has_value())
                UnloadImageColors(colorsInput.value());
            if (imageInput.has_value())
                UnloadImage(imageInput.value());
            if (textureInput.has_value())
                UnloadTexture(textureInput.value());

            imageInput = LoadImage(path.c_str());
            ImageFormat(&imageInput.value(), PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
            textureInput = LoadTextureFromImage(imageInput.value());
            colorsInput = LoadImageColors(imageInput.value());

            // Create a new output texture
            if (textureOutput.has_value())
                UnloadTexture(textureOutput.value());
            Image imageOutput = GenImageColor(imageInput->width, imageInput->height, BLACK);
            textureOutput = LoadTextureFromImage(imageOutput);
            UnloadImage(imageOutput);
        }

        // animated continuous training
        if (training) {
            currentStepTrain();
        }

        // animated eval
        if (animatedEval) {
            const int w = imageInput.value().width;
            const int h = imageInput.value().height;

            auto values = model.eval({static_cast<double>(currentX)/w, static_cast<double>(currentY)/h});
            outputPixels[w * h - 1 - evalProgress] = normalizedToColor(values);
            UpdateTexture(textureOutput.value(), &outputPixels[0]);
            currentX++;
            if (currentX == w) {
                currentX = 0;
                currentY++;
            }
            // Progress bar
            DrawRectangle(0, GetScreenHeight() - 16, GetScreenWidth(), 16, BLACK);
            const float factor = 1.0f - static_cast<float>(evalProgress) / static_cast<float>(w * h - 1);
            const float barWidth = static_cast<float>(GetScreenWidth()) * factor;
            BeginScissorMode(0, GetScreenHeight() - 16, static_cast<int>(barWidth), 16);
            DrawRectangleGradientH(0, GetScreenHeight() - 16, GetScreenWidth(), 16, RED, GREEN);
            EndScissorMode();

            // Stop condition
            evalProgress--;
            if (evalProgress < 0) {
                SetTargetFPS(60);
                animatedEval = false;
            }
        }

        // model drawing
        BeginMode2D(camera);
        const Vector2 modelDrawSize = model.computeDrawSize(neuronRadius, layerPadding, neuronPadding);
        model.draw(modelDrawSize * -1 / 2.0f, neuronRadius, layerPadding, neuronPadding);
        EndMode2D();

        // Gui drawing
        rlImGuiBegin();

        // Menu bar
        ImGui::BeginMainMenuBar();
        ImGui::Text(fmt::format("Frame time: {:.2}ms", GetFrameTime() * 1000.0f).c_str());
        ImGui::EndMainMenuBar();


        // Model performance
        if (ImGui::Begin("Model performance")) {
            if (!avgMSE.empty()) {
                ImGui::Text(fmt::format("Latest MSE: {:.6}", avgMSE[avgMSE.size() - 1]).c_str());
                ImGui::Text(fmt::format("Total training steps: {}", static_cast<int>(avgMSE.size())).c_str());
                if (training)
                    ImPlot::SetNextAxesToFit();
                ImPlot::BeginPlot("Image MSE");

                constexpr int mseViewSize = 1000;
                std::vector<double> mseView;
                const int toCopy = std::min(static_cast<int>(avgMSE.size()), mseViewSize);
                mseView.reserve(toCopy);

                for (int i = 0; i < toCopy; ++i)
                    mseView.push_back(avgMSE[static_cast<int>(avgMSE.size())-toCopy-1+i]);

                ImPlot::PlotLine("MSE##data", mseView.data(), static_cast<int>(mseView.size()));
                ImPlot::EndPlot();
            } else
                ImGui::TextColored({1, 0, 0, 1}, "WAITING FOR TRAINING DATA");
        }
        ImGui::End();

        // Model usage
        ImGui::Begin("Model usage");
        if (imageInput.has_value()) {
            const int w = imageInput.value().width;
            const int h = imageInput.value().height;
            if (ImGui::Button("Eval")) {
                animatedEval = false;
                SetTargetFPS(60);
                outputPixels = predictColors(model, w, h);
                UpdateTexture(textureOutput.value(), &outputPixels[0]);
            }
            if (ImGui::Button("Animated eval")) {
                animatedEval = true;
                outputPixels.clear();
                outputPixels.reserve(w * h);
                std::ranges::fill(outputPixels, BLACK);
                evalProgress = w * h - 1;
                currentX = 0;
                currentY = 0;
                SetTargetFPS(0);
            }
            if (!training && ImGui::Button("Train one step")) {
                SetTargetFPS(60);
                currentStepTrain();
            }
            if (training && ImGui::Button("Stop training")) {
                training = false;
                SetTargetFPS(60);
            }
            if (!training && ImGui::Button("Train")) {
                training = true;
                SetTargetFPS(0);
            }
            ImGui::Text("Learning Rate:");
            ImGui::InputDouble("##Learning rate input", &(model.learningRate));
            if (ImGui::Combo("Training method", &selectedTrainMethod, trainMethod.data(),
                             static_cast<int>(trainMethod.size()))) {
                if (selectedTrainMethod == 0)
                    currentStepTrain = stepTrainBatch;
                else
                    currentStepTrain = stepTrainSGD;
            }
        } else {
            ImGui::TextColored({1, 0, 0, 1}, "NO INPUT IMAGE");
        }
        ImGui::End();

        // Model config
        ImGui::Begin("Model config");
        if (ImGui::Button("Build MLP")) {
            model = buildCBNR(deepLayerWidths, deepLayerActivationFuncs);
            avgMSE = {};
        }
        ImGui::Text(fmt::format("Deep layer number: {}", nbDeepLayers).c_str());
        if (ImGui::Button("Add new deep Layer")) {
            deepLayerWidths.push_back(1);
            deepLayerActivationFuncs.push_back(0);
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
                ImGui::Combo(fmt::format("Activation function ##Layer {}", i).c_str(), &deepLayerActivationFuncs[i],
                             activationFuncChoices.data(), static_cast<int>(activationFuncChoices.size()));
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
                    deepLayerActivationFuncs.erase(deepLayerActivationFuncs.begin() + i);
                    nbDeepLayers--;
                }
            }
        }
        ImGui::End();

        // Input Image
        ImGui::Begin("Input Image");
        if (textureInput.has_value()) {
            ImGui::Text(fmt::format("Size: {}x{}", textureInput->width, textureInput->height).c_str());
            Utils::Gui::ImageFit(textureInput.value());
        } else {
            ImGui::TextColored({0, 1, 0, 1}, "DRAG & DROP IMAGE TO LOAD IT");
        }
        ImGui::End();

        // Output Image
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
