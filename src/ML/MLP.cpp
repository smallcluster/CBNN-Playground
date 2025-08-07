#include "MLP.h"

#include <cassert>
#include <numeric>
#include <ranges>

namespace ML {
    void MLP::clearCachedValues() const {
        for (const auto &layer: _layers)
            layer->clearCachedValues();
    }

    std::vector<double> MLP::eval(const std::vector<double> &inputs) const {
        clearCachedValues();
        _layers[0]->setValues(inputs);
        std::vector<double> v;
        for (const auto &layer: _layers)
            v = layer->eval();
        return std::move(v);
    }

    void MLP::grad(const ErrorMetric &errorMetric, const std::vector<double> &trueValues) const {
        const std::vector<double> predicted = _layers[_layers.size() - 1]->eval();
        // Computes all partial derivatives relative to the output layer
        std::vector<double> errorPDiffs(predicted.size());
        auto it = std::views::iota(0, static_cast<int>(predicted.size()));
        std::ranges::transform(it, errorPDiffs.begin(), [&](const int i) {
            return errorMetric.pdiff(predicted, trueValues, i);
        });
        _layers[_layers.size()-1]->setGradients(errorPDiffs);
        // Now run the backpropagation algorithm
        for (int i=static_cast<int>(_layers.size())-1; i >= 0; --i)
            _layers[i]->grad();
    }

    void MLP::updateWeights() const {
        for (const auto& layer : _layers) {
            layer->updateWeights(learningRate);
        }
    }



    void MLP::addLayer(std::unique_ptr<Layer> layer) {
        _layers.push_back(std::move(layer));
        if (_layers.size() >= 2)
            _layers[_layers.size() - 2]->connect(_layers[_layers.size() - 1].get());
    }

    void MLP::addLayer(int size, const ActivationFunc &activation, bool biasNeuron) {
        addLayer(std::make_unique<Layer>(size, activation, biasNeuron));
    }


    void MLP::draw(const Vector2 topLeft, const float r, const float layerPadding, const float neuronPadding) const {
        buildDrawLayout(topLeft, r, layerPadding, neuronPadding);
        for (const auto &layer: std::ranges::views::reverse(_layers))
            layer->draw(r);
    }

    void MLP::buildDrawLayout(const Vector2 topLeft, const float r, const float layerPadding,
                              const float neuronPadding) const {
        const auto [mx, my] = computeDrawSize(r, layerPadding, neuronPadding);
        Vector2 layerPos = topLeft;
        for (const auto &layer: _layers) {
            const auto [sx, sy] = layer->computeDrawSize(r, neuronPadding);
            layerPos.y = topLeft.y + (my - sy) / 2.0f;
            layer->buildDrawLayout(layerPos, r, neuronPadding);
            layerPos.x += sx + layerPadding;
        }
    }

    Vector2 MLP::computeDrawSize(const float r, const float layerPadding, const float neuronPadding) const {
        Vector2 maxSize = {0, 0};
        for (const auto &layer: _layers) {
            const auto [x, y] = layer->computeDrawSize(r, neuronPadding);
            maxSize.x += x + layerPadding;
            maxSize.y = std::min(maxSize.y, y);
        }
        maxSize.x -= layerPadding;
        return maxSize;
    }

    // Error metrics
    double mseEval(const std::vector<double> &inputs, const std::vector<double> &trueValues) {
        double r = 0.0;
        for (int i=0; i < inputs.size(); ++i)
            r += (inputs[i] - trueValues[i]) * (inputs[i] - trueValues[i]);
        return r / static_cast<double>(inputs.size());
    }

    double msePDiff(const std::vector<double> &inputs, const std::vector<double> &trueValues, const int i) {
        return 2.0 * (inputs[i] - trueValues[i]) / static_cast<double>(inputs.size());
    }

    double maeEval(const std::vector<double> &inputs, const std::vector<double> &trueValues) {
        double r = 0.0;
        for (int i=0; i < inputs.size(); ++i)
            r += std::abs(trueValues[i] - inputs[i]);
        return r / static_cast<double>(inputs.size());
    }

    double maePDiff(const std::vector<double> &inputs, const std::vector<double> &trueValues, const int i) {
        if (inputs[i] == trueValues[i])
            return 0;
        return (inputs[i] < trueValues[i] ? -1 : 1 ) / static_cast<double>(inputs.size());
    }

    static const ErrorMetric mseStruct = {mseEval, msePDiff};
    static const ErrorMetric maeStruct = {maeEval, maePDiff};


    const ErrorMetric &MLP::MSE() {
        return mseStruct;
    }

    const ErrorMetric &MLP::MAE() {
        return maeStruct;
    }


    double ErrorMetric::diff(const std::vector<double> &predicted, const std::vector<double> &trueValues) const {
        auto it = std::views::iota(0, static_cast<int>(predicted.size()));
        return std::reduce(it.begin(), it.end(), 0.0, [&](const int i, const int j) {
            return pdiff(predicted, trueValues, i) + pdiff(predicted, trueValues, j);
        });
    }
};
