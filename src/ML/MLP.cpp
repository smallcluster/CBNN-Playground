#include "MLP.h"

namespace ML {

    void MLP::clearCachedValues() const {
        for (const auto &layer: _layers) {
            layer->clearCachedValues();
        }
    }

    std::vector<double> MLP::eval(const std::vector<double> &inputs) const {
        clearCachedValues();
        _layers[0]->setValues(inputs);
        return std::move(_layers[_layers.size() - 1]->eval());
    }

    void MLP::addLayer(int size, std::mt19937_64& randomGenerator) {
        _layers.push_back(std::make_unique<Layer>(size));
        if (_layers.size() >= 2) {
            _layers[_layers.size()-2]->connect(_layers[_layers.size()-1].get(), randomGenerator);
        }
    }

    void MLP::draw(const Vector2 topLeft, const float r, const float layerPadding, const float neuronPadding) const {
        buildDrawLayout(topLeft, r, layerPadding, neuronPadding);
        for (const auto &layer: _layers) {
            layer->draw(r);
        }
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
};
