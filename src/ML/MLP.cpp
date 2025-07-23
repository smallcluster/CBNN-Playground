#include "MLP.h"

namespace ML {
    MLP::MLP(const std::function<double()> &weightInitializer) : _weightInitializer(weightInitializer) {
    }

    void MLP::clearCachedValues() const {
        for (const auto &layer: _layers) {
            layer->clearCachedValues();
        }
    }

    void MLP::eval(const std::vector<double> &inputs) const {
        clearCachedValues();
        _layers[0]->setValues(inputs);
        _layers[_layers.size() - 1]->eval();
    }

    void MLP::addLayer(std::unique_ptr<Layer> layer) {
        _layers.push_back(std::move(layer));
        if (_layers.size() >= 2) {
            for (int i = static_cast<int>(_layers.size()) - 2; i < _layers.size() - 1; i++) {
                _layers[i]->connect(_layers[i + 1].get(), _weightInitializer);
            }
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
