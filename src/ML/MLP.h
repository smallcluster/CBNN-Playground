#pragma once

#include <vector>
#include <memory>
#include <functional>

#include "Layer.h"
#include "raylib.h"

namespace ML {
    class MLP {
        std::vector<std::unique_ptr<Layer> > _layers;
    public:
        void clearCachedValues() const;

        [[nodiscard]] std::vector<double> eval(const std::vector<double> &inputs) const;

        void addLayer(int size, std::mt19937_64& randomGenerator);

        void draw(Vector2 topLeft, float r, float layerPadding, float neuronPadding) const;

        void buildDrawLayout(Vector2 topLeft, float r, float layerPadding, float neuronPadding) const;

        [[nodiscard]] Vector2 computeDrawSize(float r, float layerPadding, float neuronPadding) const;
    };
}
