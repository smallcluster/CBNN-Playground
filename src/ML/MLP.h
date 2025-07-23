#pragma once

#include <vector>
#include <memory>
#include <functional>

#include "Layer.h"
#include "raylib.h"

namespace ML {
    class MLP {
        std::vector<std::unique_ptr<Layer> > _layers;
        const std::function<double()> &_weightInitializer;

    public:
        explicit MLP(const std::function<double()> &weightInitializer);

        void clearCachedValues() const;

        void eval(const std::vector<double> &inputs) const;

        void addLayer(std::unique_ptr<Layer> layer);

        void draw(Vector2 topLeft, float r, float layerPadding, float neuronPadding) const;

        void buildDrawLayout(Vector2 topLeft, float r, float layerPadding, float neuronPadding) const;

        [[nodiscard]] Vector2 computeDrawSize(float r, float layerPadding, float neuronPadding) const;
    };
}
