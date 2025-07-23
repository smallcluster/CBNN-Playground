#pragma once

#include <vector>
#include <memory>
#include <functional>

#include "raylib.h"

#include "Neuron.h"


namespace ML {
    class Layer {
    protected:
        std::vector<std::unique_ptr<Neuron> > _neurons;

    public:
        explicit Layer(int size);

        void connect(Layer *other, const std::function<double()> &weightInitializer);

        void draw(float r) const;

        [[nodiscard]] Vector2 computeDrawSize(float r, float padding) const;

        void buildDrawLayout(Vector2 position, float r, float neuronPadding) const;

        void clearCachedValues() const;

        void eval() const;

        void setValues(const std::vector<double> &inputs) const;
    };
}
