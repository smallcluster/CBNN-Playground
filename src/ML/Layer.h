#pragma once

#include <vector>
#include <memory>
#include <random>

#include "raylib.h"

#include "Neuron.h"



namespace ML {
    class Layer {
    protected:
        std::vector<std::unique_ptr<Neuron> > _neurons;
        std::normal_distribution<double> _weightDistribution;

    public:
        explicit Layer(int size);

        void connect(Layer *other, std::mt19937_64& randomGenerator);

        void draw(float r) const;

        [[nodiscard]] Vector2 computeDrawSize(float r, float padding) const;

        void buildDrawLayout(Vector2 position, float r, float neuronPadding) const;

        void clearCachedValues() const;

        [[nodiscard]] std::vector<double> eval() const;

        void setValues(const std::vector<double> &inputs) const;
    };
}
