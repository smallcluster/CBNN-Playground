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
        std::optional<std::unique_ptr<Neuron>> _biasNeuron;
        std::normal_distribution<double> _weightDistribution;

    public:
        Layer(int size, const ActivationFunc &activation, bool biasNeuron = false);

        void connect(Layer *other);

        void draw(float r) const;

        [[nodiscard]] Vector2 computeDrawSize(float r, float padding) const;

        void buildDrawLayout(Vector2 position, float r, float neuronPadding) const;

        void clearCachedValues() const;

        [[nodiscard]] std::vector<double> eval() const;

        void grad() const;

        void setValues(const std::vector<double> &inputs) const;
        void setGradients(const std::vector<double> &gradients) const;

        void updateWeights(double learningRate) const;
    };
}
