#pragma once

#include "Layer.h"
#include "raylib.h"
#include <functional>
#include <memory>
#include <vector>

namespace ML {

    struct ErrorMetric {
        std::function<double(const std::vector<double>&, const std::vector<double>&)> eval;
        std::function<double(const std::vector<double>&, const std::vector<double>&, int i)> pdiff;
        double diff(const std::vector<double>& predicted, const std::vector<double>& trueValues) const;
    };


    class MLP {
        std::vector<std::unique_ptr<Layer> > _layers;
    public:
        double learningRate = 0.001;
        void clearCachedValues() const;

        [[nodiscard]] std::vector<double> eval(const std::vector<double> &inputs) const;
        void grad(const ErrorMetric& errorMetric, const std::vector<double>& trueValues) const;
        void updateWeights() const;


        void addLayer(std::unique_ptr<Layer> layer);

        void addLayer(int size, const ActivationFunc &activation, bool biasNeuron = false);

        void draw(Vector2 topLeft, float r, float layerPadding, float neuronPadding) const;

        void buildDrawLayout(Vector2 topLeft, float r, float layerPadding, float neuronPadding) const;

        [[nodiscard]] Vector2 computeDrawSize(float r, float layerPadding, float neuronPadding) const;

        static const ErrorMetric& MSE();
        static const ErrorMetric& MAE();
    };
}
