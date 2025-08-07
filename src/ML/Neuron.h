#pragma once

#include <algorithm>
#include <map>
#include <optional>
#include "raylib.h"
#include <functional>
#include <set>

namespace ML {

    struct ActivationFunc {
        std::function<double(double)> eval;
        std::function<double(double)> diff;
    };

    class Neuron {
    protected:
        // input neuron, weight
        std::map<Neuron *, double> _inputs;
        // input neuron, gradient
        std::map<Neuron*, std::pair<std::optional<double>, unsigned int>> _weightsGradients;

        // output neuron
        std::set<Neuron*> _outputs;


        std::optional<double> _value = {};
        std::optional<double> _aggregate = {};
        std::optional<double> _gradient = {};
        bool _backpropagated = false;

        const ActivationFunc& _activation;

        void _weightsGrad();

    public:
        explicit Neuron(const ActivationFunc& activation);
        //explicit Neuron(double constantValue);

        Vector2 position = {0, 0};

        void setValue(double val);
        void setGradient(double gradient);

        double eval();

        double grad();

        void updateWeights(double learningRate);


        void clearCachedValue();

        void connect(Neuron *other, double weight);

        void disconnect(Neuron *other);

        std::map<Neuron *, double> &inputs();

        void draw(float r) const;

        static Vector2 computeDrawSize(float r);

        static const ActivationFunc& ReLu();
        static const ActivationFunc& Identity();
        static const ActivationFunc& Sigmoid();
    };
}
