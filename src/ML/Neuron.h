#pragma once

#include <map>
#include <optional>
#include "raylib.h"
#include <functional>

namespace ML {
    class Neuron {
    protected:
        std::map<Neuron *, double> _inputs;
        std::map<Neuron *, double> _outputs;

        std::optional<double> _value = {};
        std::function<double(double)> _activation;

    public:
        explicit Neuron(const std::function<double(double)> &activation);
        //explicit Neuron(double constantValue);

        Vector2 position = {0, 0};

        void setValue(double val);

        double eval();

        void clearCachedValue();

        void connect(Neuron *other, double weight);

        void disconnect(Neuron *other);

        std::map<Neuron *, double> &inputs();

        std::map<Neuron *, double> &outputs();

        void draw(float r) const;

        static Vector2 computeDrawSize(float r);

        static const std::function<double(double)> &ReLu();

        static const std::function<double(double)> &Identity();
    };
}
