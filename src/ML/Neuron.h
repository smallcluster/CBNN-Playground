#pragma once

#include <map>
#include <optional>
#include <vector>
#include "raylib.h"

namespace ML {
    class Neuron {
    protected:
        std::map<Neuron *, double> _inputs;
        std::map<Neuron *, double> _outputs;

        std::optional<double> _value = {};

        [[nodiscard]] double _aggregate(const std::vector<double> &inputs) const;

    public:
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
    };
}
