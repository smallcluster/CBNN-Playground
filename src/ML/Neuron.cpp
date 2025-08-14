#include "Neuron.h"

#include "fmt/format.h"
#include <execution>
#include <numeric>
#include <ranges>

#include "raymath.h"

#include "../Utils/ColorsUtils.h"

namespace ML {

    Neuron::Neuron(const ActivationFunc& activation) : _activation(activation) {
    };

    void Neuron::setValue(double val) {
        _value = val;
    }

    void Neuron::setGradient(double gradient) {
        _gradient = gradient;
        // Compute the weight partial derivative
        _weightsGrad();
    }

    double Neuron::eval() {
        if (_value.has_value())
            return _value.value();

        if (!_aggregate.has_value()) {
            // Return the sum of weighted values
            double g = 0.0;
            for (const auto&[n, weight] : _inputs)
                g += n->eval() * weight;
            _aggregate = g;
        }
        _value = _activation.eval(_aggregate.value());
        return _value.value();
    }

    void Neuron::_weightsGrad() {
        for (auto n: _weightsGradients | std::ranges::views::keys) {
            double nextGradient = _gradient.value() * _activation.diff(_aggregate.value()) * n->eval();
            if (_weightsGradients[n].first.has_value()) {
                _weightsGradients[n].first = (_weightsGradients[n].second*_weightsGradients[n].first.value()+nextGradient) / (_weightsGradients[n].second+1);
                _weightsGradients[n].second = _weightsGradients[n].second + 1;
            } else {
                _weightsGradients[n].first = nextGradient;
                _weightsGradients[n].second = 1;
            }
        }
    }


    double Neuron::grad() {
        if (_gradient.has_value())
            return _gradient.value();
        // Calculate the partial gradient for this neuron activation
        double g = 0;
        for (const auto n : _outputs)
            g += n->grad() * n->_inputs[this] * n->_activation.diff(n->_aggregate.value());
        _gradient = g;
        // Now update the mean partial gradient for all input weights
        _weightsGrad();
        return _gradient.value();
    }

    void Neuron::updateWeights(const double learningRate) {
        // Modify each weight according to the computed mean partial derivative with respect
        // to an optimizer (TODO)
        for (auto [n, weight]: _inputs ) {
            if (_weightsGradients[n].first.has_value()) {
                const double gradient = _weightsGradients[n].first.value();
                _inputs[n] = weight - gradient*learningRate;
                // Clear the avg gradient
                _weightsGradients[n].first = {};
                _weightsGradients[n].second = 0;
            }
        }
    }




    void Neuron::clearCachedValue() {
        _value = {};
        _aggregate = {};
        _gradient = {};
    }

    void Neuron::connect(Neuron *other, const double weight) {
        other->_inputs[this] = weight;
        other->_weightsGradients[this] = {};
        _outputs.insert(other);
    }

    void Neuron::disconnect(Neuron *other) {
        other->_inputs.erase(this);
        other->_weightsGradients.erase(this);
        _outputs.erase(other);
    }

    std::map<Neuron *, double> &Neuron::inputs() {
        return _inputs;
    }

    void Neuron::draw(const float r) const {
        for (Neuron *key: _inputs | std::views::keys) {
            const float weight = Clamp(static_cast<float>(_inputs.at(key)), -1.0f, 1.0f);
            const Color c = Utils::Colors::UniformGradient(weight, -1.0f, 1.0f, {RED, DARKGRAY, GREEN});
            DrawLine(static_cast<int>(position.x), static_cast<int>(position.y), static_cast<int>(key->position.x),
                     static_cast<int>(key->position.y), c);
            //DrawLineBezier(position, key->position, 1.0f, c);
        }
        // DrawCircle(static_cast<int>(position.x), static_cast<int>(position.y), r, WHITE);
        DrawRectangle(static_cast<int>(position.x-r), static_cast<int>(position.y-r), 2*r, 2*r, WHITE);

        const std::string txt = fmt::format("{:.3}", _value.value_or(0));
        DrawText(txt.c_str(), static_cast<int>(position.x - MeasureText(txt.c_str(), static_cast<int>(r)) / 2.0),
                 static_cast<int>(position.y - r / 2.0), static_cast<int>(r), GRAY);
    }

    Vector2 Neuron::computeDrawSize(const float r) {
        return {2 * r, 2 * r};
    }

    // Activation functions
    double reluEval(const double x) {
        return std::max(0.0, x);
    }
    double reluDiff(const double x) {
        return x <= 0 ? 0.0 : 1.0;
    }
    double identityEval(const double x) {
        return x;
    }
    double identityDiff(const double x) {
        return 1.0;
    }
    double sigmoidEval(const double x) {
        return 1.0 / (1+std::exp(-x));
    }
    double sigmoidDiff(const double x) {
        const double v = sigmoidEval(x);
        return v*(1.0-v);
    }
    static const ActivationFunc reluStruct = {reluEval, reluDiff};
    static const ActivationFunc identityStruct = {identityEval, identityDiff};
    static const ActivationFunc sigmoidStruct = {sigmoidEval, sigmoidDiff};

    const ActivationFunc &Neuron::ReLu() {
        return reluStruct;
    }
    const ActivationFunc &Neuron::Identity() {
        return identityStruct;
    }
    const ActivationFunc &Neuron::Sigmoid() {
        return sigmoidStruct;
    }

}
