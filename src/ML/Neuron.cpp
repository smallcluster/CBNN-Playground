#include "Neuron.h"

#include <numeric>
#include <ranges>
#include "fmt/format.h"

#include "raymath.h"

#include "../Utils/ColorsUtils.h"

namespace ML {
    static const std::function<double(double)> IdActivation = [](const double x) { return x; };
    static const std::function<double(double)> ReluActivation = [](const double x) { return std::max(0.0, x); };

    Neuron::Neuron(const std::function<double(double)> &activation) : _activation(activation) {
    };

    void Neuron::setValue(double val) {
        _value = {val};
    }

    double Neuron::eval() {
        if (_value.has_value())
            return _value.value();
        std::vector<double> inputs;
        inputs.reserve(_inputs.size());
        for (auto &[neuron, weight]: _inputs)
            inputs.push_back(neuron->eval() * weight);
        _value = _activation(std::accumulate(inputs.begin(), inputs.end(), 0.0));
        return _value.value();
    }

    void Neuron::clearCachedValue() {
        _value = {};
    }

    void Neuron::connect(Neuron *other, const double weight) {
        _outputs[other] = weight;
        other->_inputs[this] = weight;
    }

    void Neuron::disconnect(Neuron *other) {
        _outputs.erase(other);
        other->_inputs.erase(this);
    }

    std::map<Neuron *, double> &Neuron::inputs() {
        return _inputs;
    }

    std::map<Neuron *, double> &Neuron::outputs() {
        return _outputs;
    }

    void Neuron::draw(const float r) const {
        for (Neuron *key: _outputs | std::views::keys) {
            const float weight = Clamp(static_cast<float>(_outputs.at(key)), -1.0f, 1.0f);
            const Color c = Utils::Colors::UniformGradient(weight, -1.0f, 1.0f, {RED, DARKGRAY, GREEN});
            DrawLine(static_cast<int>(position.x), static_cast<int>(position.y), static_cast<int>(key->position.x),
                     static_cast<int>(key->position.y), c);
            //DrawLineBezier(position, key->position, 1.0f, c);
        }
        DrawCircle(static_cast<int>(position.x), static_cast<int>(position.y), r, WHITE);

        const std::string txt = fmt::format("{:.3}", _value.value_or(0));
        DrawText(txt.c_str(), static_cast<int>(position.x - MeasureText(txt.c_str(), static_cast<int>(r)) / 2.0),
                 static_cast<int>(position.y - r / 2.0), static_cast<int>(r), GRAY);
    }

    Vector2 Neuron::computeDrawSize(const float r) {
        return {2 * r, 2 * r};
    }

    const std::function<double(double)> &Neuron::ReLu() {
        return ReluActivation;
    }

    const std::function<double(double)> &Neuron::Identity() {
        return IdActivation;
    }
}
