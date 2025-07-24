#include "Layer.h"

#include <execution>
#include <ranges>

#include "effolkronium/random.hpp"

using Random = effolkronium::random_static;

namespace ML {
    Layer::Layer(const int size, const std::function<double(double)> &activation) : _weightDistribution{
        0, std::sqrt(2.0 / static_cast<double>(size))
    } {
        _neurons.reserve(size);
        for (int i = 0; i < size; ++i)
            _neurons.push_back(std::make_unique<Neuron>(activation));
    }

    void Layer::connect(Layer *other) {
        for (std::unique_ptr<Neuron> &src: _neurons)
            for (std::unique_ptr<Neuron> &dst: other->_neurons)
                src->connect(dst.get(), Random::get(_weightDistribution));
    }

    void Layer::draw(const float r) const {
        for (const std::unique_ptr<Neuron> &n: _neurons)
            n->draw(r);
    }

    Vector2 Layer::computeDrawSize(const float r, const float padding) const {
        return {2 * r, (2 * r + padding) * static_cast<float>(_neurons.size()) - padding};
    }

    void Layer::buildDrawLayout(const Vector2 position, const float r, const float neuronPadding) const {
        float dy = r;
        for (auto &n: _neurons) {
            n->position = {position.x + r, position.y + dy};
            dy += 2 * r + neuronPadding;
        }
    }

    void Layer::clearCachedValues() const {
        for (const auto &neuron: _neurons)
            neuron->clearCachedValue();
    }

    std::vector<double> Layer::eval() const {
        std::vector<double> v(_neurons.size(), 0.0);

        std::transform(std::execution::par, _neurons.begin(), _neurons.end(), v.begin(),
                       [](const std::unique_ptr<Neuron> &n) { return n->eval(); });

        return std::move(v);
    }

    void Layer::setValues(const std::vector<double> &inputs) const {
        for (int i = 0; i < inputs.size(); ++i)
            _neurons[i]->setValue(inputs[i]);
    }
}
