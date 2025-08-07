#include "Layer.h"

#include <execution>
#include <ranges>

#include "effolkronium/random.hpp"

using Random = effolkronium::random_static;

namespace ML {
    Layer::Layer(const int size, const ActivationFunc &activation, const bool biasNeuron) : _weightDistribution{
        0, std::sqrt(2.0 / static_cast<double>(size))
    } {
        _neurons.reserve(size);
        for (int i = 0; i < size; ++i)
            _neurons.push_back(std::make_unique<Neuron>(activation));
        // bias neuron
        if (biasNeuron) {
            _biasNeuron = std::make_unique<Neuron>(ML::Neuron::Identity());
            _biasNeuron.value()->setValue(1.0);
            for (std::unique_ptr<Neuron> &n: _neurons)
                _biasNeuron.value()->connect(n.get(), Random::get(_weightDistribution));
        }
    }

    void Layer::connect(Layer *other) {
        for (std::unique_ptr<Neuron> &src: _neurons)
            for (std::unique_ptr<Neuron> &dst: other->_neurons)
                src->connect(dst.get(), Random::get(_weightDistribution));
    }

    void Layer::draw(const float r) const {
        for (const std::unique_ptr<Neuron> &n: _neurons)
            n->draw(r);
        if (_biasNeuron.has_value())
            _biasNeuron.value()->draw(r);
    }

    Vector2 Layer::computeDrawSize(const float r, const float padding) const {
        return {
            (_biasNeuron.has_value() ? 4 * r : 2 * r) + padding,
            (2 * r + padding) * static_cast<float>(_neurons.size()) - padding
        };
    }

    void Layer::buildDrawLayout(const Vector2 position, const float r, const float neuronPadding) const {
        float dy = r;
        for (auto &n: _neurons) {
            n->position = {position.x + r, position.y + dy};
            dy += 2 * r + neuronPadding;
        }
        if (_biasNeuron.has_value())
            _biasNeuron.value()->position = {position.x - 2 * r - neuronPadding, position.y};
    }

    void Layer::clearCachedValues() const {
        for (const auto &neuron: _neurons)
            neuron->clearCachedValue();
    }

    std::vector<double> Layer::eval() const {
        std::vector<double> v(_neurons.size(), 0.0);
        for (int i=0; i < v.size(); ++i)
            v[i] = _neurons[i]->eval();
        return std::move(v);
    }

    void Layer::grad() const {
        for (const auto& n : _neurons)
            n->grad();
    }

    void Layer::setValues(const std::vector<double> &inputs) const {
        for (int i = 0; i < inputs.size(); ++i)
            _neurons[i]->setValue(inputs[i]);
    }

    void Layer::setGradients(const std::vector<double> &gradients) const {
        for (int i = 0; i < gradients.size(); ++i)
            _neurons[i]->setGradient(gradients[i]);
    }

    void Layer::updateWeights(const double learningRate) const {
        for (const auto &n: _neurons) {
            n->updateWeights(learningRate);
        }
    }
}
