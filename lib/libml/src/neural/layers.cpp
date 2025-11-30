#include "libml/neural/layers.h"

#include "effolkronium/random.hpp"

using Random = effolkronium::random_static;

namespace ml {

Layer::Layer(IComputeGraph &graph) : ComputeSubGraph(graph) {}

Layer::~Layer() {
  for (const Neuron *n : _neurons)
    delete n;
}

void Layer::connectToLayer(const Layer &other) const {
  std::normal_distribution<double> d{
      0.0, std::sqrt(2.0 / static_cast<double>(_neurons.size()))};
  for (const auto _neuron : _neurons) {
    for (int j = 0; j < other.size(); ++j) {
      double w = Random::get(d);
      _neuron->connectToNeuron(other.getNeuron(j), w);
    }
  }
}
void Layer::addInput(ComputeNode &node) const {
  std::normal_distribution d{
      0.0, std::sqrt(2.0 / static_cast<double>(_neurons.size()))};
  for (const auto n : _neurons) {
    double w = Random::get(d);
    n->addInput(node, true, w);
  }
}

Neuron &Layer::getNeuron(const int index) const { return *_neurons[index]; }
int Layer::size() const { return static_cast<int>(_neurons.size()); }

void Layer::addNeuron(Neuron *n) { _neurons.push_back(n); }

ComputeNode &Layer::getWeight(const int index) const {
  int i = 0;
  for (const auto n : _neurons) {
    for (int j = 0; j < n->nbWeights(); ++j) {
      if (i == index)
        return n->getWeight(j);
      ++i;
    }
  }
  return _neurons[0]->getWeight(0);
}

int Layer::nbWeights() const {
  int total = 0;
  for (const auto n : _neurons)
    total += n->nbWeights();
  return total;
}

// Layer types
LayerReLU::LayerReLU(IComputeGraph &graph, const int size, const bool addBias)
    : Layer(graph) {
  for (int i = 0; i < size; ++i) {
    addNeuron(new NeuronReLu(*this));
  }
  // Bias
  if (addBias) {
    ConstantNode &b = Layer::nodeFactory().createConstantNode(1.0);
    b.setLabelPrefix("B: ");
    addInput(b);
  }
}
LayerSigmoid::LayerSigmoid(IComputeGraph &graph, const int size,
                           const bool addBias)
    : Layer(graph) {
  for (int i = 0; i < size; ++i) {
    addNeuron(new NeuronSigmoid(*this));
  }
  // Bias
  if (addBias) {
    ConstantNode &b = Layer::nodeFactory().createConstantNode(1.0);
    b.setLabelPrefix("B: ");
    addInput(b);
  }
}
LayerIdentity::LayerIdentity(IComputeGraph &graph, const int size,
                             const bool addBias)
    : Layer(graph) {
  for (int i = 0; i < size; ++i) {
    addNeuron(new NeuronIdentity(*this));
  }
  // Bias
  if (addBias) {
    ConstantNode &b = Layer::nodeFactory().createConstantNode(1.0);
    b.setLabelPrefix("B: ");
    addInput(b);
  }
}

// Layer builder
std::unique_ptr<Layer> LayerBuilder::build(IComputeGraph &graph) {
  switch (type) {
  case Type::ReLu:
    return std::make_unique<LayerReLU>(graph, size, bias);
  case Type::Sigmoid:
    return std::make_unique<LayerSigmoid>(graph, size, bias);
  case Type::Identity:
    return std::make_unique<LayerIdentity>(graph, size, bias);
  }
  return std::make_unique<LayerIdentity>(graph, size, bias);
}
LayerBuilder::LayerBuilder(const int size, const Type type, const bool addBias)
    : size(size), type(type), bias(addBias) {}
} // namespace ml
