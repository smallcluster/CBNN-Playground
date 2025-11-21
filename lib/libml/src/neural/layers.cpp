#include "libml/neural/layers.h"

namespace ml {

Layer::Layer(IComputeGraph &graph) : ComputeSubGraph(graph) {}

Layer::~Layer() {
  for (const Neuron *n : _neurons)
    delete n;
}

void Layer::connectToLayer(const Layer &other) const {
  for (const auto _neuron : _neurons) {
    for (int j = 0; j < other.size(); ++j) {
      _neuron->connectToNeuron(other.getNeuron(j));
    }
  }
}
void Layer::addInput(ComputeNode &node) const {
  for (const auto n : _neurons) {
    n->addInput(node, true);
  }
}

Neuron &Layer::getNeuron(const int index) const { return *_neurons[index]; }
int Layer::size() const { return static_cast<int>(_neurons.size()); }

void Layer::addNeuron(Neuron* n) {
  _neurons.push_back(n);
}

ComputeNode &Layer::getWeight(const int index) const {
  int i = 0;
  for (const auto n : _neurons) {
    if (i + n->nbWeights() > index)
      return n->getWeight(index - i);
    i += n->nbWeights();
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
  if (addBias)
    addInput(Layer::nodeFactory().createConstantNode(1.0));
}
LayerSigmoid::LayerSigmoid(IComputeGraph &graph, const int size,
                           const bool addBias)
    : Layer(graph) {
  for (int i = 0; i < size; ++i) {
    addNeuron(new NeuronSigmoid(*this));
  }
  // Bias
  if (addBias)
    addInput(Layer::nodeFactory().createConstantNode(1.0));
}
LayerIdentity::LayerIdentity(IComputeGraph &graph, const int size,
                             const bool addBias)
    : Layer(graph) {
  for (int i = 0; i < size; ++i) {
    addNeuron(new NeuronIdentity(*this));
  }
  // Bias
  if (addBias)
    addInput(Layer::nodeFactory().createConstantNode(1.0));
}

// Layer builder
std::unique_ptr<Layer> LayerBuilder::build(IComputeGraph &graph) {
  switch (_type) {
  case Type::ReLu:
    return std::make_unique<LayerReLU>(graph, _size, _bias);
  case Type::Sigmoid:
    return std::make_unique<LayerSigmoid>(graph, _size, _bias);
  case Type::Identity:
    return std::make_unique<LayerIdentity>(graph, _size, _bias);
  }
  return std::make_unique<LayerIdentity>(graph, _size, _bias);
}
LayerBuilder::LayerBuilder(const int size, const Type type, const bool addBias)
    : _size(size), _type(type), _bias(addBias) {}
} // namespace ml
