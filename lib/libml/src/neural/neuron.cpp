#include "libml/neural/neuron.h"

#include <iostream>

namespace ml {
Neuron::Neuron(IComputeGraph &graph) : ComputeSubGraph(graph) {}

ComputeNode &Neuron::output() const { return _activation->output(); }

void Neuron::addInput(ComputeNode &node, const bool addWeight) {
  if (addWeight) {
    ConstantNode &weight = nodeFactory().createConstantNode(1.0);
    _inputWeights.push_back(&weight);
    MultNode &m = nodeFactory().createMultNode();
    createEdge(weight, m, {});
    createEdge(node, m, {});
    _aggregate->addInput(m);
  } else
    _aggregate->addInput(node);
}

void Neuron::connectToNeuron(Neuron &other) const {
  other.addInput(output(), true);
}
ComputeNode &Neuron::getWeight(const int index) const {
  return *_inputWeights[index];
}
int Neuron::nbWeights() const { return static_cast<int>(_inputWeights.size()); }

NeuronReLu::NeuronReLu(IComputeGraph &graph) : Neuron(graph) {
  _aggregate = std::make_unique<SumAggregate>(*this);
  _activation = std::make_unique<ReLUActivation>(*this);
  _activation->setInput(_aggregate->output());
}

NeuronIdentity::NeuronIdentity(IComputeGraph &graph) : Neuron(graph) {
  _aggregate = std::make_unique<SumAggregate>(*this);
  _activation = std::make_unique<IdentityActivation>(*this);
  _activation->setInput(_aggregate->output());
}

NeuronSigmoid::NeuronSigmoid(IComputeGraph &graph) : Neuron(graph) {
  _aggregate = std::make_unique<SumAggregate>(*this);
  _activation = std::make_unique<SigmoidActivation>(*this);
  _activation->setInput(_aggregate->output());
}

} // namespace ml
