#include "libml/neural/neuron.h"

namespace ml {
Neuron::Neuron(IComputeGraph &graph) : ComputeSubGraph(graph) {}

ComputeNode &Neuron::output() const { return _activation->output(); }

void Neuron::addInput(ComputeNode &node, const bool addWeight,
                      const double weight) {
  if (addWeight) {
    ConstantNode &weightNodde = nodeFactory().createConstantNode(weight);
    weightNodde.setLabelPrefix("W: ");
    _inputWeights.push_back(&weightNodde);
    MultNode &m = nodeFactory().createMultNode();
    createEdge(weightNodde, m, 0);
    createEdge(node, m, 1);
    _aggregate->addInput(m);
  } else
    _aggregate->addInput(node);
}

void Neuron::connectToNeuron(Neuron &other, double weight) const {
  other.addInput(output(), true, weight);
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
