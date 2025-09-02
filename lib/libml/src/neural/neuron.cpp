#include "libml/neural/neuron.h"

namespace ml {
Neuron::Neuron(IComputeGraph &graph, std::unique_ptr<Aggregate> aggregate,
               std::unique_ptr<Activation> activation)
    : ComputeSubGraph(graph), _aggregate(std::move(aggregate)),
      _activation(std::move(activation)) {}

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

void Neuron::connectToNeuron(Neuron &other) const { other.addInput(output()); }
ComputeNode &Neuron::getWeight(const int index) const {
  return *_inputWeights[index];
}
int Neuron::nbWeights() const { return static_cast<int>(_inputWeights.size()); }

} // namespace ml