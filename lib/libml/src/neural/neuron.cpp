#include "../../include/libml/neural/neuron.h"

namespace ml {

// Aggregation
SumAggregate::SumAggregate(IComputeGraph &graph)
    : Aggregate(graph), _sum(this->Aggregate::nodeFactory().createAddNode()) {}
void SumAggregate::addInput(ComputeNode &node) { createEdge(node, _sum); }
ComputeNode &SumAggregate::output() { return _sum; }

AvgAggregate::AvgAggregate(IComputeGraph &graph)
    : Aggregate(graph), _avg(this->Aggregate::nodeFactory().createAvgNode()) {}
void AvgAggregate::addInput(ComputeNode &node) { createEdge(node, _avg); }
ComputeNode &AvgAggregate::output() { return _avg; }

// Activation
ReLUActivation::ReLUActivation(IComputeGraph &graph)
    : Activation(graph),
      _relu(this->Activation::nodeFactory().createReLUNode()) {}
void ReLUActivation::setInput(ComputeNode &node) { createEdge(node, _relu, 0); }
ComputeNode &ReLUActivation::output() { return _relu; }

SigmoidActivation::SigmoidActivation(IComputeGraph &graph)
    : Activation(graph),
      _sigmoid(this->Activation::nodeFactory().createReLUNode()) {}
void SigmoidActivation::setInput(ComputeNode &node) {
  createEdge(node, _sigmoid, 0);
}
ComputeNode &SigmoidActivation::output() { return _sigmoid; }

IdentityActivation::IdentityActivation(IComputeGraph &graph)
    : Activation(graph),
      _id(this->Activation::nodeFactory().createIdentityNode()) {}
void IdentityActivation::setInput(ComputeNode &node) {
  createEdge(node, _id, 0);
}
ComputeNode &IdentityActivation::output() { return _id; }

// Neuron
Neuron::Neuron(IComputeGraph &graph, std::unique_ptr<Aggregate> aggregate,
               std::unique_ptr<Activation> activation)
    : ComputeSubGraph(graph), _aggregate(std::move(aggregate)),
      _activation(std::move(activation)) {}

ComputeNode &Neuron::output() const {
  return _activation->output();
}

void Neuron::addInput(ComputeNode &node) {
  ConstantNode &weight = nodeFactory().createConstantNode(1.0, true);
  _inputWeights.push_back(&weight);
  MultNode &m = nodeFactory().createMultNode();
  createEdge(weight, m);
  createEdge(node, m);
  _aggregate->addInput(m);
}
void Neuron::connectToNeuron(Neuron &other) const { other.addInput(output()); }
ComputeNode &Neuron::getWeight(const int index) const {
  return *_inputWeights[index];}
int Neuron::nbWeights() const { return _inputWeights.size();}

} // namespace ml