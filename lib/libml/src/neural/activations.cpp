#include "libml/neural/activations.h"

namespace ml {

Activation::Activation(IComputeGraph &graph) : ComputeSubGraph(graph) {}

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
} // namespace ml