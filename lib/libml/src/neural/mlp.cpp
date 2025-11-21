#include "libml/neural/mlp.h"
#include "libml/neural/neuron.h"

namespace ml {

// MLP
MLP::MLP(IComputeGraph &graph, const std::vector<LayerBuilder> &layers)
    : ComputeSubGraph(graph) {

  // Create all layers
  for (LayerBuilder b : layers) {
    auto layer = b.build(*this);
    _layers.push_back(layer.release());
  }

  // Create constant inputs nodes
  for (int i = 0; i < _layers[0]->size(); ++i) {
    ComputeNode &n = ComputeSubGraph::nodeFactory().createConstantNode(0);
    _inputs.push_back(&n);
  }

  // Connect each constant node to its corresponding neuron in the input layer
  for (int i = 0; i < _inputs.size(); ++i) {
    _layers[0]->getNeuron(i).addInput(*_inputs[i], false);
  }

  // Connect each layers
  for (int i = 0; i < _layers.size() - 1; ++i) {
    _layers[i]->connectToLayer(*_layers[i + 1]);
  }

  // Keep a reference to all weights
  for (const auto layer : _layers)
    for (int i = 0; i < layer->nbWeights(); ++i)
      _weights.push_back(&layer->getWeight(i));

  // Keep a reference to all outputs
  const Layer *outLayer = _layers[_layers.size() - 1];
  for (int i = 0; i < outLayer->size(); ++i)
    _outputs.push_back(&outLayer->getNeuron(i).output());
}
MLP::~MLP() {
  for (const auto layer : _layers)
    delete layer;
}
int MLP::nbInputs() const { return static_cast<int>(_inputs.size()); }
int MLP::nbOutputs() const { return static_cast<int>(_outputs.size()); }

ComputeNode &MLP::getOutputNode(const int index) const {
  return *_outputs[index];
}

void MLP::setInput(const double value, const int index) const {
  dynamic_cast<ConstantNode *>(_inputs[index])->set(value);
}

double MLP::getOutput(const int index) const { return _outputs[index]->eval(); }
void MLP::setWeight(const double value, const int index) const {
  dynamic_cast<ConstantNode *>(_weights[index])->set(value);
}
int MLP::nbWeights() const { return static_cast<int>(_weights.size()); }
double MLP::getWeight(const int index) const { return _weights[index]->eval(); }
double MLP::getWeightDiff(const int index) const {
  return _weights[index]->diff();
}

void MLP::eval() const {
  for (int i = 0; i < _outputs.size(); ++i)
    getOutput(i);
}

void MLP::diff() const {
  for (int i = 0; i < _weights.size(); ++i)
    getWeightDiff(i);
}

} // namespace ml