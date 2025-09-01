#include "../../include/libml/neural/mlp.h"

namespace ml {

// Layer

Layer::Layer(IComputeGraph &graph)
    : ComputeSubGraph(graph),
      _bias(this->ComputeSubGraph::nodeFactory().createConstantNode(1.0,
                                                                    false)) {}
Layer::~Layer() {
  for (const Neuron *n : _neurons)
    delete n;
}

void Layer::connectToLayer(const Layer &other) const {
  for (int i = 0; i < _neurons.size(); ++i) {
    for (int j = 0; j < other.size(); ++j) {
      _neurons[i]->connectToNeuron(other.getNeuron(j));
    }
  }
}
void Layer::addInput(ComputeNode &node) const {
  for (const auto n : _neurons) {
    n->addInput(node);
  }
}

Neuron &Layer::getNeuron(const int index) const { return *_neurons[index]; }
int Layer::size() const { return _neurons.size(); }

void Layer::addNeuron(std::unique_ptr<Aggregate> aggregate,
                      std::unique_ptr<Activation> activation) {
  _neurons.push_back(
      new Neuron(*this, std::move(aggregate), std::move(activation)));
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

LayerReLU::LayerReLU(IComputeGraph &graph, const int size) : Layer(graph) {
  for (int i = 0; i < size; ++i) {
    std::unique_ptr<Aggregate> agg = std::make_unique<SumAggregate>(graph);
    std::unique_ptr<Activation> acc = std::make_unique<ReLUActivation>(graph);
    addNeuron(std::move(agg), std::move(acc));
  }
}
LayerSigmoid::LayerSigmoid(IComputeGraph &graph, const int size)
    : Layer(graph) {
  for (int i = 0; i < size; ++i) {
    std::unique_ptr<Aggregate> agg = std::make_unique<SumAggregate>(graph);
    std::unique_ptr<Activation> acc =
        std::make_unique<SigmoidActivation>(graph);
    addNeuron(std::move(agg), std::move(acc));
  }
}
LayerIdentity::LayerIdentity(IComputeGraph &graph, const int size)
    : Layer(graph) {
  for (int i = 0; i < size; ++i) {
    std::unique_ptr<Aggregate> agg = std::make_unique<SumAggregate>(graph);
    std::unique_ptr<Activation> acc =
        std::make_unique<IdentityActivation>(graph);
    addNeuron(std::move(agg), std::move(acc));
  }
}

// Layer builder
std::unique_ptr<Layer> LayerBuilder::build(IComputeGraph &graph) {
  switch (_type) {
  case Type::ReLu:
    return std::make_unique<LayerReLU>(graph, _size);
  case Type::Sigmoid:
    return std::make_unique<LayerSigmoid>(graph, _size);
  case Type::Identity:
    return std::make_unique<LayerIdentity>(graph, _size);
  }
  return std::make_unique<LayerIdentity>(graph, _size);
}
LayerBuilder::LayerBuilder(const int size, const Type type)
    : _size(size), _type(type) {}

// MLP
MLP::MLP(const int inputs, const std::vector<LayerBuilder> &layers, std::unique_ptr<Loss> loss) {
  // Setup input nodes
  for (int i=0; i < inputs; ++i) {
    ComputeNode& n = _graph.nodeFactory().createConstantNode(0, false);
    _inputs.push_back(&n);
  }
  // Create layers
  for (LayerBuilder b : layers) {
    auto layer = b.build(_graph);
    _layers.push_back(layer.get());
    layer.release();
  }
  // Connect inputs to the first layer
  for (const auto i : _inputs) {
    _layers[0]->addInput(*i);
  }
  // Connect each layers
  for (int i=0; i < _layers.size()-1; ++i) {
    _layers[i]->connectToLayer(*_layers[i+1]);
  }

  // Keep a reference to all weights
  for (const auto layer : _layers)
    for (int i=0; i < layer->nbWeights(); ++i)
      _weights.push_back(&layer->getWeight(i));

  // Pre allocate weights gradient buffer
  _gradients.resize(_layers[_layers.size()-1]->size(), 0.0);
}
MLP::~MLP() {
  for (const auto layer : _layers)
    delete layer;
}
void MLP::setInput(const double value, const int index) const {
  static_cast<ConstantNode*>(_inputs[index])->set(value);
}

double MLP::getOutput(const int index) const { return _outputs[index]; }
void MLP::setWeight(const double value, const int index) const {
  static_cast<ConstantNode *>(_weights[index])->set(value);
}
int MLP::nbWeights() const {
  return _weights.size();
}
double MLP::getWeight(const int index) const {
  return _weights[index]->eval();
}
double MLP::getWeightGradient(const int index) const {
  return _gradients[index];
}

void MLP::eval() {
  _graph.clearCache();
  const Layer * outLayer = _layers[_layers.size()-1];
  for (int i=0; i < outLayer->size(); ++i)
    _outputs[i] = outLayer->getNeuron(i).output().eval();
}

void MLP::computeGradient() {
  for (int i=0; i < _weights.size(); ++i)
    _gradients[i] = _weights[i]->diff();
}

void MLP::clearGradientHistory() {
  _graph.clearDiffHistory();
}


} // namespace ml