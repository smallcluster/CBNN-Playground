#pragma once

#include <vector>

#include "compute/graph.h"

namespace ml {

class Neuron {
public:
  ~Neuron();
  void connect(Neuron &other);
  void disconnect(Neuron &other);

private:
  std::vector<ComputeNode *> _nodes;
  std::vector<ComputeNode *> _weights;
  ComputeGraph &_graph;
  Neuron();
};

class Layer {
public:
  ~Layer();
  void connect(Layer &other);
  void disconnect(Layer &other);

private:
  std::vector<Neuron *> _neurons;
  ComputeGraph &_graph;
  Layer();
};

class MLP {
private:
  std::vector<Layer *> _layers;
  ComputeGraph _graph;
};

} // namespace ml