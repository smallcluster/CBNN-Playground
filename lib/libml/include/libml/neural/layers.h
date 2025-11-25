#pragma once

#include "libml/compute/graph.h"
#include "libml/neural/neuron.h"

#include <memory>
#include <vector>

namespace ml {

class Layer : public ComputeSubGraph {
public:
  ~Layer() override;
  void connectToLayer(const Layer &other) const;
  void addInput(ComputeNode &node) const;
  Neuron &getNeuron(int index) const;
  int size() const;
  void addNeuron(Neuron *n);
  ComputeNode &getWeight(int index) const;
  int nbWeights() const;

protected:
  explicit Layer(IComputeGraph &graph);

private:
  std::vector<Neuron *> _neurons;
};

class LayerReLU final : public Layer {
public:
  explicit LayerReLU(IComputeGraph &graph, int size, bool addBias = true);
};

class LayerSigmoid final : public Layer {
public:
  explicit LayerSigmoid(IComputeGraph &graph, int size, bool addBias = true);
};

class LayerIdentity final : public Layer {
public:
  explicit LayerIdentity(IComputeGraph &graph, int size, bool addBias = true);
};

class LayerBuilder {
public:
  enum class Type { ReLu, Sigmoid, Identity };
  int size = 0;
  Type type = Type::Identity;
  bool bias = false;
  LayerBuilder() = default;
  explicit LayerBuilder(int size, Type type, bool addBias);
  std::unique_ptr<Layer> build(IComputeGraph &graph);
};
} // namespace ml