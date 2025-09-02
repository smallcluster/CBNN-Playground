#pragma once

#include "libml/compute/graph.h"
#include "libml/neural/neuron.h"

#include <vector>
#include <memory>

namespace ml {


class Layer : public ComputeSubGraph {
public:
  ~Layer() override;
  void connectToLayer(const Layer &other) const;
  void addInput(ComputeNode& node) const;
  [[nodiscard]] Neuron& getNeuron(int index) const;
  [[nodiscard]] int size() const;
  void addNeuron(std::unique_ptr<Aggregate> aggregate, std::unique_ptr<Activation> activation);
  [[nodiscard]] ComputeNode& getWeight(int index) const;
  [[nodiscard]] int nbWeights() const;
protected:
  explicit Layer(IComputeGraph& graph);
private:
  std::vector<Neuron *> _neurons;
};

class LayerReLU final : public Layer {
public:
  explicit LayerReLU(IComputeGraph& graph, int size, bool addBias = true);
};

class LayerSigmoid final : public Layer {
public:
  explicit LayerSigmoid(IComputeGraph& graph, int size, bool addBias = true);
};

class LayerIdentity final : public Layer {
public:
  explicit LayerIdentity(IComputeGraph& graph, int size, bool addBias = true);
};


class LayerBuilder {
public:
  enum class Type {
    ReLu,
    Sigmoid,
    Identity
  };
  std::unique_ptr<Layer> build(IComputeGraph& graph);
  explicit LayerBuilder(int size, Type type, bool addBias);
private:
  int _size;
  Type _type;
  bool _bias;
};
}