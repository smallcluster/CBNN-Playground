#pragma once

#include "activations.h"
#include "aggregations.h"
#include "libml/compute/graph.h"
#include <memory>
#include <vector>

namespace ml {

class Neuron : public ComputeSubGraph {
public:
  ComputeNode &output() const;
  void addInput(ComputeNode &node, bool addWeight, double weight);
  void connectToNeuron(Neuron &other, double weight) const;
  ComputeNode &getWeight(int index) const;
  int nbWeights() const;

protected:
  explicit Neuron(IComputeGraph &graph);
  std::unique_ptr<Aggregate> _aggregate;
  std::unique_ptr<Activation> _activation;

private:
  std::vector<ComputeNode *> _inputWeights;
};

class NeuronReLu final : public Neuron {
public:
  explicit NeuronReLu(IComputeGraph &graph);
};

class NeuronIdentity final : public Neuron {
public:
  explicit NeuronIdentity(IComputeGraph &graph);
};

class NeuronSigmoid final : public Neuron {
public:
  explicit NeuronSigmoid(IComputeGraph &graph);
};

} // namespace ml
