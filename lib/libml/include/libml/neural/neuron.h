#pragma once

#include "libml/compute/graph.h"
#include "activations.h"
#include "aggregations.h"
#include <memory>
#include <vector>

namespace ml {

class Neuron final : public ComputeSubGraph {
public:
  explicit Neuron(IComputeGraph &graph, std::unique_ptr<Aggregate> aggregate,
                  std::unique_ptr<Activation> activation);
  [[nodiscard]] ComputeNode &output() const;
  void addInput(ComputeNode &node, bool addWeight = true);
  void connectToNeuron(Neuron &other) const;
  [[nodiscard]] ComputeNode &getWeight(int index) const;
  [[nodiscard]] int nbWeights() const;

private:
  std::unique_ptr<Aggregate> _aggregate;
  std::unique_ptr<Activation> _activation;
  std::vector<ComputeNode *> _inputWeights;
};

} // namespace ml
