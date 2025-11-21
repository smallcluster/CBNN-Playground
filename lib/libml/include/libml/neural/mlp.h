#pragma once

#include <vector>

#include "libml/compute/graph.h"
#include "libml/neural/layers.h"
#include "libml/neural/losses.h"

namespace ml {

class MLP final : public ComputeSubGraph {
public:
  explicit MLP(IComputeGraph &graph, const std::vector<LayerBuilder> &layers);
  ~MLP() override;
  [[nodiscard]] int nbInputs() const;
  [[nodiscard]] int nbOutputs() const;
  [[nodiscard]] ComputeNode &getOutputNode(int index) const;
  [[nodiscard]] int nbWeights() const;
  void setInput(double value, int index) const;
  void setWeight(double value, int index) const;
  [[nodiscard]] double getOutput(int index) const;
  [[nodiscard]] double getWeight(int index) const;
  [[nodiscard]] double getWeightDiff(int index) const;
  void eval() const;
  void diff() const;

private:
  std::vector<Layer *> _layers;
  std::vector<ComputeNode *> _inputs;
  std::vector<ComputeNode *> _outputs;
  std::vector<ComputeNode *> _weights;
};

} // namespace ml