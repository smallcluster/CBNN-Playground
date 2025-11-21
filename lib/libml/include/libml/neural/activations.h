#pragma once

#include "libml/compute/graph.h"
#include "libml/compute/nodes.h"

namespace ml {

class Activation : public ComputeSubGraph {
public:
  virtual void setInput(ComputeNode &node) = 0;
  virtual ComputeNode &output() = 0;

protected:
  explicit Activation(IComputeGraph &graph);
};

class ReLUActivation final : public Activation {
public:
  explicit ReLUActivation(IComputeGraph &graph);
  void setInput(ComputeNode &node) override;
  ComputeNode &output() override;

private:
  ComputeNode &_relu;
};

class SigmoidActivation final : public Activation {
public:
  explicit SigmoidActivation(IComputeGraph &graph);
  void setInput(ComputeNode &node) override;
  ComputeNode &output() override;

private:
  ComputeNode &_sigmoid;
};

class IdentityActivation final : public Activation {
public:
  explicit IdentityActivation(IComputeGraph &graph);
  void setInput(ComputeNode &node) override;
  ComputeNode &output() override;

private:
  ComputeNode &_id;
};

} // namespace ml