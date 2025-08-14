#pragma once

#include "compute/graph.h"
#include "compute/nodes.h"

namespace ml {

class Loss : public ComputeSubGraph {
public:
  explicit Loss(ComputeGraph &graph);
  virtual void addInput(ComputeNode &predicted, ComputeNode &trueValue) = 0;
};

class L2Loss : public Loss {
public:
  explicit L2Loss(ComputeGraph &graph);
  void addInput(ComputeNode &predicted, ComputeNode &trueValue) override;
protected:
  AddNode &sumNode() const;
private:
  AddNode &_sum;
};

class MSELoss final : public L2Loss {
public:
  void addInput(ComputeNode &predicted, ComputeNode &trueValue) override;
  explicit MSELoss(ComputeGraph &graph);
private:
  CteDivideNode& _div;
};

class L1Loss final : public Loss {
public:
  explicit L1Loss(ComputeGraph &graph);
  void addInput(ComputeNode &predicted, ComputeNode &trueValue) override;
private:
  AddNode& _sum;
};

} // namespace ml