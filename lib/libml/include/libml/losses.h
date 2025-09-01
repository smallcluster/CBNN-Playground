#pragma once

#include "compute/graph.h"
#include "compute/nodes.h"

namespace ml {

class Loss : public ComputeSubGraph {
public:
  virtual void addInput(ComputeNode &predicted, ComputeNode &trueValue) = 0;
protected:
  explicit Loss(IComputeGraph &graph);
};

class L2Loss : public Loss {
public:
  explicit L2Loss(IComputeGraph &graph);
  void addInput(ComputeNode &predicted, ComputeNode &trueValue) override;
protected:
  AddNode &sumNode() const;
private:
  AddNode &_sum;
};

class MSELoss final : public L2Loss {
public:
  void addInput(ComputeNode &predicted, ComputeNode &trueValue) override;
  explicit MSELoss(IComputeGraph &graph);
private:
  CteDivideNode& _div;
};

class L1Loss final : public Loss {
public:
  explicit L1Loss(IComputeGraph &graph);
  void addInput(ComputeNode &predicted, ComputeNode &trueValue) override;
private:
  AddNode& _sum;
};

} // namespace ml