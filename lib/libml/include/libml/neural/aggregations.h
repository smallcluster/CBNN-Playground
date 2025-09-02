#pragma once

#include "libml/compute/graph.h"
#include "libml/compute/nodes.h"

namespace ml {
class Aggregate : public ComputeSubGraph {
public:
  virtual void addInput(ComputeNode &node) = 0;
  virtual ComputeNode &output() = 0;

protected:
  explicit Aggregate(IComputeGraph &graph);
};

class SumAggregate final : public Aggregate {
public:
  explicit SumAggregate(IComputeGraph &graph);
  void addInput(ComputeNode &node) override;
  ComputeNode &output() override;

private:
  ComputeNode &_sum;
};

class AvgAggregate final : public Aggregate {
public:
  explicit AvgAggregate(IComputeGraph &graph);
  void addInput(ComputeNode &node) override;
  ComputeNode &output() override;

private:
  ComputeNode &_avg;
};
} // namespace ml