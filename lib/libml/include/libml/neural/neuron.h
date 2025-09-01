#pragma once

#include "../compute/graph.h"
#include <memory>
#include <vector>

namespace ml {


class Aggregate : public ComputeSubGraph {
public:
  virtual void addInput(ComputeNode& node) = 0;
  virtual ComputeNode& output() = 0;
protected:
  explicit Aggregate(IComputeGraph& graph);
};

class SumAggregate final : public Aggregate {
public:
  explicit SumAggregate(IComputeGraph& graph);
  void addInput(ComputeNode& node) override;
  ComputeNode& output() override;
private:
  ComputeNode& _sum;
};

class AvgAggregate final : public Aggregate {
public:
  explicit AvgAggregate(IComputeGraph& graph);
  void addInput(ComputeNode& node) override;
  ComputeNode& output() override;
private:
  ComputeNode& _avg;
};

class Activation : public ComputeSubGraph {
public:
  virtual void setInput(ComputeNode& node) = 0;
  virtual ComputeNode& output() = 0;
protected:
  explicit Activation(IComputeGraph& graph);
};

class ReLUActivation final : public Activation {
public:
  explicit ReLUActivation(IComputeGraph& graph);
  void setInput(ComputeNode& node) override;
  ComputeNode& output() override;
private:
  ComputeNode& _relu;
};

class SigmoidActivation final : public Activation {
public:
  explicit SigmoidActivation(IComputeGraph& graph);
  void setInput(ComputeNode& node) override;
  ComputeNode& output() override;
private:
  ComputeNode& _sigmoid;
};

class IdentityActivation final : public Activation {
public:
  explicit IdentityActivation(IComputeGraph& graph);
  void setInput(ComputeNode& node) override;
  ComputeNode& output() override;
private:
  ComputeNode& _id;
};


class Neuron final : public ComputeSubGraph {
public:
  explicit Neuron(IComputeGraph& graph, std::unique_ptr<Aggregate> aggregate, std::unique_ptr<Activation> activation);
  ComputeNode& output() const;
  void addInput(ComputeNode& node);
  void connectToNeuron(Neuron& other) const;
  ComputeNode& getWeight(int index) const;
  int nbWeights() const;
private:
  std::unique_ptr<Aggregate> _aggregate;
  std::unique_ptr<Activation> _activation;
  std::vector<ComputeNode*> _inputWeights;
};


} // namespace ml
