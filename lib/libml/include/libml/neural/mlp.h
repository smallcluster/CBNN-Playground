#pragma once

#include <vector>

#include "../compute/graph.h"
#include "../losses.h"
#include "../neuron.h"

namespace ml {

class Layer : public ComputeSubGraph {
public:
  ~Layer() override;
  void connectToLayer(const Layer &other) const;
  void addInput(ComputeNode& node) const;
  Neuron& getNeuron(int index) const;
  int size() const;
  void addNeuron(std::unique_ptr<Aggregate> aggregate, std::unique_ptr<Activation> activation);
  ComputeNode& getWeight(int index) const;
  int nbWeights() const;
protected:
  explicit Layer(IComputeGraph& graph);
private:
  std::vector<Neuron *> _neurons;
  ComputeNode& _bias;
};

class LayerReLU final : public Layer {
public:
  explicit LayerReLU(IComputeGraph& graph, int size);
};

class LayerSigmoid final : public Layer {
public:
  explicit LayerSigmoid(IComputeGraph& graph, int size);
};

class LayerIdentity final : public Layer {
public:
  explicit LayerIdentity(IComputeGraph& graph, int size);
};


class LayerBuilder {
public:
  enum class Type {
    ReLu,
    Sigmoid,
    Identity
  };
  std::unique_ptr<Layer> build(IComputeGraph& graph);
  explicit LayerBuilder(int size, Type type = Type::Identity);
private:
  int _size;
  Type _type;
};


class MLP {
public:
  explicit MLP(int inputs, const std::vector<LayerBuilder> &layers, std::unique_ptr<Loss> loss);
  ~MLP();

  void setInput(double value, int index) const;
  double getOutput(int index) const;
  void setWeight(double value, int index) const;
  int nbWeights() const;
  double getWeight(int index) const;
  double getWeightGradient(int index) const;

  void eval();
  void computeGradient();
  void clearGradientHistory();
private:
  std::vector<double> _outputs;
  std::vector<double> _gradients;

  std::vector<Layer *> _layers;
  std::vector<ComputeNode*> _inputs;
  std::vector<ComputeNode*> _weights;
  std::unique_ptr<Loss> _loss;
  ComputeGraph _graph;
};

} // namespace ml