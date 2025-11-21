#pragma once

#include "libml/neural/dataset.h"
#include "libml/neural/losses.h"
#include "libml/neural/mlp.h"

#include <memory>
#include <random>
#include <vector>

namespace ml {

class ContinuousMean {
public:
  void add(double value);
  [[nodiscard]] double get() const;
  [[nodiscard]] int size() const;

private:
  double _value = 0.0;
  int _size = 0;
};

class Optimizer : public ComputeSubGraph {
public:
  virtual bool optimize() = 0;

protected:
  explicit Optimizer(MLP &mlp, DataSet &dataSet, std::unique_ptr<Loss> loss);
  void _forward();
  void _backward() const;
  virtual int nextTrainingIndex() = 0;
  void setLoss(std::unique_ptr<Loss> loss);
  MLP &_mlp;
  DataSet &_dataSet;
  std::unique_ptr<Loss> _loss;
  std::vector<ComputeNode *> _trueValues;
};

class BatchOptimizer final : public Optimizer {
public:
  explicit BatchOptimizer(MLP &mlp, DataSet &dataSet,
                          std::unique_ptr<Loss> loss,
                          double learningRate = 0.01, double momentum = 0.0);
  bool optimize() override;

protected:
  int nextTrainingIndex() override;

private:
  int _currentInput = 0;
  double _learningRate;
  double _momentum;
  std::vector<double> _velocities;
  std::vector<ContinuousMean> _avgGradient;
};

class SGDOptimizer final : public Optimizer {
public:
  explicit SGDOptimizer(MLP &mlp, DataSet &dataSet, std::unique_ptr<Loss> loss,
                        double learningRate = 0.01, double momentum = 0.0,
                        bool nesterov = false);
  bool optimize() override;

protected:
  int nextTrainingIndex() override;

private:
  int _currentInput = 0;
  double _learningRate;
  double _momentum;
  std::vector<double> _velocities;
  std::vector<int> _indices;
  bool _nesterov;
  std::random_device _randomDevice;
  std::mt19937 _randomGenerator;
};

} // namespace ml
