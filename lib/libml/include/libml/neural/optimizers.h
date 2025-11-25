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
  double get() const;
  int size() const;

private:
  double _value = 0.0;
  int _size = 0;
};

class Optimizer : public ComputeSubGraph {
public:
  virtual bool optimize() = 0;
  virtual void setDataset(DataSet &dataSet);
  Loss &getLoss();

protected:
  explicit Optimizer(MLP &mlp, std::unique_ptr<Loss> loss);
  void _forward();
  void _backward() const;
  virtual int nextTrainingIndex() = 0;
  void setLoss(std::unique_ptr<Loss> loss);
  MLP &_mlp;
  DataSet *_dataSet = nullptr;
  std::unique_ptr<Loss> _loss;
  std::vector<ComputeNode *> _trueValues;
};

class BatchOptimizer final : public Optimizer {
public:
  double learningRate;
  double momentum;
  explicit BatchOptimizer(MLP &mlp, std::unique_ptr<Loss> loss,
                          double learningRate = 0.01, double momentum = 0.0);
  bool optimize() override;

protected:
  int nextTrainingIndex() override;

private:
  int _currentInput = 0;
  std::vector<double> _previousUpdate;
  std::vector<ContinuousMean> _avgGradient;
};

class SGDOptimizer final : public Optimizer {
public:
  double learningRate;
  double momentum;
  bool nesterov;
  explicit SGDOptimizer(MLP &mlp, std::unique_ptr<Loss> loss,
                        double learningRate = 0.01, double momentum = 0.0,
                        bool nesterov = false);
  bool optimize() override;
  void setDataset(DataSet &dataSet) override;

protected:
  int nextTrainingIndex() override;

private:
  int _currentInput = 0;

  std::vector<double> _previousUpdate;
  std::vector<int> _indices;

  std::random_device _randomDevice;
  std::mt19937 _randomGenerator;
};

} // namespace ml
