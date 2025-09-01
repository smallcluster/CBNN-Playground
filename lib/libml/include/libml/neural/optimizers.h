#pragma once

#include "mlp.h"
#include <random>
#include <vector>

namespace ml {



class Optimizer {
public:
  virtual ~Optimizer() = default;
  void optimize();
protected:
  explicit Optimizer(MLP &mlp, DataSet& dataSet);
  virtual InputData& nextTrainingInput() = 0;
  virtual void _optimize() = 0;
  MLP& _mlp;
  DataSet& _dataSet;
};

class BatchOptimizer final : Optimizer {
public:
  explicit BatchOptimizer(MLP &mlp, DataSet& dataSet, double learningRate = 0.01, double momentum = 0.0);
protected:
  void _optimize() override;
  InputData& nextTrainingInput() override;
private:
  int _currentInput = 0;
  double _learningRate;
  double _momentum;
  std::vector<double> _velocities;
};

class SGDOptimizer final : Optimizer {
public:
  explicit SGDOptimizer(MLP &mlp, DataSet& dataSet, double learningRate = 0.01, double momentum = 0.0, bool nesterov = false);
protected:
  InputData &nextTrainingInput() override;
  void _optimize() override;
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
