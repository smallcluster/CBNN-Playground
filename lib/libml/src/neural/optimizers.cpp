#include "../../include/libml/neural/optimizers.h"

#include <algorithm>
#include <random>

namespace ml {

// Optimizer
//------------------------------------------------------------------------------

Optimizer::Optimizer(MLP &mlp, DataSet &dataSet)
    : _mlp(mlp), _dataSet(dataSet) {}

void Optimizer::optimize() {
  // Forward pass
  const InputData &d = nextTrainingInput();
  for (int i = 0; i < d.size(); ++i)
    _mlp.setInput(d.get(i), i);
  _mlp.eval();

  // Backward pass
  _mlp.computeGradient();

  _optimize();
}

// BatchOptimizer
//------------------------------------------------------------------------------
BatchOptimizer::BatchOptimizer(MLP &mlp, DataSet &dataSet,
                               const double learningRate, const double momentum)
    : Optimizer(mlp, dataSet), _learningRate(learningRate), _momentum(momentum),
      _velocities(_mlp.nbWeights(), 0.0) {}

InputData &BatchOptimizer::nextTrainingInput() {
  return _dataSet.get(_currentInput);
}

void BatchOptimizer::_optimize() {
  ++_currentInput;
  if (_currentInput == _dataSet.size()) {
    _currentInput = 0;
    // Apply new weight values from the avg gradient
    for (int i = 0; i < _velocities.size(); ++i) {
      _velocities[i] = _momentum * _velocities[i] -
                       _learningRate * _mlp.getWeightGradient(i);
      _mlp.setWeight(_mlp.getWeightGradient(i) + _velocities[i], i);
    }
    // clear avg gradients
    _mlp.clearGradientHistory();
  }
}

// SGDOptimizer
//------------------------------------------------------------------------------
SGDOptimizer::SGDOptimizer(MLP &mlp, DataSet &dataSet,
                           const double learningRate, const double momentum,
                           const bool nesterov)
    : Optimizer(mlp, dataSet), _learningRate(learningRate), _momentum(momentum),
      _velocities(_mlp.nbWeights(), 0.0), _indices(_mlp.nbWeights()), _nesterov(nesterov), _randomGenerator(_randomDevice()) {
  for (int i=0; i < _indices.size(); ++i)
    _indices[i] = i;
  std::ranges::shuffle(_indices, _randomGenerator);
}

InputData &SGDOptimizer::nextTrainingInput() {
  return _dataSet.get(_indices[_currentInput]);
}

void SGDOptimizer::_optimize() {
  ++_currentInput;
  if (_currentInput == _dataSet.size()) {
    _currentInput = 0;
    // Reshuffle the inputs
    std::ranges::shuffle(_indices, _randomGenerator);
  }
  // Apply new weight values from the currently computed gradient
  for (int i = 0; i < _velocities.size(); ++i) {
    const double g = _mlp.getWeightGradient(i);
    const double v = _momentum * _velocities[i] - _learningRate * g;
    _velocities[i] = v;
    const double w = _mlp.getWeightGradient(i);
    if (_nesterov)
      _mlp.setWeight(w + v, i);
    else
      _mlp.setWeight(w + _momentum * v - _learningRate * g, i);
  }
  // clear gradients
  _mlp.clearGradientHistory();
}

} // namespace ml