#include "libml/neural/optimizers.h"

#include <algorithm>
#include <random>

namespace ml {

// ContinuousMean
//------------------------------------------------------------------------------
void ContinuousMean::add(const double value) {
  if (_size > 0)
    _value = (_size * _value + value) / (_size + 1);
  else
    _value = value;
  ++_size;
}
double ContinuousMean::get() const { return _value; }
int ContinuousMean::size() const { return _size; }

// Optimizer
//------------------------------------------------------------------------------

Optimizer::Optimizer(MLP &mlp, DataSet &dataSet, std::unique_ptr<Loss> loss)
    : ComputeSubGraph(mlp), _mlp(mlp), _dataSet(dataSet),
      _loss(std::move(loss)) {

  // Create Cte nodes for each true inputs
  for (int i = 0; i < mlp.nbOutputs(); ++i) {
    _trueValues.push_back(
        &ComputeSubGraph::nodeFactory().createConstantNode(0.0));
  }

  // Connect the mlp and the true inputs to the loss sub graph
  for (int i = 0; i < mlp.nbOutputs(); ++i) {
    _loss->addInput(mlp.getOutputNode(i), *_trueValues[i]);
  }

}

void Optimizer::_forward() {
  const int index = nextTrainingIndex();
  // Set inputs of MLP and Loss function
  for (int i = 0; i < _dataSet.inputTable().width(); ++i) {
    const double v = _dataSet.inputTable().get(index, i);
    _mlp.setInput(v, i);
    dynamic_cast<ConstantNode*>(_trueValues[i])->set(v);
  }
  // We eval the mlp AND the loss !
  _loss->output().eval();
}

void Optimizer::_backward() const {
  _mlp.diff();
}

void Optimizer::setLoss(std::unique_ptr<Loss> loss) {
  _loss.reset();
  _loss = std::move(loss);
  // Connect the mlp and the true inputs to the loss sub graph
  for (int i = 0; i < _mlp.nbOutputs(); ++i) {
    _loss->addInput(_mlp.getOutputNode(i), *_trueValues[i]);
  }
}

// BatchOptimizer
//------------------------------------------------------------------------------
BatchOptimizer::BatchOptimizer(MLP &mlp, DataSet &dataSet,
                               std::unique_ptr<Loss> loss,
                               const double learningRate, const double momentum)
    : Optimizer(mlp, dataSet, std::move(loss)), _learningRate(learningRate),
      _momentum(momentum), _velocities(_mlp.nbWeights(), 0.0),
      _avgGradient(_mlp.nbWeights()) {}

int BatchOptimizer::nextTrainingIndex() {
  return _currentInput;
}

void BatchOptimizer::optimize() {
  _forward();
  _backward();
  for (int i = 0; i < _mlp.nbWeights(); ++i)
    _avgGradient[i].add(_mlp.getWeightDiff(i));

  // Next input
  ++_currentInput;

  if (_currentInput == _dataSet.size()) {
    _currentInput = 0;
    // Apply new weight values from the avg gradient
    for (int i = 0; i < _mlp.nbWeights(); ++i) {
      _velocities[i] =
          _momentum * _velocities[i] - _learningRate * _avgGradient[i].get();
      _mlp.setWeight(_avgGradient[i].get() + _velocities[i], i);
      // Clear AVG gradient
      _avgGradient[i] = {};
    }
  }
}

// SGDOptimizer
//------------------------------------------------------------------------------
SGDOptimizer::SGDOptimizer(MLP &mlp, DataSet &dataSet,
                           std::unique_ptr<Loss> loss,
                           const double learningRate, const double momentum,
                           const bool nesterov)
    : Optimizer(mlp, dataSet, std::move(loss)), _learningRate(learningRate),
      _momentum(momentum), _velocities(_mlp.nbWeights(), 0.0),
      _indices(_mlp.nbWeights()), _nesterov(nesterov),
      _randomGenerator(_randomDevice()) {
  for (int i = 0; i < _indices.size(); ++i)
    _indices[i] = i;
  std::ranges::shuffle(_indices, _randomGenerator);
}

int SGDOptimizer::nextTrainingIndex() {
  return _indices[_currentInput];
}

void SGDOptimizer::optimize() {
  _forward();
  _backward();

  ++_currentInput;

  if (_currentInput == _dataSet.size()) {
    _currentInput = 0;
    // Reshuffle the inputs
    std::ranges::shuffle(_indices, _randomGenerator);
  }

  // Apply new weight values from the gradient
  for (int i = 0; i < _velocities.size(); ++i) {
    const double g = _mlp.getWeightDiff(i);
    const double v = _momentum * _velocities[i] - _learningRate * g;
    _velocities[i] = v;
    const double w = _mlp.getWeightDiff(i);
    if (_nesterov)
      _mlp.setWeight(w + v, i);
    else
      _mlp.setWeight(w + _momentum * v - _learningRate * g, i);
  }
}

} // namespace ml