#include "libml/neural/optimizers.h"

#include <algorithm>
#include <assert.h>
#include <chrono>

#include "effolkronium/random.hpp"

using Random = effolkronium::random_static;

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

Loss &Optimizer::getLoss() { return *_loss; }

Optimizer::Optimizer(MLP &mlp, std::unique_ptr<Loss> loss)
    : ComputeSubGraph(static_cast<IComputeGraph &>(mlp)), _mlp(mlp),
      _loss(std::move(loss)) {

  // Create Cte nodes for each true inputs
  for (int i = 0; i < mlp.nbOutputs(); ++i) {
    _trueValues.push_back(
        &this->ComputeSubGraph::nodeFactory().createConstantNode(0.0));
  }

  // Connect the mlp and the true inputs to the loss sub graph
  for (int i = 0; i < mlp.nbOutputs(); ++i) {
    _loss->addInput(mlp.getOutputNode(i), *_trueValues[i]);
  }
}

void Optimizer::setDataset(DataSet &dataSet) { _dataSet = &dataSet; }

void Optimizer::_forward() {
  assert(_dataSet != nullptr && "ERROR: no DataSet");

  const int index = nextTrainingIndex();
  // Set inputs of MLP
  for (int i = 0; i < _dataSet->inputTable().width(); ++i) {
    const double v = _dataSet->inputTable().get(index, i);
    _mlp.setInput(v, i);
  }
  // Set true values for the loss
  for (int i = 0; i < _dataSet->outputTable().width(); ++i) {
    const double v = _dataSet->outputTable().get(index, i);
    static_cast<ConstantNode *>(_trueValues[i])->set(v);
  }

  // We eval the mlp with the loss !
  _loss->loss = _loss->output().eval();
}

void Optimizer::_backward() const { _mlp.diff(); }

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
BatchOptimizer::BatchOptimizer(MLP &mlp, std::unique_ptr<Loss> loss,
                               const double learningRate, const double momentum)
    : Optimizer(mlp, std::move(loss)), learningRate(learningRate),
      momentum(momentum), _previousUpdate(_mlp.nbWeights(), 0.0),
      _avgGradient(_mlp.nbWeights()) {}

int BatchOptimizer::nextTrainingIndex() { return _currentInput; }

bool BatchOptimizer::optimize() {
  _forward();
  _backward();
  for (int i = 0; i < _mlp.nbWeights(); ++i)
    _avgGradient[i].add(_mlp.getWeightDiff(i));

  // Next input
  ++_currentInput;

  if (_currentInput == _dataSet->size()) {
    _currentInput = 0;
    // Apply new weight values from the avg gradient
    for (int i = 0; i < _mlp.nbWeights(); ++i) {
      const double newWeight = _mlp.getWeight(i) +
                               momentum * _previousUpdate[i] -
                               learningRate * _avgGradient[i].get();
      _mlp.setWeight(newWeight, i);
      _previousUpdate[i] = newWeight;
      // Clear AVG gradient
      _avgGradient[i] = {};
    }
    return false;
  }
  return true;
}

// SGDOptimizer
//------------------------------------------------------------------------------
SGDOptimizer::SGDOptimizer(MLP &mlp, std::unique_ptr<Loss> loss,
                           const double learningRate, const double momentum,
                           const bool nesterov)
    : Optimizer(mlp, std::move(loss)), learningRate(learningRate),
      momentum(momentum), _previousUpdate(_mlp.nbWeights(), 0.0),
      nesterov(nesterov), _randomGenerator(_randomDevice()) {}

void SGDOptimizer::setDataset(DataSet &dataSet) {
  _dataSet = &dataSet;
  _indices = std::vector<int>(dataSet.size());
  for (int i = 0; i < _indices.size(); ++i)
    _indices[i] = i;
  Random::shuffle(_indices);
}

int SGDOptimizer::nextTrainingIndex() { return _indices[_currentInput]; }

bool SGDOptimizer::optimize() {
  _forward();
  _backward();

  ++_currentInput;

  if (_currentInput == _dataSet->size()) {
    _currentInput = 0;
    Random::shuffle(_indices);
  }

  // Apply new weight values from the gradient
  for (int i = 0; i < _mlp.nbWeights(); ++i) {
    double newWeight = _mlp.getWeight(i);
    const double g = _mlp.getWeightDiff(i);
    if (nesterov)
      newWeight +=
          momentum * (momentum * _previousUpdate[i] - learningRate * g) -
          learningRate * g;
    else
      newWeight += momentum * _previousUpdate[i] - learningRate * g;
    _mlp.setWeight(newWeight, i);
    _previousUpdate[i] = newWeight;
  }

  return false;
}

} // namespace ml