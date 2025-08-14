#include "../include/libml/losses.h"

namespace ml {

// Base
Loss::Loss(ComputeGraph &graph) : ComputeSubGraph(graph) {}

// L2 Loss
L2Loss::L2Loss(ComputeGraph &graph) : Loss(graph), _sum(this->Loss::nodeFactory().createAddNode()) {
}

void L2Loss::addInput(ComputeNode &predicted, ComputeNode &trueValue) {
  SubNode &sub = nodeFactory().createSubNode();
  createEdge(trueValue, sub, 0);
  createEdge(predicted, sub, 1);
  CtePowerNode &pow = nodeFactory().createCtePowerNode(2);
  createEdge(sub, pow);
  createEdge(pow, _sum);

}
AddNode &L2Loss::sumNode() const { return _sum; }

// MSE
MSELoss::MSELoss(ComputeGraph &graph) : L2Loss(graph), _div(L2Loss::nodeFactory().createCteDivNode(0)) {
  L2Loss::createEdge(sumNode(), _div);
}
void MSELoss::addInput(ComputeNode &predicted, ComputeNode &trueValue) {
  L2Loss::addInput(predicted, trueValue);
  _div.setCte(_div.getCte() + 1);
}

// L1Loss
L1Loss::L1Loss(ComputeGraph &graph) : Loss(graph), _sum(Loss::nodeFactory().createAddNode()) {
}

void L1Loss::addInput(ComputeNode &predicted, ComputeNode &trueValue) {
  SubNode &sub = nodeFactory().createSubNode();
  createEdge(predicted, sub);
  CtePowerNode &pow = nodeFactory().createCtePowerNode(2);
  createEdge(sub, pow, 0);
  createEdge(pow, _sum, 1);
}

} // namespace ml