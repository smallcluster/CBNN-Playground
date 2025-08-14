#include "../../include/libml/compute/nodes.h"

#include "../../include/libml/compute/graph.h"

#include <cmath>
#include <ranges>

namespace ml {

// BASE
ComputeNode::ComputeNode() : ComputeNode(false) {}
ComputeNode::ComputeNode(const bool updateGradient)
    : _updateGradient(updateGradient) {}

double ComputeNode::eval() {
  if (_cachedEval.has_value())
    return _cachedEval.value();
  const double r = _eval();
  _cachedEval = r;
  return r;
}

double ComputeNode::diff() {
  if (_cachedGradient.has_value() && !_updateGradient)
    return _cachedGradient.value().get();
  double g = 0.0;
  for (int i = 0; i < _outputs.size(); ++i)
    g += _outputs[i]->diff() *
         _outputs[i]->_pdiff(_outputs[i]->_slots.get(*this));
  if (!_cachedGradient.has_value() || !_updateGradient) {
    ContinuousMean c;
    _cachedGradient = c;
  }
  _cachedGradient.value() << g;
  return _cachedGradient->get();
}

void ComputeNode::clearCache() {
  _cachedEval = {};
  _cachedGradient = {};
}

void ComputeNode::connect(ComputeNode &other,
                          const std::optional<unsigned int> &slot) {
  _outputs.push_back(&other);
  if (slot.has_value()) {
    other._slots.set(slot.value(), *this);
  } else {
    const unsigned int newSlot = other._slots.size() - 1;
    other._slots.set(newSlot, *this);
  }
}

void ComputeNode::disconnect(ComputeNode &other) {
  _outputs.erase(std::ranges::find(_outputs, &other));
  other._slots.erase(*this);
}
void ComputeNode::clearInputs() {
  for (const std::vector<ComputeNode *> tmp = _slots.getNodes();
       ComputeNode * n : tmp)
    n->disconnect(*this);
}
void ComputeNode::clearOutputs() {
  for (const std::vector<ComputeNode *> tmp; ComputeNode * n : tmp)
    disconnect(*n);
}
void ComputeNode::clearConnections() {
  clearInputs();
  clearOutputs();
}

ComputeNode &ComputeNode::inputAt(const unsigned int index) {
  return _slots.get(index);
}
ComputeNode &ComputeNode::outputAt(const unsigned int index) const {
  return *(_outputs[index]);
}
unsigned int ComputeNode::nbOutputs() const {
  return static_cast<unsigned int>(_outputs.size());
}
unsigned int ComputeNode::nbInputs() const { return _slots.size(); }

// IDENTITY
const char *IdentityNode::label() { return "Id"; }
double IdentityNode::_eval() { return inputAt(0).eval(); }
double IdentityNode::_pdiff(const unsigned int index) { return 1.0; }

// CONSTANT
ConstantNode::ConstantNode(const double value, const bool updateGradient,
                           const std::string &label)
    : ComputeNode(updateGradient) {
  _value = value;
  _label = label;
}
const char *ConstantNode::label() {
  if (_label.empty())
    return std::to_string(_value).c_str();
  return _label.c_str();
}
void ConstantNode::set(const double value) { _value = value; }
void ConstantNode::setLabel(const std::string &label) { _label = label; }
double ConstantNode::_eval() { return _value; }
double ConstantNode::_pdiff(const unsigned int index) { return 0.0; }

// MULTIPLICATION
const char *MultNode::label() { return "*"; }
double MultNode::_eval() {
  double r = inputAt(0).eval();
  for (int i = 1; i < nbInputs(); ++i)
    r *= inputAt(i).eval();
  return r;
}
double MultNode::_pdiff(const unsigned int index) {
  double r = 1.0;
  for (int i = 0; i < nbInputs(); ++i)
    if (i != index)
      r *= inputAt(i).eval();
  return r;
}

CteMultNode::CteMultNode(const double cte) : ComputeNode(false) { _cte = cte; }
const char *CteMultNode::label() {
  return ("*" + std::to_string(_cte)).c_str();
}
void CteMultNode::setCte(const double cte) { _cte = cte; }
double CteMultNode::getCte() const { return _cte; }
double CteMultNode::_eval() { return inputAt(0).eval() * _cte; }
double CteMultNode::_pdiff(const unsigned int index) { return _cte; }

// DIVISION
const char *DivideNode::label() { return "/"; }
double DivideNode::_eval() { return inputAt(0).eval() / inputAt(1).eval(); }
double DivideNode::_pdiff(const unsigned int index) {
  const double x = inputAt(1).eval();
  if (index == 0)
    return 1.0 / x;
  return -inputAt(0).eval() / (x * x);
}

CteDivideNode::CteDivideNode(const double cte) : ComputeNode(false) {
  _cte = cte;
}
const char *CteDivideNode::label() {
  return ("/" + std::to_string(_cte)).c_str();
}
void CteDivideNode::setCte(const double cte) { _cte = cte; }
double CteDivideNode::getCte() const { return _cte; }
double CteDivideNode::_eval() { return inputAt(0).eval() / _cte; }
double CteDivideNode::_pdiff(const unsigned int index) { return 1.0 / _cte; }

// SUBSTRACTION
const char *SubNode::label() { return "-"; }
double SubNode::_eval() { return inputAt(0).eval() - inputAt(1).eval(); }
double SubNode::_pdiff(const unsigned int index) {
  return index == 0 ? 1.0 : -1.0;
}

const char *UnarySubNode::label() { return "-"; }
double UnarySubNode::_eval() { return -inputAt(0).eval(); }
double UnarySubNode::_pdiff(const unsigned int index) { return -1.0; }

// ADDITION
const char *AddNode::label() { return "+"; }
double AddNode::_eval() {
  double r = inputAt(0).eval();
  for (int i = 1; i < nbInputs(); ++i)
    r += inputAt(i).eval();
  return r;
}
double AddNode::_pdiff(const unsigned int index) { return 1.0; }

// ACTIVATION FUNCTIONS
const char *ReLUNode::label() { return "ReLU"; }
double ReLUNode::_eval() { return std::max(0.0, inputAt(0).eval()); }
double ReLUNode::_pdiff(const unsigned int index) {
  return inputAt(0).eval() <= 0 ? 0.0 : 1.0;
}

const char *SigmoidNode::label() { return "Sigmoid"; }
double SigmoidNode::_eval() { return 1.0 / (1 + std::exp(-inputAt(0).eval())); }
double SigmoidNode::_pdiff(const unsigned int index) {
  const double v = eval();
  return v * (1.0 - v);
}

// POWER
const char *CtePowerNode::label() {
  return ("^" + std::to_string(_power)).c_str();
}
CtePowerNode::CtePowerNode(const int power) : ComputeNode(false) {
  _power = power;
}
int CtePowerNode::getPower() const { return _power; }
void CtePowerNode::setPower(const int power) { _power = power; }
double CtePowerNode::_eval() { return std::pow(inputAt(0).eval(), _power); }
double CtePowerNode::_pdiff(const unsigned int index) {
  return _power * std::pow(inputAt(0).eval(), _power - 1);
}

const char *PowerNode::label() { return "^"; }
double PowerNode::_eval() {
  return std::pow(inputAt(0).eval(), inputAt(1).eval());
}
double PowerNode::_pdiff(const unsigned int index) {
  if (index == 0)
    return inputAt(1).eval() *
           std::pow(inputAt(0).eval(), inputAt(1).eval() - 1);
  return std::pow(inputAt(0).eval(), inputAt(1).eval()) *
         std::log(inputAt(1).eval());
}

// EXP
const char *ExpNode::label() { return "exp"; }
double ExpNode::_eval() { return std::exp(inputAt(0).eval()); }
double ExpNode::_pdiff(const unsigned int index) {
  return std::exp(inputAt(0).eval());
}

// LN
const char *LnNode::label() { return "ln"; }
double LnNode::_eval() { return std::log(inputAt(0).eval()); }
double LnNode::_pdiff(const unsigned int index) {
  return 1.0 / inputAt(0).eval();
}

// ABS
const char *AbsNode::label() { return "abs"; }
double AbsNode::_eval() { return std::abs(inputAt(0).eval()); }
double AbsNode::_pdiff(const unsigned int index) {
  const double v = inputAt(0).eval();
  if (v == 0.0)
    return 0.0;
  return v < 0 ? -1.0 : 1.0;
}

// 1/x
const char *InvertNode::label() { return "1/x"; }
double InvertNode::_eval() { return 1.0 / inputAt(0).eval(); }
double InvertNode::_pdiff(const unsigned int index) {
  const double v = inputAt(0).eval();
  return -1.0 / (v * v);
}

// NODE FACTORY
NodeFactory::NodeFactory(IComputeGraph &graph) : _graph(graph) {}

IdentityNode &NodeFactory::createIdentityNode() const {
  auto n = std::make_unique<IdentityNode>();
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}

ConstantNode &NodeFactory::createConstantNode(const double value,
                                              const bool updateGradient) const {
  auto n = std::make_unique<ConstantNode>(value, updateGradient);
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
MultNode &NodeFactory::createMultNode() const {
  auto n = std::make_unique<MultNode>();
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
DivideNode &NodeFactory::createDivideNode() const {
  auto n = std::make_unique<DivideNode>();
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
SubNode &NodeFactory::createSubNode() const {
  auto n = std::make_unique<SubNode>();
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
UnarySubNode &NodeFactory::createUnarySubNode() const {
  auto n = std::make_unique<UnarySubNode>();
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
AddNode &NodeFactory::createAddNode() const {
  auto n = std::make_unique<AddNode>();
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
ReLUNode &NodeFactory::createReLUNode() const {
  auto n = std::make_unique<ReLUNode>();
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
SigmoidNode &NodeFactory::createSigmoidNode() const {
  auto n = std::make_unique<SigmoidNode>();
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
CtePowerNode &NodeFactory::createCtePowerNode(const int power) const {
  auto n = std::make_unique<CtePowerNode>(power);
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
PowerNode &NodeFactory::createPowerNode() const {
  auto n = std::make_unique<PowerNode>();
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
ExpNode &NodeFactory::createExpNode() const {
  auto n = std::make_unique<ExpNode>();
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
CteMultNode &NodeFactory::createCteMultNode(const double cte) const {
  auto n = std::make_unique<CteMultNode>(cte);
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
CteDivideNode &NodeFactory::createCteDivNode(const double cte) const {
  auto n = std::make_unique<CteDivideNode>(cte);
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
LnNode &NodeFactory::createLnNode() const {
  auto n = std::make_unique<LnNode>();
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
AbsNode &NodeFactory::createAbsNode() const {
  auto n = std::make_unique<AbsNode>();
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}

} // namespace ml