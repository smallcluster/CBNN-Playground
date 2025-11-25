#include "libml/compute/nodes.h"
#include "libml/compute/graph.h"

#include <cmath>
#include <ranges>

namespace ml {

// SLOTS
int Slots::get(ComputeNode &node) { return _inputIndices[&node]; }

ComputeNode &Slots::get(const int index) { return *_inputs[index]; }

void Slots::set(const int index, ComputeNode &node) {
  _inputs[index] = &node;
  _inputIndices[&node] = index;
}

void Slots::erase(ComputeNode &node) {
  const int index = _inputIndices[&node];
  _inputIndices.erase(&node);
  _inputs.erase(index);
}

void Slots::erase(const int index) {
  ComputeNode *node = _inputs[index];
  _inputs.erase(index);
  _inputIndices.erase(node);
}

int Slots::size() const { return static_cast<int>(_inputs.size()); }
std::vector<ComputeNode *> Slots::getNodes() {
  auto v = std::ranges::views::values(_inputs);
  return {v.begin(), v.end()};
}
std::vector<int> Slots::getIndices() {
  auto v = std::ranges::views::keys(_inputs);
  return {v.begin(), v.end()};
}

// BASE

ComputeNode::ComputeNode(const uint32_t id) : _id(id) {}
int ComputeNode::incOwnerCount() { return ++_ownerCount; };
int ComputeNode::decOwnerCount() { return --_ownerCount; };
int ComputeNode::ownerCount() { return _ownerCount; };
uint32_t ComputeNode::id() { return _id; }

double ComputeNode::eval() {
  if (_invalidateCache)
    _clearCache();

  if (_cachedEval.has_value())
    return _cachedEval.value();

  const double r = _eval();
  _cachedEval = r;
  return r;
}

double ComputeNode::diff() {
  if (_invalidateCache)
    _clearCache();

  if (_cachedGradient.has_value())
    return _cachedGradient.value();

  // The last node of the graph structure is always one
  double g = _outputs.empty() ? 1.0 : 0.0;
  for (const auto &_output : _outputs) {
    const double v =
        _output->diff() * _output->pdiff(_output->_slots.get(*this));
    g += v;
  }

  _cachedGradient = g;
  return _cachedGradient.value();
}

void ComputeNode::invalidateCache() {
  if (!_invalidateCache) {
    _invalidateCache = true;

    // propagate request to its outputs and inputs
    for (const auto n : _outputs)
      n->invalidateCache();
    for (int i = 0; i < _slots.size(); ++i)
      _slots.get(i).invalidateCache();
  }
}

int ComputeNode::connect(ComputeNode &other, const std::optional<int> &slot) {
  _outputs.push_back(&other);
  int newSlot = slot.has_value() ? slot.value() : other._slots.size();
  other._slots.set(newSlot, *this);
  invalidateCache();
  return newSlot;
}

void ComputeNode::disconnect(ComputeNode &other) {
  invalidateCache();
  _outputs.erase(std::ranges::find(_outputs, &other));
  other._slots.erase(*this);
}

void ComputeNode::clearInputs() {
  const std::vector<ComputeNode *> tmp = _slots.getNodes();
  for (ComputeNode *n : tmp)
    n->disconnect(*this);
}

void ComputeNode::clearOutputs() {
  const std::vector<ComputeNode *> tmp{_outputs.begin(), _outputs.end()};
  for (ComputeNode *n : tmp)
    disconnect(*n);
}
void ComputeNode::clearConnections() {
  clearInputs();
  clearOutputs();
}

ComputeNode &ComputeNode::inputAt(const int index) { return _slots.get(index); }
ComputeNode &ComputeNode::outputAt(const int index) const {
  return *_outputs[index];
}
int ComputeNode::nbOutputs() const { return static_cast<int>(_outputs.size()); }
int ComputeNode::nbInputs() const { return _slots.size(); }

void ComputeNode::_clearCache() {
  _cachedEval = {};
  _cachedGradient = {};
  _invalidateCache = false;
}

// IDENTITY
IdentityNode::IdentityNode(const uint32_t id) : ComputeNode(id) {}
const char *IdentityNode::label() { return "Id"; }
double IdentityNode::_eval() { return inputAt(0).eval(); }
double IdentityNode::pdiff(const int index) { return 1.0; }
void IdentityNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void IdentityNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

// CONSTANT
ConstantNode::ConstantNode(const uint32_t id, const double value,
                           const std::string &label)
    : ComputeNode(id) {
  _value = value;
  _label = label;
}
const char *ConstantNode::label() {
  if (_label.empty())
    return std::to_string(_value).c_str();
  return _label.c_str();
}
void ConstantNode::set(const double value) {
  _value = value;
  invalidateCache();
}
void ConstantNode::setLabel(const std::string &label) { _label = label; }
double ConstantNode::_eval() { return _value; }
double ConstantNode::pdiff(const int index) { return 0.0; }
void ConstantNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void ConstantNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

// MULTIPLICATION
MultNode::MultNode(const uint32_t id) : ComputeNode(id) {}
const char *MultNode::label() { return "*"; }
double MultNode::_eval() {
  double r = inputAt(0).eval();
  for (int i = 1; i < nbInputs(); ++i)
    r *= inputAt(i).eval();
  return r;
}
double MultNode::pdiff(const int index) {
  double r = 1.0;
  for (int i = 0; i < nbInputs(); ++i)
    if (i != index)
      r *= inputAt(i).eval();
  return r;
}
void MultNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void MultNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

CteMultNode::CteMultNode(const uint32_t id, const double cte)
    : ComputeNode(id), _cte(cte) {}
const char *CteMultNode::label() {
  return ("*" + std::to_string(_cte)).c_str();
}
void CteMultNode::setCte(const double cte) {
  _cte = cte;
  invalidateCache();
}
double CteMultNode::getCte() const { return _cte; }
double CteMultNode::_eval() { return inputAt(0).eval() * _cte; }
double CteMultNode::pdiff(const int index) { return _cte; }
void CteMultNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void CteMultNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

// DIVISION
DivideNode::DivideNode(const uint32_t id) : ComputeNode(id) {}
const char *DivideNode::label() { return "/"; }
double DivideNode::_eval() { return inputAt(0).eval() / inputAt(1).eval(); }
double DivideNode::pdiff(const int index) {
  const double x = inputAt(1).eval();
  if (index == 0)
    return 1.0 / x;
  return -inputAt(0).eval() / (x * x);
}
void DivideNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void DivideNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

CteDivideNode::CteDivideNode(const uint32_t id, const double cte)
    : ComputeNode(id), _cte(cte) {}
const char *CteDivideNode::label() {
  return ("/" + std::to_string(_cte)).c_str();
}
void CteDivideNode::setCte(const double cte) {
  _cte = cte;
  invalidateCache();
}
double CteDivideNode::getCte() const { return _cte; }
double CteDivideNode::_eval() { return inputAt(0).eval() / _cte; }
double CteDivideNode::pdiff(const int index) { return 1.0 / _cte; }
void CteDivideNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void CteDivideNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

// SUBSTRACTION
SubNode::SubNode(const uint32_t id) : ComputeNode(id) {}
const char *SubNode::label() { return "-"; }
double SubNode::_eval() { return inputAt(0).eval() - inputAt(1).eval(); }
double SubNode::pdiff(const int index) { return index == 0 ? 1.0 : -1.0; }
void SubNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void SubNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

UnarySubNode::UnarySubNode(const uint32_t id) : ComputeNode(id) {}
const char *UnarySubNode::label() { return "-"; }
double UnarySubNode::_eval() { return -inputAt(0).eval(); }
double UnarySubNode::pdiff(const int index) { return -1.0; }
void UnarySubNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void UnarySubNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

// ADDITION
AddNode::AddNode(const uint32_t id) : ComputeNode(id) {}
const char *AddNode::label() { return "+"; }
double AddNode::_eval() {
  double r = 0.0;
  for (int i = 0; i < nbInputs(); ++i)
    r += inputAt(i).eval();
  return r;
}
double AddNode::pdiff(const int index) { return 1.0; }
void AddNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void AddNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

// ACTIVATION FUNCTIONS
ReLUNode::ReLUNode(const uint32_t id) : ComputeNode(id) {}
const char *ReLUNode::label() { return "ReLU"; }
double ReLUNode::_eval() { return std::max(0.0, inputAt(0).eval()); }
double ReLUNode::pdiff(const int index) {
  return inputAt(0).eval() <= 0 ? 0.0 : 1.0;
}
void ReLUNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void ReLUNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

SigmoidNode::SigmoidNode(const uint32_t id) : ComputeNode(id) {}
const char *SigmoidNode::label() { return "Sigmoid"; }
double SigmoidNode::_eval() { return 1.0 / (1 + std::exp(-inputAt(0).eval())); }
double SigmoidNode::pdiff(const int index) {
  const double v = eval();
  return v * (1.0 - v);
}
void SigmoidNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void SigmoidNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

// POWER

const char *CtePowerNode::label() {
  return ("^" + std::to_string(_power)).c_str();
}
CtePowerNode::CtePowerNode(const uint32_t id, const int power)
    : ComputeNode(id), _power(power) {}
int CtePowerNode::getPower() const { return _power; }
void CtePowerNode::setPower(const int power) {
  _power = power;
  invalidateCache();
}
double CtePowerNode::_eval() { return std::pow(inputAt(0).eval(), _power); }
double CtePowerNode::pdiff(const int index) {
  return _power * std::pow(inputAt(0).eval(), _power - 1);
}
void CtePowerNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void CtePowerNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

PowerNode::PowerNode(const uint32_t id) : ComputeNode(id) {}
const char *PowerNode::label() { return "^"; }
double PowerNode::_eval() {
  return std::pow(inputAt(0).eval(), inputAt(1).eval());
}
double PowerNode::pdiff(const int index) {
  if (index == 0)
    return inputAt(1).eval() *
           std::pow(inputAt(0).eval(), inputAt(1).eval() - 1);
  return std::pow(inputAt(0).eval(), inputAt(1).eval()) *
         std::log(inputAt(1).eval());
}
void PowerNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void PowerNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

// EXP
ExpNode::ExpNode(const uint32_t id) : ComputeNode(id) {}
const char *ExpNode::label() { return "exp"; }
double ExpNode::_eval() { return std::exp(inputAt(0).eval()); }
double ExpNode::pdiff(const int index) { return std::exp(inputAt(0).eval()); }
void ExpNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void ExpNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

// LN
LnNode::LnNode(const uint32_t id) : ComputeNode(id) {}
const char *LnNode::label() { return "ln"; }
double LnNode::_eval() { return std::log(inputAt(0).eval()); }
double LnNode::pdiff(const int index) { return 1.0 / inputAt(0).eval(); }
void LnNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void LnNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

// ABS
AbsNode::AbsNode(const uint32_t id) : ComputeNode(id) {}
const char *AbsNode::label() { return "abs"; }
double AbsNode::_eval() { return std::abs(inputAt(0).eval()); }
double AbsNode::pdiff(const int index) {
  const double v = inputAt(0).eval();
  if (v == 0.0)
    return 0.0;
  return v < 0 ? -1.0 : 1.0;
}
void AbsNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void AbsNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

// 1/x
InvertNode::InvertNode(const uint32_t id) : ComputeNode(id) {}
const char *InvertNode::label() { return "1/x"; }
double InvertNode::_eval() { return 1.0 / inputAt(0).eval(); }
double InvertNode::pdiff(const int index) {
  const double v = inputAt(0).eval();
  return -1.0 / (v * v);
}
void InvertNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void InvertNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

// AVG
AvgNode::AvgNode(const uint32_t id) : ComputeNode(id) {}
const char *AvgNode::label() { return "AVG"; }
double AvgNode::_eval() {
  double avg = 0;
  const int n = nbInputs();
  for (int i = 0; i < n; ++i)
    avg += inputAt(i).eval();
  return avg / n;
}
double AvgNode::pdiff(const int index) { return 1.0 / nbInputs(); }
void AvgNode::forwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbOutputs(); ++i)
    outputAt(i).forwardVisit(v);
}
void AvgNode::backwardVisit(ComputeNodeVisitor &v) {
  v.visit(*this);
  for (int i = 0; i < nbInputs(); ++i)
    inputAt(i).backwardVisit(v);
}

// NODE FACTORY
NodeFactory::NodeFactory(IComputeGraph &graph) : _graph(graph) {}

IdentityNode &NodeFactory::createIdentityNode() const {
  auto n = std::make_unique<IdentityNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}

ConstantNode &NodeFactory::createConstantNode(const double value) const {
  auto n = std::make_unique<ConstantNode>(_graph.newId(), value);
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
MultNode &NodeFactory::createMultNode() const {
  auto n = std::make_unique<MultNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
DivideNode &NodeFactory::createDivideNode() const {
  auto n = std::make_unique<DivideNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
SubNode &NodeFactory::createSubNode() const {
  auto n = std::make_unique<SubNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
UnarySubNode &NodeFactory::createUnarySubNode() const {
  auto n = std::make_unique<UnarySubNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
AddNode &NodeFactory::createAddNode() const {
  auto n = std::make_unique<AddNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
ReLUNode &NodeFactory::createReLUNode() const {
  auto n = std::make_unique<ReLUNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
SigmoidNode &NodeFactory::createSigmoidNode() const {
  auto n = std::make_unique<SigmoidNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
CtePowerNode &NodeFactory::createCtePowerNode(const int power) const {
  auto n = std::make_unique<CtePowerNode>(_graph.newId(), power);
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
PowerNode &NodeFactory::createPowerNode() const {
  auto n = std::make_unique<PowerNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
ExpNode &NodeFactory::createExpNode() const {
  auto n = std::make_unique<ExpNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
CteMultNode &NodeFactory::createCteMultNode(const double cte) const {
  auto n = std::make_unique<CteMultNode>(_graph.newId(), cte);
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
CteDivideNode &NodeFactory::createCteDivNode(const double cte) const {
  auto n = std::make_unique<CteDivideNode>(_graph.newId(), cte);
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
LnNode &NodeFactory::createLnNode() const {
  auto n = std::make_unique<LnNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
AbsNode &NodeFactory::createAbsNode() const {
  auto n = std::make_unique<AbsNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
AvgNode &NodeFactory::createAvgNode() const {
  auto n = std::make_unique<AvgNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}
InvertNode &NodeFactory::createInvertNode() const {
  auto n = std::make_unique<InvertNode>(_graph.newId());
  const auto ptr = n.get();
  _graph.registerNode(std::move(n));
  return *ptr;
}

} // namespace ml