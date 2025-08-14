#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "utils.h"

namespace ml {

class ComputeNode {
public:
  virtual ~ComputeNode() = default;
  double eval();
  double diff();
  void clearCache();
  virtual const char *label() = 0;
  ComputeNode();
  explicit ComputeNode(bool updateGradient);
  void connect(ComputeNode &other,
               const std::optional<unsigned int> &slot = {});
  void disconnect(ComputeNode &other);
  void clearInputs();
  void clearOutputs();
  void clearConnections();
  ComputeNode &inputAt(unsigned int index);
  ComputeNode &outputAt(unsigned int index) const;
  unsigned int nbInputs() const;
  unsigned int nbOutputs() const;
private:
  std::optional<double> _cachedEval = {};
  std::optional<ContinuousMean> _cachedGradient;
  Slots _slots;
  std::vector<ComputeNode *> _outputs;
  bool _updateGradient = false;
  virtual double _eval() = 0;
  virtual double _pdiff(const unsigned int index = 0) = 0;
};

class IdentityNode final : public ComputeNode {
public:
  const char *label() override;

private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class ConstantNode final : public ComputeNode {
public:
  const char *label() override;
  void set(const double value);
  void setLabel(const std::string &label);
  explicit ConstantNode(const double value, const bool updateGradient = false,
                        const std::string &label = "");
private:
  double _value;
  std::string _label;
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class MultNode final : public ComputeNode {
public:
  const char *label() override;
private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class CteMultNode final : public ComputeNode {
public:
  const char *label() override;
  void setCte(const double cte);
  double getCte() const;
  explicit CteMultNode(const double cte);
private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
  double _cte;
};

class DivideNode final : public ComputeNode {
public:
  const char *label() override;
private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class CteDivideNode final : public ComputeNode {
public:
  const char *label() override;
  void setCte(const double cte);
  double getCte() const;
  explicit CteDivideNode(const double cte);
private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
  double _cte;
};

class SubNode final : public ComputeNode {
public:
  const char *label() override;
private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class UnarySubNode final : public ComputeNode {
public:
  const char *label() override;
private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class AddNode final : public ComputeNode {
public:
  const char *label() override;

private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class ReLUNode final : public ComputeNode {
public:
  const char *label() override;

private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class SigmoidNode final : public ComputeNode {
public:
  const char *label() override;

private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class CtePowerNode final : public ComputeNode {

public:
  const char *label() override;
  void setPower(const int power);
  int getPower() const;
  explicit CtePowerNode(const int power);

private:

  int _power;
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class PowerNode final : public ComputeNode {
public:
  const char *label() override;

private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class ExpNode final : public ComputeNode {
public:
  const char *label() override;

private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class LnNode final : public ComputeNode {
public:
  const char *label() override;

private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class AbsNode final : public ComputeNode {
public:
  const char *label() override;

private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class InvertNode final : public ComputeNode {
public:
  const char *label() override;

private:
  double _eval() override;
  double _pdiff(const unsigned int index = 0) override;
};

class IComputeGraph;

class NodeFactory {
public:
  explicit NodeFactory(IComputeGraph& graph);
  IdentityNode &createIdentityNode() const;
  ConstantNode &createConstantNode(const double value, const bool updateGradient) const;
  MultNode &createMultNode() const;
  DivideNode &createDivideNode() const;
  SubNode &createSubNode() const;
  UnarySubNode &createUnarySubNode() const;
  AddNode &createAddNode() const;
  ReLUNode &createReLUNode() const;
  SigmoidNode &createSigmoidNode() const;
  CtePowerNode &createCtePowerNode(const int power) const;
  PowerNode &createPowerNode() const;
  ExpNode &createExpNode() const;
  CteMultNode &createCteMultNode(const double cte) const;
  CteDivideNode &createCteDivNode(const double cte) const;
  LnNode &createLnNode() const;
  AbsNode &createAbsNode() const;
private:
  NodeFactory() = delete;
  IComputeGraph& _graph;
};

} // namespace ml
