#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ml {

class ComputeNode;

class Slots {
public:
  Slots() = default;
  int get(ComputeNode &node);
  ComputeNode &get(int index);
  void set(int index, ComputeNode &node);
  void erase(ComputeNode &node);
  void erase(int index);
  [[nodiscard]] int size() const;
  std::vector<ComputeNode *> getNodes();
  std::vector<int> getIndices();
private:
  std::map<ComputeNode *, int> _inputIndices = {};
  std::map<int, ComputeNode *> _inputs = {};
};

class ComputeNode {
public:
  virtual ~ComputeNode() = default;
  virtual const char *label() = 0;
  virtual double pdiff(int index) = 0;
  double eval();
  double diff();
  void invalidateCache();
  void connect(ComputeNode &other, const std::optional<int> &slot = {});
  void disconnect(ComputeNode &other);
  void clearInputs();
  void clearOutputs();
  void clearConnections();
  ComputeNode &inputAt(int index);
  [[nodiscard]] ComputeNode &outputAt(int index) const;
  [[nodiscard]] int nbInputs() const;
  [[nodiscard]] int nbOutputs() const;

  int incOwnerCount();
  int decOwnerCount();
  int ownerCount();

protected:
  ComputeNode() = default;

private:
  std::optional<double> _cachedEval = {};
  std::optional<double> _cachedGradient = {};
  Slots _slots;
  std::vector<ComputeNode *> _outputs;
  virtual double _eval() = 0;
  bool _invalidateCache = false;
  void _clearCache();
  int _ownerCount = 0;
};

class IdentityNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class ConstantNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;
  void set(double value);
  void setLabel(const std::string &label);
  explicit ConstantNode(double value, const std::string &label = "");

private:
  double _value;
  std::string _label;
  double _eval() override;
};

class MultNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class CteMultNode final : public ComputeNode {
public:
  const char *label() override;
  void setCte(double cte);
  [[nodiscard]] double getCte() const;
  explicit CteMultNode(double cte);
  double pdiff(int index) override;

private:
  double _eval() override;
  double _cte;
};

class DivideNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class CteDivideNode final : public ComputeNode {
public:
  const char *label() override;
  void setCte(double cte);
  [[nodiscard]] double getCte() const;
  explicit CteDivideNode(double cte);
  double pdiff(int index) override;

private:
  double _eval() override;
  double _cte;
};

class SubNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class UnarySubNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class AddNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class ReLUNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class SigmoidNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class CtePowerNode final : public ComputeNode {
public:
  const char *label() override;
  void setPower(int power);
  [[nodiscard]] int getPower() const;
  explicit CtePowerNode(int power);
  double pdiff(int index) override;

private:
  int _power;
  double _eval() override;
};

class PowerNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class ExpNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class LnNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class AbsNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class InvertNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class AvgNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;

private:
  double _eval() override;
};

class IComputeGraph;

class NodeFactory {
public:
  NodeFactory() = delete;
  explicit NodeFactory(IComputeGraph &graph);
  [[nodiscard]] IdentityNode &createIdentityNode() const;
  [[nodiscard]] ConstantNode &createConstantNode(double value) const;
  [[nodiscard]] MultNode &createMultNode() const;
  [[nodiscard]] DivideNode &createDivideNode() const;
  [[nodiscard]] SubNode &createSubNode() const;
  [[nodiscard]] UnarySubNode &createUnarySubNode() const;
  [[nodiscard]] AddNode &createAddNode() const;
  [[nodiscard]] ReLUNode &createReLUNode() const;
  [[nodiscard]] SigmoidNode &createSigmoidNode() const;
  [[nodiscard]] CtePowerNode &createCtePowerNode(int power) const;
  [[nodiscard]] PowerNode &createPowerNode() const;
  [[nodiscard]] ExpNode &createExpNode() const;
  [[nodiscard]] CteMultNode &createCteMultNode(double cte) const;
  [[nodiscard]] CteDivideNode &createCteDivNode(double cte) const;
  [[nodiscard]] LnNode &createLnNode() const;
  [[nodiscard]] AbsNode &createAbsNode() const;
  [[nodiscard]] AvgNode &createAvgNode() const;

private:
  IComputeGraph &_graph;
};

} // namespace ml
