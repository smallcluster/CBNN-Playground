#pragma once

#include <cstdint>
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
  int size() const;
  std::vector<ComputeNode *> getNodes();
  std::vector<int> getIndices();

private:
  std::map<ComputeNode *, int> _inputIndices = {};
  std::map<int, ComputeNode *> _inputs = {};
};

class ComputeNodeVisitor;

class ComputeNode {
public:
  explicit ComputeNode(uint32_t id);
  virtual ~ComputeNode() = default;
  virtual const char *label() = 0;
  virtual double pdiff(int index) = 0;
  double eval();
  double diff();
  void invalidateCache();
  int connect(ComputeNode &other, const std::optional<int> &slot = {});
  void disconnect(ComputeNode &other);
  void clearInputs();
  void clearOutputs();
  void clearConnections();
  ComputeNode &inputAt(int index);
  ComputeNode &outputAt(int index) const;
  int nbInputs() const;
  int nbOutputs() const;

  int incOwnerCount();
  int decOwnerCount();
  int ownerCount();

  uint32_t id();

  virtual void forwardVisit(ComputeNodeVisitor &v) = 0;
  virtual void backwardVisit(ComputeNodeVisitor &v) = 0;

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
  uint32_t _id;
};

class IdentityNode final : public ComputeNode {
public:
  explicit IdentityNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class ConstantNode final : public ComputeNode {
public:
  const char *label() override;
  double pdiff(int index) override;
  void set(double value);
  void setLabel(const std::string &label);
  explicit ConstantNode(uint32_t id, double value,
                        const std::string &label = "");
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _value;
  std::string _label;
  double _eval() override;
};

class MultNode final : public ComputeNode {
public:
  explicit MultNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class CteMultNode final : public ComputeNode {
public:
  const char *label() override;
  void setCte(double cte);
  double getCte() const;
  explicit CteMultNode(uint32_t id, double cte);
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
  double _cte;
};

class DivideNode final : public ComputeNode {
public:
  explicit DivideNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class CteDivideNode final : public ComputeNode {
public:
  const char *label() override;
  void setCte(double cte);
  double getCte() const;
  explicit CteDivideNode(uint32_t id, double cte);
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
  double _cte;
};

class SubNode final : public ComputeNode {
public:
  explicit SubNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class UnarySubNode final : public ComputeNode {
public:
  explicit UnarySubNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class AddNode final : public ComputeNode {
public:
  explicit AddNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class ReLUNode final : public ComputeNode {
public:
  explicit ReLUNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class SigmoidNode final : public ComputeNode {
public:
  explicit SigmoidNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class CtePowerNode final : public ComputeNode {
public:
  const char *label() override;
  void setPower(int power);
  int getPower() const;
  explicit CtePowerNode(uint32_t id, int power);
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  int _power;
  double _eval() override;
};

class PowerNode final : public ComputeNode {
public:
  explicit PowerNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class ExpNode final : public ComputeNode {
public:
  explicit ExpNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class LnNode final : public ComputeNode {
public:
  explicit LnNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class AbsNode final : public ComputeNode {
public:
  explicit AbsNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class InvertNode final : public ComputeNode {
public:
  explicit InvertNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class AvgNode final : public ComputeNode {
public:
  explicit AvgNode(uint32_t id);
  const char *label() override;
  double pdiff(int index) override;
  void forwardVisit(ComputeNodeVisitor &v) override;
  void backwardVisit(ComputeNodeVisitor &v) override;

private:
  double _eval() override;
};

class IComputeGraph;

class NodeFactory {
public:
  NodeFactory() = delete;
  explicit NodeFactory(IComputeGraph &graph);
  IdentityNode &createIdentityNode() const;
  ConstantNode &createConstantNode(double value) const;
  MultNode &createMultNode() const;
  DivideNode &createDivideNode() const;
  SubNode &createSubNode() const;
  UnarySubNode &createUnarySubNode() const;
  AddNode &createAddNode() const;
  ReLUNode &createReLUNode() const;
  SigmoidNode &createSigmoidNode() const;
  CtePowerNode &createCtePowerNode(int power) const;
  PowerNode &createPowerNode() const;
  ExpNode &createExpNode() const;
  CteMultNode &createCteMultNode(double cte) const;
  CteDivideNode &createCteDivNode(double cte) const;
  LnNode &createLnNode() const;
  AbsNode &createAbsNode() const;
  AvgNode &createAvgNode() const;
  InvertNode &createInvertNode() const;

private:
  IComputeGraph &_graph;
};

class ComputeNodeVisitor {
public:
  virtual void visit(IdentityNode &n) = 0;
  virtual void visit(ConstantNode &n) = 0;
  virtual void visit(MultNode &n) = 0;
  virtual void visit(DivideNode &n) = 0;
  virtual void visit(UnarySubNode &n) = 0;
  virtual void visit(SubNode &n) = 0;
  virtual void visit(AddNode &n) = 0;
  virtual void visit(ReLUNode &n) = 0;
  virtual void visit(SigmoidNode &n) = 0;
  virtual void visit(CtePowerNode &n) = 0;
  virtual void visit(PowerNode &n) = 0;
  virtual void visit(ExpNode &n) = 0;
  virtual void visit(CteMultNode &n) = 0;
  virtual void visit(CteDivideNode &n) = 0;
  virtual void visit(LnNode &n) = 0;
  virtual void visit(AbsNode &n) = 0;
  virtual void visit(AvgNode &n) = 0;
  virtual void visit(InvertNode &n) = 0;
  virtual ~ComputeNodeVisitor() = default;

protected:
  ComputeNodeVisitor() = default;
};

} // namespace ml
