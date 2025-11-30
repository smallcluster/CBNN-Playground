#pragma once

#include "nodes.h"

#include <cstdint>
#include <set>
#include <sstream>

namespace ml {

class GraphvizVisitor : public ComputeNodeVisitor {
public:
  bool visit(IdentityNode &n);
  bool visit(ConstantNode &n);
  bool visit(MultNode &n);
  bool visit(DivideNode &n);
  bool visit(UnarySubNode &n);
  bool visit(SubNode &n);
  bool visit(AddNode &n);
  bool visit(ReLUNode &n);
  bool visit(SigmoidNode &n);
  bool visit(CtePowerNode &n);
  bool visit(PowerNode &n);
  bool visit(ExpNode &n);
  bool visit(CteMultNode &n);
  bool visit(CteDivideNode &n);
  bool visit(LnNode &n);
  bool visit(AbsNode &n);
  bool visit(AvgNode &n);
  bool visit(InvertNode &n);
  ~GraphvizVisitor();
  GraphvizVisitor();

  void saveToFile(const std::string &path);

private:
  std::set<uint32_t> _ids;
  std::stringstream _s;
  std::stringstream _edges;
  std::stringstream _nodes;

  bool genDot(ComputeNode &n, std::optional<std::string> color);
};

} // namespace ml