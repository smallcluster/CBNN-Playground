#include "libml/compute/visitors.h"

#include <fstream>

namespace ml {

bool GraphvizVisitor::genDot(ComputeNode &n, std::optional<std::string> color) {
  uint32_t id = n.id();
  if (_ids.contains(id))
    return true;
  _ids.insert(id);
  std::string cn =
      (color.has_value() ? " color=\"" + color.value() + "\"" : "");
  _nodes << std::to_string(id) + " [label=\"" + n.label() + "\"" + cn + "];\n";
  std::string ce =
      (color.has_value() ? " [color=\"" + color.value() + "\"]" : "");
  for (int i = 0; i < n.nbInputs(); ++i) {
    uint32_t inputId = n.inputAt(i).id();
    _edges << std::to_string(inputId) + " -> " + std::to_string(id) + ce +
                  ";\n";
  }
  return false;
}

bool GraphvizVisitor::visit(ConstantNode &n) { return genDot(n, {}); }

bool GraphvizVisitor::visit(IdentityNode &n) { return genDot(n, {"magenta"}); }
bool GraphvizVisitor::visit(ReLUNode &n) { return genDot(n, {"magenta"}); }
bool GraphvizVisitor::visit(SigmoidNode &n) { return genDot(n, {"magenta"}); }

bool GraphvizVisitor::visit(MultNode &n) { return genDot(n, {"blue"}); }
bool GraphvizVisitor::visit(CteMultNode &n) { return genDot(n, {"darkblue"}); }
bool GraphvizVisitor::visit(DivideNode &n) { return genDot(n, {"orange"}); }
bool GraphvizVisitor::visit(CteDivideNode &n) {
  return genDot(n, {"darkorange"});
}

bool GraphvizVisitor::visit(SubNode &n) { return genDot(n, {"red"}); }
bool GraphvizVisitor::visit(UnarySubNode &n) { return genDot(n, {"darkred"}); }
bool GraphvizVisitor::visit(AddNode &n) { return genDot(n, {"green"}); }

bool GraphvizVisitor::visit(ExpNode &n) { return genDot(n, {"cyan"}); }
bool GraphvizVisitor::visit(LnNode &n) { return genDot(n, {"yellow"}); }

bool GraphvizVisitor::visit(CtePowerNode &n) {
  return genDot(n, {"sandybrown"});
}
bool GraphvizVisitor::visit(PowerNode &n) { return genDot(n, {"saddlebrown"}); }

bool GraphvizVisitor::visit(AbsNode &n) { return genDot(n, {"teal"}); }
bool GraphvizVisitor::visit(AvgNode &n) { return genDot(n, {"darkgreen"}); }

bool GraphvizVisitor::visit(InvertNode &n) {
  return genDot(n, {"lightsalmon"});
}

GraphvizVisitor::~GraphvizVisitor() {}
GraphvizVisitor::GraphvizVisitor() {}

void GraphvizVisitor::saveToFile(const std::string &path) {
  _s << "digraph G {\n";
  _s << "rankdir=LR;\n";
  _s << "nodesep=0.5;\n";
  _s << "ranksep=2.0;\n";
  _s << "overlap = false;\n";
  _s << "splines=ortho;\n";
  _s << _nodes.str();
  _s << "\n";
  _s << _edges.str();
  _s << "}";

  std::ofstream outFile;
  outFile.open(path, std::ios_base::out);
  outFile << _s.str();
  outFile.close();
}

} // namespace ml