#include "libml/compute/graph.h"

namespace ml {

bool ComputeEdge::operator<(const ComputeEdge &other) const {
  return &src < &other.src || (&src == &other.src && &dst < &other.dst);
}

ComputeGraph::ComputeGraph() : _nodeFactory(*this) {}

ComputeGraph::~ComputeGraph() {
  for (const auto n : _nodes)
    delete n;
}

ComputeEdge ComputeGraph::createEdge(ComputeNode &src, ComputeNode &dst,
                                     const std::optional<int> &slot) {
  src.connect(dst, slot);
  const ComputeEdge e = {src, dst};
  _edges.insert(e);
  return e;
}
void ComputeGraph::removeEdge(const ComputeEdge &edge) {
  edge.src.disconnect(edge.dst);
  _edges.erase(edge);
}

std::set<ComputeEdge> &ComputeGraph::getEdges() { return _edges; }

void ComputeGraph::removeNode(ComputeNode &node) {
  node.clearConnections();
  std::set<ComputeEdge> toRemove;
  for (auto e : _edges)
    if (&e.src == &node || &e.dst == &node)
      toRemove.insert(e);
  for (auto e : toRemove)
    _edges.erase(e);
  _nodes.erase(std::ranges::find(_nodes, &node));
  
  if(node.decOwnerCount() == 0)
  delete &node;
}

ComputeNode &ComputeGraph::nodeAt(const int index) const {
  return *_nodes[index];
}

int ComputeGraph::nbNodes() const {
  return static_cast<int>(_nodes.size());
}
NodeFactory &ComputeGraph::nodeFactory() { return _nodeFactory; }

void ComputeGraph::registerNode(std::unique_ptr<ComputeNode> node) {
  node->incOwnerCount();
  _nodes.push_back(node.release());
}

// SUB GRAPH
ComputeSubGraph::ComputeSubGraph(IComputeGraph &graph)
    : _graph(graph), _nodeFactory(*this) {}

ComputeSubGraph::~ComputeSubGraph() {
  for (const auto node : _nodes)
    ComputeSubGraph::removeNode(*node);
}

ComputeEdge ComputeSubGraph::createEdge(ComputeNode &src, ComputeNode &dst,
                            const std::optional<int> &slot) {
  const ComputeEdge e = _graph.createEdge(src, dst, slot);
  _edges.insert(e);
  return e;
}
void ComputeSubGraph::removeEdge(const ComputeEdge &edge) {
  _edges.erase(edge);
  _graph.removeEdge(edge);
}
std::set<ComputeEdge> &ComputeSubGraph::getEdges() { return _edges; }

void ComputeSubGraph::removeNode(ComputeNode &node) {
  node.decOwnerCount();
  std::set<ComputeEdge> toRemove;
  for (auto e : _edges)
    if (&e.src == &node || &e.dst == &node)
      toRemove.insert(e);
  for (auto e : toRemove)
    removeEdge(e);
  if (const auto it = std::ranges::find(_nodes, &node); it != _nodes.end())
    _nodes.erase(it);
  _graph.removeNode(node);
}
ComputeNode &ComputeSubGraph::nodeAt(const int index) const {
  return *_nodes[index];
}
int ComputeSubGraph::nbNodes() const {
  return static_cast<int>(_nodes.size());
}
NodeFactory &ComputeSubGraph::nodeFactory() { return _nodeFactory; }
void ComputeSubGraph::registerNode(std::unique_ptr<ComputeNode> node) {
  node->incOwnerCount();
  _nodes.push_back(node.get());
  _graph.registerNode(std::move(node));
}
IComputeGraph &ComputeSubGraph::baseGraph() const { return _graph; }


} // namespace ml