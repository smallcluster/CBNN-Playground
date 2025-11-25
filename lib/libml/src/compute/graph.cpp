#include "libml/compute/graph.h"

#include <algorithm>

namespace ml {

bool ComputeEdge::operator==(const ComputeEdge &e) const {
  return src == e.src && dst == e.dst && slot == e.slot;
}
bool ComputeEdge::operator<(const ComputeEdge &e) const {
  return src < e.src || (src == e.src && dst < e.dst) ||
         (src == e.src && dst == e.dst && slot < e.slot);
}

ComputeGraph::ComputeGraph() : _nodeFactory(*this) {}

ComputeGraph::~ComputeGraph() {
  for (const auto n : _nodes)
    delete n;
}

uint32_t ComputeGraph::newId() { return _nextId++; }

ComputeEdge ComputeGraph::createEdge(ComputeNode &src, ComputeNode &dst,
                                     const std::optional<int> &slot) {
  int newSlot = src.connect(dst, slot);
  ComputeEdge e = {&src, &dst, newSlot};
  if (std::find(_edges.begin(), _edges.end(), e) == _edges.end()) {
    _edges.push_back(e);
  }
  return e;
}

void ComputeGraph::removeEdge(ComputeEdge &edge) {
  edge.src->disconnect(*edge.dst);
  std::remove(_edges.begin(), _edges.end(), edge);
}

std::vector<ComputeEdge> &ComputeGraph::getEdges() { return _edges; }

void ComputeGraph::removeNode(ComputeNode &node) {
  node.clearConnections();
  std::erase_if(_edges, [&node](ComputeEdge &e) {
    return e.src == &node || e.dst == &node;
  });
  _nodes.erase(std::ranges::find(_nodes, &node));
  if (node.decOwnerCount() == 0)
    delete &node;
}

ComputeNode &ComputeGraph::nodeAt(const int index) const {
  return *_nodes[index];
}

int ComputeGraph::nbNodes() const { return static_cast<int>(_nodes.size()); }
NodeFactory &ComputeGraph::nodeFactory() { return _nodeFactory; }

void ComputeGraph::registerNode(std::unique_ptr<ComputeNode> node) {
  node->incOwnerCount();
  _nodes.push_back(node.release());
}

// SUB GRAPH
ComputeSubGraph::ComputeSubGraph(IComputeGraph &graph)
    : _graph(graph), _nodeFactory(*this) {}

ComputeSubGraph::~ComputeSubGraph() {
  for (const auto n : _nodes) {
    n->decOwnerCount();
    _graph.removeNode(*n);
  }
}

uint32_t ComputeSubGraph::newId() { return _graph.newId(); }

ComputeEdge ComputeSubGraph::createEdge(ComputeNode &src, ComputeNode &dst,
                                        const std::optional<int> &slot) {
  ComputeEdge e = _graph.createEdge(src, dst, slot);
  if (std::find(_edges.begin(), _edges.end(), e) == _edges.end()) {
    _edges.push_back(e);
  }
  return e;
}

void ComputeSubGraph::removeEdge(ComputeEdge &edge) {
  std::remove(_edges.begin(), _edges.end(), edge);
  _graph.removeEdge(edge);
}

std::vector<ComputeEdge> &ComputeSubGraph::getEdges() { return _edges; }

void ComputeSubGraph::removeNode(ComputeNode &node) {
  node.decOwnerCount();
  std::erase_if(_edges, [&node](ComputeEdge &e) {
    return e.src == &node || e.dst == &node;
  });
  _nodes.erase(std::ranges::find(_nodes, &node));
  _graph.removeNode(node);
}

ComputeNode &ComputeSubGraph::nodeAt(const int index) const {
  return *_nodes[index];
}
int ComputeSubGraph::nbNodes() const { return static_cast<int>(_nodes.size()); }
NodeFactory &ComputeSubGraph::nodeFactory() { return _nodeFactory; }
void ComputeSubGraph::registerNode(std::unique_ptr<ComputeNode> node) {
  node->incOwnerCount();
  _nodes.push_back(node.get());
  _graph.registerNode(std::move(node));
}
IComputeGraph &ComputeSubGraph::baseGraph() const { return _graph; }

} // namespace ml