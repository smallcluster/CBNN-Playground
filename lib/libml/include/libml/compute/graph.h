#pragma once

#include "graph.h"

#include <memory>
#include <set>
#include <vector>

#include "nodes.h"

namespace ml {

struct ComputeEdge {
  ComputeNode &src;
  ComputeNode &dst;
  bool operator<(const ComputeEdge &other) const;
};

class IComputeGraph {
public:
  virtual ~IComputeGraph() = 0;
  virtual ComputeEdge
  createEdge(ComputeNode &src, ComputeNode &dst,
             const std::optional<unsigned int> &slot = {}) = 0;
  virtual void removeEdge(const ComputeEdge &edge) = 0;
  virtual std::set<ComputeEdge> &getEdges() = 0;
  virtual void removeNode(ComputeNode &node) = 0;
  virtual ComputeNode &nodeAt(unsigned int index) const = 0;
  virtual unsigned int nbNodes() const = 0;
  virtual NodeFactory &nodeFactory() = 0;
  virtual void registerNode(std::unique_ptr<ComputeNode> node) = 0;
};

class ComputeGraph : public IComputeGraph {
public:
  ComputeGraph();
  ~ComputeGraph() override;
  ComputeEdge createEdge(ComputeNode &src, ComputeNode &dst,
                         const std::optional<unsigned int> &slot = {}) override;
  void removeEdge(const ComputeEdge &edge) override;
  std::set<ComputeEdge> &getEdges() override;
  void removeNode(ComputeNode &node) override;
  ComputeNode &nodeAt(unsigned int index) const override;
  unsigned int nbNodes() const override;
  NodeFactory &nodeFactory() override;
  void registerNode(std::unique_ptr<ComputeNode> node) override;

private:
  std::vector<ComputeNode *> _nodes;
  std::set<ComputeEdge> _edges;
  NodeFactory _nodeFactory;
  // Not copyable
  ComputeGraph &operator=(const ComputeGraph &) = delete;
  ComputeGraph(const ComputeGraph &) = delete;
};

class ComputeSubGraph : public IComputeGraph {
public:
  explicit ComputeSubGraph(IComputeGraph &graph);
  ComputeEdge createEdge(ComputeNode &src, ComputeNode &dst,
                         const std::optional<unsigned int> &slot = {}) override;
  void removeEdge(const ComputeEdge &edge) override;
  std::set<ComputeEdge> &getEdges() override;
  void removeNode(ComputeNode &node) override;
  ComputeNode &nodeAt(unsigned int index) const override;
  unsigned nbNodes() const override;
  NodeFactory &nodeFactory() override;
  void registerNode(std::unique_ptr<ComputeNode> node) override;
  IComputeGraph& baseGraph() const;
private:
  IComputeGraph &_graph;
  NodeFactory _nodeFactory;
  std::vector<ComputeNode *> _nodes;
  std::set<ComputeEdge> _edges;
};

} // namespace ml
