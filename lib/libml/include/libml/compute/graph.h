#pragma once

#include <memory>
#include <set>
#include <vector>

#include "libml/compute/nodes.h"

namespace ml {

struct ComputeEdge {
  ComputeNode &src;
  ComputeNode &dst;
  bool operator<(const ComputeEdge &other) const;
};

class IComputeGraph {
public:
  IComputeGraph() = default;
  virtual ~IComputeGraph() = default;
  virtual ComputeEdge
  createEdge(ComputeNode &src, ComputeNode &dst, const std::optional<int> &slot) = 0;
  virtual void removeEdge(const ComputeEdge &edge) = 0;
  virtual std::set<ComputeEdge> &getEdges() = 0;
  virtual void removeNode(ComputeNode &node) = 0;
  [[nodiscard]] virtual ComputeNode &nodeAt(int index) const = 0;
  [[nodiscard]] virtual int nbNodes() const = 0;
  virtual NodeFactory &nodeFactory() = 0;
  virtual void registerNode(std::unique_ptr<ComputeNode> node) = 0;
};

class ComputeGraph final : public IComputeGraph {
public:
  ComputeGraph();
  ~ComputeGraph() override;
  ComputeEdge createEdge(ComputeNode &src, ComputeNode &dst, const std::optional<int> &slot) override;
  void removeEdge(const ComputeEdge &edge) override;
  std::set<ComputeEdge> &getEdges() override;
  void removeNode(ComputeNode &node) override;
  [[nodiscard]] ComputeNode &nodeAt(int index) const override;
  [[nodiscard]] int nbNodes() const override;
  NodeFactory &nodeFactory() override;
  void registerNode(std::unique_ptr<ComputeNode> node) override;
  // Not copyable
  ComputeGraph &operator=(const ComputeGraph &) = delete;
  ComputeGraph(const ComputeGraph &) = delete;
private:
  std::vector<ComputeNode *> _nodes;
  std::set<ComputeEdge> _edges;
  NodeFactory _nodeFactory;
};

class ComputeSubGraph : public IComputeGraph {
public:
  explicit ComputeSubGraph(IComputeGraph &graph);
  ~ComputeSubGraph() override;
  ComputeEdge createEdge(ComputeNode &src, ComputeNode &dst, const std::optional<int> &slot) override;
  void removeEdge(const ComputeEdge &edge) override;
  std::set<ComputeEdge> &getEdges() override;
  void removeNode(ComputeNode &node) override;
  [[nodiscard]] ComputeNode &nodeAt(int index) const override;
  [[nodiscard]] int nbNodes() const override;
  NodeFactory &nodeFactory() override;
  void registerNode(std::unique_ptr<ComputeNode> node) override;
  [[nodiscard]] IComputeGraph& baseGraph() const;
private:
  IComputeGraph &_graph;
  NodeFactory _nodeFactory;
  std::vector<ComputeNode *> _nodes;
  std::set<ComputeEdge> _edges;
};

} // namespace ml
