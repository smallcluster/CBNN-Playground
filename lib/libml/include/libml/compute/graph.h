#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "libml/compute/nodes.h"

namespace ml {

struct ComputeEdge {
  ComputeNode *src;
  ComputeNode *dst;
  int slot;
  bool operator==(const ComputeEdge &e) const;
  bool operator<(const ComputeEdge &e) const;
};

class IComputeGraph {
public:
  IComputeGraph() = default;
  virtual ~IComputeGraph() = default;
  virtual ComputeEdge createEdge(ComputeNode &src, ComputeNode &dst,
                                 const std::optional<int> &slot) = 0;
  virtual void removeEdge(ComputeEdge &edge) = 0;
  virtual std::vector<ComputeEdge> &getEdges() = 0;
  virtual void removeNode(ComputeNode &node) = 0;
  virtual ComputeNode &nodeAt(int index) const = 0;
  virtual int nbNodes() const = 0;
  virtual NodeFactory &nodeFactory() = 0;
  virtual void registerNode(std::unique_ptr<ComputeNode> node) = 0;
  virtual uint32_t newId() = 0;
  //  Not copyable
  IComputeGraph &operator=(const IComputeGraph &) = delete;
  IComputeGraph(const IComputeGraph &) = delete;
};

class ComputeGraph final : public IComputeGraph {
public:
  ComputeGraph();
  ~ComputeGraph() override;
  ComputeEdge createEdge(ComputeNode &src, ComputeNode &dst,
                         const std::optional<int> &slot) override;
  void removeEdge(ComputeEdge &edge) override;
  std::vector<ComputeEdge> &getEdges() override;
  void removeNode(ComputeNode &node) override;
  ComputeNode &nodeAt(int index) const override;
  int nbNodes() const override;
  NodeFactory &nodeFactory() override;
  void registerNode(std::unique_ptr<ComputeNode> node) override;
  //  Not copyable
  ComputeGraph &operator=(const ComputeGraph &) = delete;
  ComputeGraph(const ComputeGraph &) = delete;
  uint32_t newId() override;

private:
  std::vector<ComputeNode *> _nodes;
  std::vector<ComputeEdge> _edges;
  NodeFactory _nodeFactory;
  uint32_t _nextId = 0;
};

class ComputeSubGraph : public IComputeGraph {
public:
  explicit ComputeSubGraph(IComputeGraph &graph);
  ~ComputeSubGraph() override;
  ComputeEdge createEdge(ComputeNode &src, ComputeNode &dst,
                         const std::optional<int> &slot) override;
  void removeEdge(ComputeEdge &edge) override;
  std::vector<ComputeEdge> &getEdges() override;
  void removeNode(ComputeNode &node) override;
  ComputeNode &nodeAt(int index) const override;
  int nbNodes() const override;
  NodeFactory &nodeFactory() override;
  void registerNode(std::unique_ptr<ComputeNode> node) override;
  IComputeGraph &baseGraph() const;
  //  Not copyable
  ComputeSubGraph &operator=(const ComputeSubGraph &) = delete;
  ComputeSubGraph(const ComputeSubGraph &) = delete;
  uint32_t newId() override;

private:
  IComputeGraph &_graph;
  NodeFactory _nodeFactory;
  std::vector<ComputeNode *> _nodes;
  std::vector<ComputeEdge> _edges;
};

} // namespace ml
