#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace graph {

class Node {
public:
  std::string label;
};

class ISimpleGraph {
public:
  virtual ~ISimpleGraph() = 0;
  virtual void addNode(std::unique_ptr<Node> node) = 0;
  virtual void removeNode(Node *node) = 0;
  virtual void connect(Node *n1, Node *n2) = 0;
  virtual void disconnect(Node *n1, Node *n2) = 0;
  virtual std::vector<const Node *> getNeighbors(Node *n) = 0;
  virtual bool isDirected() = 0;
  virtual void makeDirected() = 0;
  virtual void makeUndirected() = 0;
};

class EdgeListGraph final : public ISimpleGraph {
  bool _directed;
  std::map<Node *, std::set<Node *>> _node_list;

public:
  explicit EdgeListGraph(bool directed);
  ~EdgeListGraph() override;
  void addNode(std::unique_ptr<Node> node) override;
  void removeNode(Node *node) override;
  void connect(Node *n1, Node *n2) override;
  void disconnect(Node *n1, Node *n2) override;
  std::vector<const Node *> getNeighbors(Node *n) override;
  bool isDirected() override;
  void makeDirected() override;
  void makeUndirected() override;
};

} // namespace graph
