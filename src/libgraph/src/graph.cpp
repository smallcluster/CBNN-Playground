#include "graph.h"

#include <ranges>

namespace graph {
    EdgeListGraph::EdgeListGraph(const bool directed) : _directed(directed) {}

    EdgeListGraph::~EdgeListGraph() {
        for (const Node* n : _node_list | std::ranges::views::keys) {
            delete n;
        }
    }

    void EdgeListGraph::addNode(std::unique_ptr<Node> node) {
        _node_list[node.release()] = {};
    }

    void EdgeListGraph::removeNode(Node* node) {
        _node_list.erase(node);
    }

    void EdgeListGraph::connect(Node* n1, Node* n2) {
        _node_list[n1].insert(n2);
        if (!_directed) {
            _node_list[n2].insert(n1);
        }
    }

    void EdgeListGraph::disconnect(Node* n1, Node* n2) {
        _node_list[n1].erase(n2);
        if (!_directed) {
            _node_list[n2].erase(n1);
        }
    }

    std::vector<const Node*> EdgeListGraph::getNeighbors(Node* n) {
        return {_node_list[n].begin(), _node_list[n].end()};
    }

    bool EdgeListGraph::isDirected() {
        return _directed;
    }

    void EdgeListGraph::makeDirected() {
        _directed = true;
    }

    void EdgeListGraph::makeUndirected() {
        if (_directed) {
            for (const auto& [n1, dst] : _node_list) {
                for (Node* n2 : dst) {
                    connect(n2, n1);
                }
            }
            _directed = false;
        }
    }
} // namespace graph
