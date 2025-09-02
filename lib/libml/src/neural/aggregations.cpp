#include "libml/neural/aggregations.h"

namespace ml {

Aggregate::Aggregate(IComputeGraph &graph) : ComputeSubGraph(graph) {}

SumAggregate::SumAggregate(IComputeGraph &graph)
    : Aggregate(graph), _sum(this->Aggregate::nodeFactory().createAddNode()) {}
void SumAggregate::addInput(ComputeNode &node) { createEdge(node, _sum, {}); }
ComputeNode &SumAggregate::output() { return _sum; }

AvgAggregate::AvgAggregate(IComputeGraph &graph)
    : Aggregate(graph), _avg(this->Aggregate::nodeFactory().createAvgNode()) {}
void AvgAggregate::addInput(ComputeNode &node) { createEdge(node, _avg, {}); }
ComputeNode &AvgAggregate::output() { return _avg; }

} // namespace ml