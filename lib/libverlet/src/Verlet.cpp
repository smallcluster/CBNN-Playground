#include "libverlet/verlet.h"

namespace verlet {

DistanceConstraint::DistanceConstraint(Node &n1, Node &n2)
    : _src(n1), _dst(n2) {
  _restLength = (_dst.position - _src.position).norm();
}

DistanceConstraint::DistanceConstraint(Node &n1, Node &n2, float restLength)
    : _src(n1), _dst(n2), _restLength(restLength) {}

World::~World() {
  for (IConstraint *c : _constraints)
    delete c;
  for (Node *n : _nodes)
    delete n;
}

void World::update() {}

} // namespace verlet
