#include "libverlet/Verlet.h"

namespace Verlet {
    DistanceConstraint::DistanceConstraint(const Node *n1, const Node *n2) {
        _src = n1;
        _dst = n2;
        const Vector2 dir = n2->position-n1->position;
        _restLength = std::sqrt(dir.x*dir.x+dir.y*dir.y);
    }

    World::~World() {
        for (IConstraint* c : _constraints)
            delete c;
        for (Node* n : _nodes)
            delete n;
    }

    void World::update() {

    }



}
