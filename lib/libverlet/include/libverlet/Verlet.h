#pragma once

#include "libmath/math.h"

#include <vector>

namespace verlet {

struct Node {
  math::Vec2 position = {0, 0};
  math::Vec2 oldPosition = {0, 0};
  float radius = 16.0;
  float mass = 1.0f;
  bool pinned = false;
};

class IConstraint {
public:
  virtual void solve(double dt) = 0;
  virtual ~IConstraint() = 0;
};

class DistanceConstraint final : public IConstraint {
  Node &_src;
  Node &_dst;
  float _restLength;

public:
  DistanceConstraint(Node &n1, Node &n2);
  DistanceConstraint(Node &n1, Node &n2, float restLength);
  void solve(double dt) override;
  ~DistanceConstraint() = default;
};

class World {
  std::vector<Node *> _nodes;
  std::vector<IConstraint *> _constraints;
  double _dt = 1.0 / 60.0;
  int _substeps = 10;
  int _constraintSteps = 10;

public:
  World();
  void update();
  ~World();
};

} // namespace verlet
