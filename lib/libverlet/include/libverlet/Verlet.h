#pragma once

namespace verlet {

    struct Node {
        Vector2 position = {0,0};
        Vector2 oldPosition = {0,0};
        float radius = 16.0;
        float mass = 1.0f;
        bool pinned = false;
    };

    class IConstraint {
    public:
        virtual void solve(double dt) = 0;
    };

    class DistanceConstraint final : public IConstraint {
        Node* _src;
        Node* _dst;
        float _restLength;
    public:
        DistanceConstraint(const Node* n1, const Node* n2);
        void solve(double dt) override;
    };


    class World {
        std::vector<Node*> _nodes;
        std::vector<IConstraint*> _constraints;
        double _dt = 1.0/60.0;
        int _substeps = 10;
        int _constraintSteps = 10;
    public:
        World();
        explicit World(const Graph::ISimpleGraph& graph);
        void update();
        ~World();
    };

}

