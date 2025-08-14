#pragma once

#include <map>
#include <vector>

namespace ml {
class ContinuousMean {
public:
  void operator<<(const double value);
  double get() const;
  int size() const;

private:
  double _value = 0.0;
  int _size = 0;
};

class ComputeNode;

class Slots {
public:
  unsigned int get(ComputeNode &node);
  ComputeNode &get(unsigned int index);
  void set(unsigned int index, ComputeNode &node);
  void erase(ComputeNode &node);
  void erase(unsigned int index);
  unsigned int size() const;
  std::vector<ComputeNode *> getNodes();
  std::vector<unsigned int> getIndices();

private:
  std::map<ComputeNode *, unsigned int> _inputIndices;
  std::map<unsigned int, ComputeNode *> _inputs;
};

} // namespace ml
