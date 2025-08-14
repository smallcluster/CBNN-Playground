#include "../../include/libml/compute/utils.h"

#include <ranges>

namespace ml {
void ContinuousMean::operator<<(const double value) {
  if (_size > 0)
    _value = (_size * _value + value) / (_size + 1);
  else
    _value = value;
  ++_size;
}
double ContinuousMean::get() const { return _value; }
int ContinuousMean::size() const { return _size; }

// SLOTS
unsigned int Slots::get(ComputeNode &node) { return _inputIndices[&node]; }
ComputeNode &Slots::get(const unsigned int index) { return *(_inputs[index]); }
void Slots::set(const unsigned int index, ComputeNode &node) {
  _inputs[index] = &node;
  _inputIndices[&node] = index;
}
void Slots::erase(ComputeNode &node) {
  const unsigned int index = _inputIndices[&node];
  _inputIndices.erase(&node);
  _inputs.erase(index);
}
void Slots::erase(const unsigned int index) {
  ComputeNode *node = _inputs[index];
  _inputs.erase(index);
  _inputIndices.erase(node);
}
unsigned int Slots::size() const {
  return static_cast<unsigned int>(_inputs.size());
}
std::vector<ComputeNode *> Slots::getNodes() {
  auto v = std::ranges::views::values(_inputs);
  return {v.begin(), v.end()};
}
std::vector<unsigned int> Slots::getIndices() {
  auto v = std::ranges::views::keys(_inputs);
  return {v.begin(), v.end()};
}
} // namespace ml