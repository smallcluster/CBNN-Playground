#include <utility>

#include "libml/neural/dataset.h"

namespace ml {

DataTable::DataTable(const int width, const std::vector<double> &data)
    : _width(width), _data(data) {}
int DataTable::width() const { return _width; }
int DataTable::size() const { return static_cast<int>(_data.size()) / _width; }
double DataTable::get(const int line, const int column) const {
  return _data[line * _width + column];
}

DataSet::DataSet(DataTable inputTable, DataTable outputTable)
    : _inputTable(std::move(inputTable)), _outputTable(std::move(outputTable)) {
}
const DataTable &DataSet::inputTable() const { return _inputTable; }
const DataTable &DataSet::outputTable() const { return _outputTable; }
int DataSet::size() const { return _inputTable.size(); }
} // namespace ml