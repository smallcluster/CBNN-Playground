#pragma once

#include <vector>

namespace ml {

class DataTable {
public:
  DataTable(int width, const std::vector<double> &data);
  int width() const;
  int size() const;
  double get(int line, int column) const;

private:
  int _width;
  std::vector<double> _data;
};

class DataSet {
public:
  DataSet(DataTable inputTable, DataTable outputTable);
  const DataTable &inputTable() const;
  const DataTable &outputTable() const;
  int size() const;

private:
  DataTable _inputTable;
  DataTable _outputTable;
};
} // namespace ml