#pragma once

#include <vector>

namespace ml {

class DataTable {
public:
  DataTable(int width, const std::vector<double> &data);
  [[nodiscard]] int width() const;
  [[nodiscard]] int size() const;
  [[nodiscard]] double get(int line, int column) const;

private:
  int _width;
  std::vector<double> _data;
};

class DataSet {
public:
  DataSet(DataTable inputTable, DataTable outputTable);
  [[nodiscard]] const DataTable &inputTable() const;
  [[nodiscard]] const DataTable &outputTable() const;
  [[nodiscard]] int size() const;

private:
  DataTable _inputTable;
  DataTable _outputTable;
};
} // namespace ml