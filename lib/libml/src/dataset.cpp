#include "../include/libml/dataset.h"

namespace ml {

InputData::InputData(const std::vector<double> &data) : _data(data) {}
double InputData::get(const int index) const { return _data[index]; }
const std::vector<double> &InputData::get() { return _data; }
int InputData::size() const { return _data.size(); }

DataSet::DataSet() {}
DataSet::~DataSet() {
  for (const auto d : _data)
    delete d;
}
void DataSet::addData(std::unique_ptr<InputData> data) {
  _data.push_back(data.get());
  data.release();
}
InputData &DataSet::get(const int index) const { return *_data[index]; }

int DataSet::size() const { return _data.size(); }

} // namespace ml