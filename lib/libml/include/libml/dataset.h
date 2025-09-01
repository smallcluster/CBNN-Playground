#pragma once

#include <vector>
#include <memory>

namespace ml {
  class InputData {
  public:
    explicit InputData(const std::vector<double>& data);
    double get(int index) const;
    const std::vector<double>& get();
    int size() const;
  private:
    std::vector<double> _data;
  };

  class DataSet {
  public:
    DataSet();
    ~DataSet();
    void addData(std::unique_ptr<InputData> data);
    InputData& get(int index) const;
    int size() const;
  private:
    std::vector<InputData*> _data;
  };
}