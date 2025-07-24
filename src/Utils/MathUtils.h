#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace Utils::Math {
    inline void softMax(std::vector<double> &values) {
        std::transform(values.begin(), values.end(), values.begin(), [](const double v) { return std::exp(v); });
        const double sum = std::accumulate(values.begin(), values.end(), 0.0);
        std::transform(values.begin(), values.end(), values.begin(), [=](const double v) { return v / sum; });
    }

    inline void clamp(std::vector<double> &values, const double min, const double max) {
        std::transform(values.begin(), values.end(), values.begin(),
                       [=](const double v) { return std::clamp(v, min, max); });
    }
}
