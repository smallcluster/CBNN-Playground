#pragma once

#include <vector>
#include "raylib.h"

namespace Utils::Colors {
    inline Color UniformGradient(const float value, const float min, const float max, std::vector<Color> colors) {
        const float range = max - min;
        const float t = (value - min) / range;
        const int src = static_cast<int>(t * (static_cast<float>(colors.size()) - 1));

        if (src >= colors.size() - 1)
            return colors[colors.size() - 1];

        const int dst = src + 1;
        const auto [r1, g1, b1, a1] = colors[src];
        const auto [r2, g2, b2, a2] = colors[dst];
        const float newMin = static_cast<float>(src) / static_cast<float>(colors.size() - 1);
        const float newMax = static_cast<float>(dst) / static_cast<float>(colors.size() - 1);
        const float newRange = newMax - newMin;
        const float factor = (t - newMin) / newRange;
        return {
            static_cast<unsigned char>(static_cast<float>(r1) * (1.0f - factor) + static_cast<float>(r2) * factor),
            static_cast<unsigned char>(static_cast<float>(g1) * (1.0f - factor) + static_cast<float>(g2) * factor),
            static_cast<unsigned char>(static_cast<float>(b1) * (1.0f - factor) + static_cast<float>(b2) * factor),
            static_cast<unsigned char>(static_cast<float>(a1) * (1.0f - factor) + static_cast<float>(a2) * factor)
        };
    }
}
