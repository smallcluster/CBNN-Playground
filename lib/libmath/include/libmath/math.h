#pragma once

#include <cmath>

namespace math {

struct Vec2 {
  union {
    float x;
    float u;
    float r;
  };
  union {
    float y;
    float v;
    float g;
  };
  static const Vec2 ZEROS() { return {0, 0}; }
  static const Vec2 ONES() { return {1, 1}; }
  static const Vec2 CTE(const float cte) { return {cte, cte}; }
  inline float dot(const Vec2 &v) const { return x * v.x + y * v.y; }
  inline float dot(const float s) const {return dot(Vec2::CTE(s));}
  inline float normSq() const { return dot(*this); }
  inline float norm() const { return std::sqrt(normSq()); }
  inline Vec2 normalized() const {
    const float n = norm();
    return {x / n, y / n};
  }
  inline float angle(const Vec2 &v) {
    return std::acos(dot(v) / (norm() * v.norm()));
  }
  inline float angle(const float s) {
      const Vec2 v = Vec2::CTE(s);
    return std::acos(dot(v) / (norm() * v.norm()));
  }
  inline float cross(const Vec2 &v) {
    return x*v.y-y*v.x;
  }
  inline float cross(const float s) {
    return x*s-y*s;
  }
};

inline Vec2 operator+(const Vec2 &v1, const Vec2 &v2) {
  return {v1.x + v2.x, v1.y + v2.y};
}
inline Vec2 operator-(const Vec2 &v1, const Vec2 &v2) {
  return {v1.x - v2.x, v1.y - v2.y};
}
inline Vec2 operator*(const Vec2 &v1, const Vec2 &v2) {
  return {v1.x * v2.x, v1.y * v2.y};
}
inline Vec2 operator/(const Vec2 &v1, const Vec2 &v2) {
  return {v1.x / v2.x, v1.y / v2.y};
}

inline Vec2 operator+(const Vec2 &v, const float s) {
  return v + Vec2::CTE(s);
}
inline Vec2 operator-(const Vec2 &v, const float s) {
  return v - Vec2::CTE(s);
}
inline Vec2 operator*(const Vec2 &v, const float s) {
  return v * Vec2::CTE(s);
}
inline Vec2 operator/(const Vec2 &v, const float s) {
  return v / Vec2::CTE(s);
}

inline bool operator==(const Vec2 &v1, const Vec2 &v2) {
  return (v1.x == v2.x) && (v1.y == v2.y);
}
inline bool operator!=(const Vec2 &v1, const Vec2 &v2) {
  return (v1.x != v2.x) || (v1.y != v2.y);
}
inline bool operator<(const Vec2 &v1, const Vec2 &v2) {
  return v1.x < v2.x || (v1.x == v2.x && v1.y < v2.y);
}
inline bool operator<=(const Vec2 &v1, const Vec2 &v2) {
  return v1.x <= v2.x || (v1.x == v2.x && v1.y <= v2.y);
}
inline bool operator>(const Vec2 &v1, const Vec2 &v2) {
  return v1.x > v2.x || (v1.x == v2.x && v1.y > v2.y);
}
inline bool operator>=(const Vec2 &v1, const Vec2 &v2) {
  return v1.x >= v2.x || (v1.x == v2.x && v1.y >= v2.y);
}

inline bool operator==(const Vec2 &v, const float s) {
  return v == Vec2::CTE(s);
}
inline bool operator!=(const Vec2 &v, const float s) {
  return v != Vec2::CTE(s);
}
inline bool operator<(const Vec2 &v, const float s) {
  return v < Vec2::CTE(s);
}
inline bool operator<=(const Vec2 &v, const float s) {
  return v <= Vec2::CTE(s);
}
inline bool operator>(const Vec2 &v, const float s) {
  return v > Vec2::CTE(s);
}
inline bool operator>=(const Vec2 &v, const float s) {
  return v >= Vec2::CTE(s);
}

inline Vec2 operator+(const float s, const Vec2 &v) {
  return Vec2::CTE(s) + v;
}
inline Vec2 operator-(const float s, const Vec2 &v) {
  return Vec2::CTE(s) - v;
}
inline Vec2 operator*(const float s, const Vec2 &v) {
  return Vec2::CTE(s) * v;
}
inline Vec2 operator/(const float s, const Vec2 &v) {
  return Vec2::CTE(s) / v;
}
inline bool operator==(const float s, const Vec2 &v) {
  return Vec2::CTE(s) == v;
}
inline bool operator!=(const float s, const Vec2 &v) {
  return Vec2::CTE(s) != v;
}
inline bool operator<(const float s, const Vec2 &v) {
  return Vec2::CTE(s) < v;
}
inline bool operator<=(const float s, const Vec2 &v) {
  return Vec2::CTE(s) <= v;
}
inline bool operator>(const float s, const Vec2 &v) {
  return Vec2::CTE(s) > v;
}
inline bool operator>=(const float s, const Vec2 &v) {
  return Vec2::CTE(s) >= v;
}

} // namespace verlet
