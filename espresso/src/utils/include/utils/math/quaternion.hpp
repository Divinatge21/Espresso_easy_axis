/*
 * Copyright (C) 2010-2022 The ESPResSo project
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
 *   Max-Planck-Institute for Polymer Research, Theory Group
 *
 * This file is part of ESPResSo.
 *
 * ESPResSo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESPResSo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef UTILS_MATH_QUATERNION_HPP
#define UTILS_MATH_QUATERNION_HPP
/** \file
 *  Quaternion algebra.
 */

#include "utils/Vector.hpp"
#include "utils/constants.hpp"
#include "utils/quaternion.hpp"

#include <cmath>
#include <limits>

namespace Utils {

/** Convert quaternion to director
 *  @return A (non-normalized) director.
 */
template <class T>
Vector<T, 3> convert_quaternion_to_director(Quaternion<T> const &quat) {
  return {2 * (quat[1] * quat[3] + quat[0] * quat[2]),
          2 * (quat[2] * quat[3] - quat[0] * quat[1]),
          quat[0] * quat[0] - quat[1] * quat[1] - quat[2] * quat[2] +
              quat[3] * quat[3]};
}

template <typename T>
Quaternion<T> conjugate(Quaternion<T> const &quat) {
    return {
        quat[0],
        -quat[1],
        -quat[2],
        -quat[3]
    };
}

// Calculate the inverse of a quaternion
template <typename T>
Quaternion<T> inverse(Quaternion<T> const &quat) {
    T norm_sq = quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3];
    return {
        quat[0] / norm_sq,
        -quat[1] / norm_sq,
        -quat[2] / norm_sq,
        -quat[3] / norm_sq
    };
}

// Calculate the difference between two quaternions
template <typename T>
Quaternion<T> quaternion_difference(Quaternion<T> const &q1, Quaternion<T> const &q2) {
    Quaternion<T> q1_inv = inverse(q1);
    return q2 * q1_inv;
}

template <typename T>
Quaternion<T> quaternion_difference_con(Quaternion<T> const &q1, Quaternion<T> const &q2) {
    Quaternion<T> q2_conj = conjugate(q2); // Можно использовать conjugate вместо inverse
    return q1 + q2_conj; // Обратите внимание, что сложение кватернионов обычно не имеет особого смысла при расчете "разницы".  Этот код может не делать то, что вы задумали.
}


/** Convert director to quaternion
 *  @param d  Director
 *  @return A (non-normalized) quaternion from the director, or {1, 0, 0, 0}
 *  if the director is the null vector.
 */
template <class T>
Quaternion<T> convert_director_to_quaternion(Vector<T, 3> const &d) {

  auto const dm = d.norm();

  // null vectors cannot be converted to quaternions
  if (dm < std::numeric_limits<T>::epsilon()) {
    return {{{{1, 0, 0, 0}}}};
  }

  // Calculate angles
  auto const d_xy = std::sqrt(d[0] * d[0] + d[1] * d[1]);
  T theta2, phi2;
  if (d_xy == 0) {
    // Here the director is co-linear with the z-axis
    // We need to distinguish between (0, 0, +d_z) and (0, 0, -d_z)
    theta2 = (d[2] > 0) ? 0 : Utils::pi<T>() / 2;
    phi2 = 0;
  } else {
    // Here we take care of all other directions
    // We suppose that theta2 = theta/2 and phi2 = (phi - pi/2)/2,
    // where angles theta and phi are in spherical coordinates
    theta2 = std::acos(d[2] / dm) / 2;
    // here we do not use the signum function due to the edge case d[1] = 0
    auto const phi = ((d[1] > 0) ? 1 : -1) * std::acos(d[0] / d_xy);
    phi2 = phi / 2 - Utils::pi<T>() / 4;
  }

  // Calculate the quaternion from the angles
  auto const cos_theta2 = std::cos(theta2);
  auto const sin_theta2 = std::sin(theta2);
  auto const cos_phi2 = std::cos(phi2);
  auto const sin_phi2 = std::sin(phi2);
  return {{{{cos_theta2 * cos_phi2, -sin_theta2 * cos_phi2,
             -sin_theta2 * sin_phi2, cos_theta2 * sin_phi2}}}};
}

/** Convert director to quaternion
 *  @param dipol  Director dip
 *  @return A (non-normalized) quaternion from the director, or {1, 0, 0, 0}
 *  if the director is the null vector.
 */
template <class T>
Quaternion<T> convert_dip_director_to_quaternion(Vector<T, 3> const &dipol) {

  auto const dm = dipol.norm();

  // null vectors cannot be converted to quaternions
  if (dm < std::numeric_limits<T>::epsilon()) {
    return {{{{1, 0, 0, 0}}}};
  }

  // Calculate angles
  auto const d_xy = std::sqrt(dipol[0] * dipol[0] + dipol[1] * dipol[1]);
  T theta2, phi2;
  if (d_xy == 0) {
    // Here the director is co-linear with the z-axis
    // We need to distinguish between (0, 0, +d_z) and (0, 0, -d_z)
    theta2 = (dipol[2] > 0) ? 0 : Utils::pi<T>() / 2;
    phi2 = 0;
  } else {
    // Here we take care of all other directions
    // We suppose that theta2 = theta/2 and phi2 = (phi - pi/2)/2,
    // where angles theta and phi are in spherical coordinates
    theta2 = std::acos(dipol[2] / dm) / 2;
    // here we do not use the signum function due to the edge case d[1] = 0
    auto const phi = ((dipol[1] > 0) ? 1 : -1) * std::acos(dipol[0] / d_xy);
    phi2 = phi / 2 - Utils::pi<T>() / 4;
  }

  // Calculate the quaternion from the angles
  auto const cos_theta2 = std::cos(theta2);
  auto const sin_theta2 = std::sin(theta2);
  auto const cos_phi2 = std::cos(phi2);
  auto const sin_phi2 = std::sin(phi2);
  return {{{{cos_theta2 * cos_phi2, -sin_theta2 * cos_phi2,
             -sin_theta2 * sin_phi2, cos_theta2 * sin_phi2}}}};
}

 

} // namespace Utils
#endif
