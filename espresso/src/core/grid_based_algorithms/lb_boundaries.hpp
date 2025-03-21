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
/** \file
 *
 * Boundary conditions for lattice Boltzmann fluid dynamics.
 * Header file for \ref lb_boundaries.cpp.
 *
 * In the current version only simple bounce back walls are implemented. Thus
 * after the streaming step, in all wall nodes all populations are bounced
 * back from where they came from.
 *
 */

#ifndef LBBOUNDARIES_H
#define LBBOUNDARIES_H

#include "lbboundaries/LBBoundary.hpp"

#include "config.hpp"

#include <utils/Span.hpp>

#include <array>
#include <memory>
#include <vector>

namespace LBBoundaries {
using LB_Fluid = std::array<Utils::Span<double>, 19>;

extern std::vector<std::shared_ptr<LBBoundary>> lbboundaries;
#if defined(LB_BOUNDARIES) || defined(LB_BOUNDARIES_GPU)

/** Initializes the constraints in the system.
 *  This function determines the lattice sites which belong to boundaries
 *  and marks them with a corresponding flag.
 */
void lb_init_boundaries();

void add(const std::shared_ptr<LBBoundary> &);
void remove(const std::shared_ptr<LBBoundary> &);

/**
 * @brief Check the boundary velocities.
 * Sanity check if the velocity defined at LB boundaries is within the Mach
 * number limit of the scheme, i.e. u < 0.2.
 */
bool sanity_check_mach_limit();

#endif // (LB_BOUNDARIES) || (LB_BOUNDARIES_GPU)
} // namespace LBBoundaries
#endif /* LB_BOUNDARIES_H */
