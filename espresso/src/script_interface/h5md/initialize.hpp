/*
 * Copyright (C) 2015-2022 The ESPResSo project
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

#ifndef SCRIPT_INTERFACE_H5MD_INITIALIZE_HPP
#define SCRIPT_INTERFACE_H5MD_INITIALIZE_HPP

#include "config.hpp"
#ifdef H5MD

#include <script_interface/ObjectHandle.hpp>

#include <utils/Factory.hpp>

namespace ScriptInterface {
namespace Writer {

void initialize(Utils::Factory<ObjectHandle> *om);

} /* namespace Writer */
} /* namespace ScriptInterface */

#endif // H5MD
#endif // SCRIPT_INTERFACE_H5MD_INITIALIZE_HPP
