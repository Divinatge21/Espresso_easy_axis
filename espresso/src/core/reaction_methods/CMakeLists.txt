#
# Copyright (C) 2021-2022 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

target_sources(
  Espresso_core
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ReactionAlgorithm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ReactionEnsemble.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ConstantpHEnsemble.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/WidomInsertion.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)

if(WITH_TESTS)
  add_subdirectory(tests)
endif(WITH_TESTS)
