#
# Copyright (C) 2020-2022 The ESPResSo project
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

include(unit_test)

unit_test(NAME properties_test SRC properties.cpp DEPENDS
          Espresso::particle_observables)
unit_test(NAME algorithms_test SRC algorithms.cpp DEPENDS
          Espresso::particle_observables)
unit_test(NAME observables_test SRC observables.cpp DEPENDS
          Espresso::particle_observables)
