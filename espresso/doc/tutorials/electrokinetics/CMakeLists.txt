#
# Copyright (C) 2016-2022 The ESPResSo project
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

configure_tutorial_target(TARGET tutorial_ek DEPENDS electrokinetics.ipynb
                          figures/schlitzpore_3d.png scripts/eof_analytical.py)

nb_export(TARGET tutorial_ek SUFFIX "" FILE "electrokinetics.ipynb" HTML_RUN
          VAR_SUBST "integration_length=600;dt=0.5")
