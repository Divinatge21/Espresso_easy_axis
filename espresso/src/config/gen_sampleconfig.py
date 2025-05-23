#!/usr/bin/env python3
# Copyright (C) 2013,2014,2018-2022 The ESPResSo project
# Copyright (C) 2012 Olaf Lenz
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

# This script writes the sample list of features to myconfig-sample.hpp

import fileinput
import inspect
import sys
import os
# find featuredefs.py
moduledir = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.join(moduledir, '..', 'src'))
import featuredefs

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} DEFFILE", file=sys.stderr)
    exit(2)

deffilename = sys.argv[1]

defs = featuredefs.defs(deffilename)

featuresdone = set()

for line in fileinput.input(deffilename):
    line = line.strip()

    # Handle empty and comment lines
    if not line:
        print()
        continue
    elif line.startswith('#'):
        continue
    elif line.startswith('//') or line.startswith('/*'):
        print(line)
        continue

    # Tokenify line
    feature = line.split(None, 1)[0]

    if feature in defs.features and feature not in featuresdone:
        print('//#define %s' % feature)
        featuresdone.add(feature)
