# Copyright (C) 2016-2022 The ESPResSo project
# Copyright (C) 2014 Olaf Lenz
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
# This script generates gen_pxiconfig.cpp, which in turn generates myconfig.pxi.
#
import inspect
import sys
import os
# find featuredefs.py
moduledir = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.join(moduledir, '..', '..', 'config'))
import featuredefs

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} DEFFILE CPPFILE", file=sys.stderr)
    exit(2)

deffilename, cfilename = sys.argv[1:3]

print("Reading definitions from " + deffilename + "...")
defs = featuredefs.defs(deffilename)
print("Done.")

# generate cpp-file
print("Writing " + cfilename + "...")
cfile = open(cfilename, 'w')

cfile.write("""
#include "config.hpp"
#include <iostream>
int main() {

std::cout << "# This file was autogenerated." << std::endl
          << "# Do not modify it or your changes will be overwritten!" << std::endl;

""")

template = """
#ifdef {0}
std::cout << "DEF {0} = 1" << std::endl;
#else
std::cout << "DEF {0} = 0" << std::endl;
#endif
"""

for feature in defs.allfeatures:
    cfile.write(template.format(feature))

cfile.write("""
}
""")

cfile.close()
print("Done.")
