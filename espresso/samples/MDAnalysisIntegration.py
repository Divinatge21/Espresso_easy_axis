#
# Copyright (C) 2010-2022 The ESPResSo project
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

"""
Show how to expose configuration to ``MDAnalysis`` at run time. The
functions of ``MDAnalysis`` can be used to perform some analysis or
convert the frame to other formats (CHARMM, GROMACS, ...). For more
details, see :ref:`Writing various formats using MDAnalysis`.
"""
import espressomd
import espressomd.MDA_ESP
import numpy as np
import MDAnalysis as mda

# set up a minimal sample system

system = espressomd.System(box_l=[10.0, 10.0, 10.0])
np.random.seed(seed=42)

system.time_step = 0.001
system.cell_system.skin = 0.1

for i in range(10):
    new_part = system.part.add(pos=np.random.random(3) * system.box_l,
                               v=np.random.random(3))
    if i >= 5:
        new_part.q = 1.0
        new_part.type = 1

#
# ========================================================="
# Example #1: prepare the stream and access various        "
# quantities from MDAnalysis                               "
# ========================================================="
#

eos = espressomd.MDA_ESP.Stream(system)

u = mda.Universe(eos.topology, eos.trajectory)


# let's have a look at the universe
print(u)

# Inspect atoms
print(u.atoms)

print("Positions:")
print(u.atoms.positions)
print("Velocities:")
print(u.atoms.velocities)
print("Forces:")
print(u.atoms.forces)
print("Names:")
print(u.atoms.names)
print("IDs:")
print(u.atoms.ids)
print("Types:")
print(u.atoms.types)
print("Charges:")
print(u.atoms.charges)

#
# ========================================================="
# Example #2: Write the configuration to a PDB file        "
# ========================================================="
#


u.atoms.write("system.pdb")
print("===> The initial configuration has been written to system.pdb ")


#
# ========================================================="
# Example #3: Calculate a radial distribution function     "
# ========================================================="
#
from MDAnalysis.analysis.rdf import InterRDF

charged = u.select_atoms("prop charge  > 0")
rdf = InterRDF(charged, charged, nbins=7, range=(0, 10))

# This runs so far only over the single frame we have loaded.
# Multiframe averaging must be done by hand
rdf.run()

#
# ========================================================="
# Example #4: Saving frames to a GROMACS's TRR trajectory
# ========================================================="
#

from MDAnalysis.coordinates.TRR import TRRWriter

W = TRRWriter("traj.trr", n_atoms=len(system.part))

for i in range(100):
    # integrate
    system.integrator.run(1)
    # replace last frame
    # TODO loading new frames will be automated in future versions
    u.load_new(eos.trajectory)
    # append it to the .trr trajectory
    W.write_next_timestep(u.trajectory.ts)

print("===> The trajectory has been saved in the traj.trr file")
