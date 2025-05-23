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

import unittest as ut
import unittest_decorators as utx
import tests_common
import numpy as np

import espressomd
import espressomd.magnetostatics
import espressomd.analyze
import espressomd.cuda_init
import espressomd.galilei


@utx.skipIfMissingGPU()
@utx.skipIfMissingFeatures(["DIPOLAR_BARNES_HUT", "LENNARD_JONES"])
class BH_DDS_gpu_multCPU_test(ut.TestCase):
    system = espressomd.System(box_l=[1, 1, 1])
    np.random.seed(71)

    def vectorsTheSame(self, a, b):
        tol = 5E-2
        vec_len = np.linalg.norm(a - b)
        rel = 2 * vec_len / (np.linalg.norm(a) + np.linalg.norm(b))
        return rel <= tol

    def test(self):
        pf_bh_gpu = 2.34
        pf_dds_gpu = 3.524
        ratio_dawaanr_bh_gpu = pf_dds_gpu / pf_bh_gpu
        l = 15
        system = self.system
        system.box_l = 3 * [l]
        system.periodicity = 3 * [False]
        system.time_step = 1E-4
        system.cell_system.skin = 0.1

        part_dip = np.zeros((3))

        for n in [128, 541]:
            dipole_modulus = 1.3
            part_pos = np.random.random((n, 3)) * l
            part_dip = dipole_modulus * tests_common.random_dipoles(n)
            system.part.add(pos=part_pos, dip=part_dip,
                            v=n * [(0, 0, 0)], omega_body=n * [(0, 0, 0)])

            system.non_bonded_inter[0, 0].lennard_jones.set_params(
                epsilon=10.0, sigma=0.5, cutoff=0.55, shift="auto")
            system.thermostat.set_langevin(kT=0.0, gamma=10.0, seed=42)
            g = espressomd.galilei.GalileiTransform()
            g.kill_particle_motion(rotation=True)
            system.integrator.set_vv()

            system.non_bonded_inter[0, 0].lennard_jones.set_params(
                epsilon=0.0, sigma=0.0, cutoff=-1, shift=0.0)

            system.cell_system.skin = 0.0
            system.time_step = 0.01
            system.thermostat.turn_off()

            # gamma should be zero in order to avoid the noise term in force
            # and torque
            system.thermostat.set_langevin(kT=1.297, gamma=0.0)

            dds_gpu = espressomd.magnetostatics.DipolarDirectSumGpu(
                prefactor=pf_dds_gpu)
            system.actors.add(dds_gpu)
            # check MD cell reset has no impact
            system.change_volume_and_rescale_particles(system.box_l[0], "x")
            system.periodicity = system.periodicity
            system.cell_system.node_grid = system.cell_system.node_grid
            system.integrator.run(steps=0, recalc_forces=True)

            dawaanr_f = np.copy(system.part.all().f)
            dawaanr_t = np.copy(system.part.all().torque_lab)
            dawaanr_e = system.analysis.energy()["total"]

            del dds_gpu
            system.actors.clear()

            system.integrator.run(steps=0, recalc_forces=True)
            bh_gpu = espressomd.magnetostatics.DipolarBarnesHutGpu(
                prefactor=pf_bh_gpu, epssq=200.0, itolsq=8.0)
            system.actors.add(bh_gpu)
            # check MD cell reset has no impact
            system.change_volume_and_rescale_particles(system.box_l[0], "x")
            system.periodicity = system.periodicity
            system.cell_system.node_grid = system.cell_system.node_grid
            system.integrator.run(steps=0, recalc_forces=True)

            bhgpu_f = np.copy(system.part.all().f)
            bhgpu_t = np.copy(system.part.all().torque_lab)
            bhgpu_e = system.analysis.energy()["total"]

            # compare
            for i in range(n):
                self.assertTrue(
                    self.vectorsTheSame(
                        np.array(dawaanr_t[i]),
                        ratio_dawaanr_bh_gpu * np.array(bhgpu_t[i])),
                    msg='Torques on particle do not match. i={0} dawaanr_t={1} '
                        'ratio_dawaanr_bh_gpu*bhgpu_t={2}'.format(
                        i, np.array(dawaanr_t[i]),
                        ratio_dawaanr_bh_gpu * np.array(bhgpu_t[i])))
                self.assertTrue(
                    self.vectorsTheSame(
                        np.array(dawaanr_f[i]),
                        ratio_dawaanr_bh_gpu * np.array(bhgpu_f[i])),
                    msg='Forces on particle do not match: i={0} dawaanr_f={1} '
                        'ratio_dawaanr_bh_gpu*bhgpu_f={2}'.format(
                        i, np.array(dawaanr_f[i]),
                        ratio_dawaanr_bh_gpu * np.array(bhgpu_f[i])))
            self.assertLessEqual(
                abs(dawaanr_e - bhgpu_e * ratio_dawaanr_bh_gpu),
                abs(1E-3 * dawaanr_e),
                msg='Energies for dawaanr {0} and bh_gpu {1} do not match.'
                    .format(dawaanr_e, ratio_dawaanr_bh_gpu * bhgpu_e))

            system.integrator.run(steps=0, recalc_forces=True)

            del bh_gpu
            system.actors.clear()
            system.part.clear()


if __name__ == '__main__':
    ut.main()
