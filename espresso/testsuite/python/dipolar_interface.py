#
# Copyright (C) 2013-2022 The ESPResSo project
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

import espressomd.magnetostatics


@utx.skipIfMissingFeatures(["DIPOLES"])
class Test(ut.TestCase):
    system = espressomd.System(box_l=[10., 10., 10.])
    n_nodes = system.cell_system.get_state()["n_nodes"]

    def setUp(self):
        self.system.box_l = [10., 10., 10.]
        self.system.periodicity = [True, True, True]
        self.system.part.add(pos=(0.1, 0.1, 0.1), dip=(1.3, 2.1, -6.0))
        self.system.part.add(pos=(0.2, 0.2, 0.2), dip=(7.3, 6.1, -4.0))

    def tearDown(self):
        self.system.part.clear()
        self.system.actors.clear()

    if espressomd.has_features("DIPOLES") and n_nodes == 1:
        test_dds_cpu = tests_common.generate_test_for_actor_class(
            system, espressomd.magnetostatics.DipolarDirectSumCpu,
            dict(prefactor=3.4))

    if espressomd.has_features("DIPOLES") and n_nodes == 1:
        test_dds_cpu = tests_common.generate_test_for_actor_class(
            system, espressomd.magnetostatics.DipolarDirectSumWithReplicaCpu,
            dict(prefactor=3.4, n_replica=3))

    if espressomd.has_features(
            "DIPOLAR_DIRECT_SUM") and espressomd.gpu_available():
        test_dds_gpu = tests_common.generate_test_for_actor_class(
            system, espressomd.magnetostatics.DipolarDirectSumGpu,
            dict(prefactor=3.4))

    if espressomd.has_features(
            "DIPOLAR_BARNES_HUT") and espressomd.gpu_available():
        test_dds_gpu = tests_common.generate_test_for_actor_class(
            system, espressomd.magnetostatics.DipolarBarnesHutGpu,
            dict(prefactor=3.4, epssq=200.0, itolsq=8.0))

    if espressomd.has_features("DIPOLES") and n_nodes == 1:
        test_dds_replica = tests_common.generate_test_for_actor_class(
            system, espressomd.magnetostatics.DipolarDirectSumWithReplicaCpu,
            dict(prefactor=3.4, n_replica=2))

    if espressomd.has_features("DP3M"):
        test_dp3m_metallic = tests_common.generate_test_for_actor_class(
            system, espressomd.magnetostatics.DipolarP3M,
            dict(prefactor=2., epsilon=0., mesh_off=[0.6, 0.7, 0.8], r_cut=1.4,
                 cao=2, mesh=[8, 8, 8], alpha=12., accuracy=0.01, tune=False))
        test_dp3m_non_metallic = tests_common.generate_test_for_actor_class(
            system, espressomd.magnetostatics.DipolarP3M,
            dict(prefactor=3., epsilon=3., mesh_off=[0.6, 0.7, 0.8], r_cut=1.6,
                 cao=3, mesh=[8, 8, 8], alpha=14., accuracy=0.01, tune=False))
        test_dp3m_dlc = tests_common.generate_test_for_actor_class(
            system, espressomd.magnetostatics.DLC,
            dict(gap_size=2., maxPWerror=0.1, far_cut=1.,
                 actor=espressomd.magnetostatics.DipolarP3M(
                     cao=2, tune=False, mesh=8, prefactor=2.,
                     r_cut=1.4, alpha=12., accuracy=0.01)))

    @ut.skipIf(n_nodes == 1, "only runs for 2 or more MPI ranks")
    def test_mpi_sanity_checks(self):
        DDSR = espressomd.magnetostatics.DipolarDirectSumWithReplicaCpu
        DDSC = espressomd.magnetostatics.DipolarDirectSumCpu
        # some actors don't support parallelization
        with self.assertRaisesRegex(RuntimeError, 'MPI parallelization not supported'):
            DDSC(prefactor=1.)
        with self.assertRaisesRegex(RuntimeError, 'MPI parallelization not supported'):
            DDSR(prefactor=1., n_replica=2)

    @ut.skipIf(n_nodes != 1, "only runs for 1 MPI rank")
    def test_ddswr_mixed_particles(self):
        # check that non-magnetic particles don't influence the DDS kernels
        actor = espressomd.magnetostatics.DipolarDirectSumWithReplicaCpu(
            prefactor=1., n_replica=2)
        self.system.actors.add(actor)
        energy1 = self.system.analysis.energy()["dipolar"]
        self.system.part.add(pos=(0.4, 0.2, 0.2), dip=(0.0, 0.0, 0.0))
        energy2 = self.system.analysis.energy()["dipolar"]
        self.assertAlmostEqual(energy1, energy2, delta=1e-12)

    @ut.skipIf(n_nodes != 1, "only runs for 1 MPI rank")
    def test_dds_mixed_particles(self):
        # check that non-magnetic particles don't influence the DDS kernels
        actor = espressomd.magnetostatics.DipolarDirectSumCpu(prefactor=1.)
        self.system.actors.add(actor)
        energy1 = self.system.analysis.energy()["dipolar"]
        self.system.part.add(pos=(0.4, 0.2, 0.2), dip=(0.0, 0.0, 0.0))
        energy2 = self.system.analysis.energy()["dipolar"]
        self.assertAlmostEqual(energy1, energy2, delta=1e-12)

    @ut.skipIf(n_nodes != 1, "only runs for 1 MPI rank")
    def test_exceptions_serial(self):
        DDSR = espressomd.magnetostatics.DipolarDirectSumWithReplicaCpu
        DDSC = espressomd.magnetostatics.DipolarDirectSumCpu
        MDLC = espressomd.magnetostatics.DLC
        # check invalid prefactors
        with self.assertRaisesRegex(ValueError, "Parameter 'prefactor' must be > 0"):
            DDSC(prefactor=-1.)
        # check runtime errors and input parameters
        dds = DDSC(prefactor=1.)
        with self.assertRaisesRegex(RuntimeError, "Parameter 'actor' is missing"):
            MDLC(gap_size=2., maxPWerror=0.1)
        with self.assertRaisesRegex(ValueError, "Parameter 'actor' of type Dipoles::DipolarDirectSum isn't supported by DLC"):
            MDLC(gap_size=2., maxPWerror=0.1, actor=dds)
        with self.assertRaisesRegex(ValueError, "Parameter 'n_replica' must be >= 0"):
            DDSR(prefactor=1., n_replica=-2)
        with self.assertRaisesRegex(ValueError, "Parameter 'prefactor' must be > 0"):
            DDSR(prefactor=-2., n_replica=1)
        with self.assertRaisesRegex(RuntimeError, 'with replica does not support a periodic system with zero replica'):
            DDSR(prefactor=1., n_replica=0)
        # run sanity checks
        self.system.periodicity = [True, True, False]
        ddsr = DDSR(prefactor=1., n_replica=1)
        with self.assertRaisesRegex(Exception, r"DLC: requires periodicity \(1 1 1\)"):
            mdlc = MDLC(gap_size=1., maxPWerror=1e-5, actor=ddsr)
            self.system.actors.add(mdlc)
        self.assertEqual(len(self.system.actors), 0)
        self.system.periodicity = [True, True, True]
        self.system.change_volume_and_rescale_particles(10. + 2e-3, "y")
        with self.assertRaisesRegex(Exception, "box size in x direction is different from y direction"):
            mdlc = MDLC(gap_size=1., maxPWerror=1e-5, actor=ddsr)
            self.system.actors.add(mdlc)
        self.assertEqual(len(self.system.actors), 0)
        # check it's safe to resize the box, i.e. there are no currently
        # active sanity check in the core
        self.system.change_volume_and_rescale_particles(10., "y")
        with self.assertRaisesRegex(Exception, "box size in x direction is different from y direction"):
            ddsr = DDSR(prefactor=1., n_replica=1)
            mdlc = MDLC(gap_size=1., maxPWerror=1e-5, actor=ddsr)
            self.system.actors.add(mdlc)
            self.system.change_volume_and_rescale_particles(9., "x")
        self.system.actors.clear()
        self.system.change_volume_and_rescale_particles(10., "x")

    @utx.skipIfMissingFeatures(["DP3M"])
    def test_exceptions_parallel(self):
        DP3M = espressomd.magnetostatics.DipolarP3M
        MDLC = espressomd.magnetostatics.DLC
        dp3m_params = dict(prefactor=1., epsilon=0.1, accuracy=1e-6,
                           mesh=[49, 49, 49], cao=7, r_cut=4.5, alpha=0.9)
        with self.assertRaisesRegex(ValueError, "Parameter 'prefactor' must be > 0"):
            espressomd.magnetostatics.DipolarP3M(
                **{**dp3m_params, 'prefactor': -2.})
        with self.assertRaisesRegex(ValueError, "DipolarP3M mesh has to be an integer or integer list of length 3"):
            espressomd.magnetostatics.DipolarP3M(
                **{**dp3m_params, 'mesh': [49, 49]})
        dp3m = DP3M(**dp3m_params)
        mdlc = MDLC(gap_size=2., maxPWerror=0.1, actor=dp3m)
        with self.assertRaisesRegex(ValueError, "Parameter 'actor' of type Dipoles::DipolarLayerCorrection isn't supported by DLC"):
            MDLC(gap_size=2., maxPWerror=0.1, actor=mdlc)


if __name__ == "__main__":
    ut.main()
