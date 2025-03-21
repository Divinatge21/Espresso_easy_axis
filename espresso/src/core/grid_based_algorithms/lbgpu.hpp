/*
 * Copyright (C) 2010-2022 The ESPResSo project
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
 *  %Lattice Boltzmann implementation on GPUs.
 *
 *  Implementation in lbgpu.cpp.
 */

#ifndef LBGPU_HPP
#define LBGPU_HPP

#include "config.hpp"

#ifdef CUDA
#include "OptionalCounter.hpp"

#include <utils/Vector.hpp>
#include <utils/index.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

/* For the D3Q19 model most functions have a separate implementation
 * where the coefficients and the velocity vectors are hardcoded
 * explicitly. This saves a lot of multiplications with 1's and 0's
 * thus making the code more efficient. */
#define LBQ 19

/** Parameters for the lattice Boltzmann system for GPU. */
struct LB_parameters_gpu {
  /** number density (LB units) */
  float rho;
  /** mu (LJ units) */
  float mu;
  /** viscosity (LJ) units */
  float viscosity;
  /** relaxation rate of shear modes */
  float gamma_shear;
  /** relaxation rate of bulk modes */
  float gamma_bulk;
  /** relaxation rate of odd modes */
  float gamma_odd;
  /** relaxation rate of even modes */
  float gamma_even;
  /** flag determining whether gamma_shear, gamma_odd, and gamma_even are
   *  calculated from gamma_shear in such a way to yield a TRT LB with minimized
   *  slip at bounce-back boundaries
   */
  bool is_TRT;

  float bulk_viscosity;

  /** lattice spacing (LJ units) */
  float agrid;

  /** time step for fluid propagation (LJ units)
   *  Note: Has to be larger than MD time step!
   */
  float tau;

  Utils::Array<unsigned int, 3> dim;

  unsigned int number_of_nodes;
#ifdef LB_BOUNDARIES_GPU
  unsigned int number_of_boundnodes;
#endif

  bool external_force_density;

  Utils::Array<float, 3> ext_force_density;

  // Thermal energy
  float kT;
};

/* this structure is almost duplicated for memory efficiency. When the stress
   tensor element are needed at every timestep, this features should be
   explicitly switched on */
struct LB_rho_v_pi_gpu {
  /** density of the node */
  float rho;
  /** velocity of the node */
  Utils::Array<float, 3> v;
  /** pressure tensor */
  Utils::Array<float, 6> pi;
};

struct LB_node_force_density_gpu {
  Utils::Array<float, 3> *force_density;
#if defined(VIRTUAL_SITES_INERTIALESS_TRACERS) || defined(EK_DEBUG)

  // We need the node forces for the velocity interpolation at the virtual
  // particles' position. However, LBM wants to reset them immediately
  // after the LBM update. This variable keeps a backup
  Utils::Array<float, 3> *force_density_buf;
#endif
};

/************************************************************/
/** \name Exported Variables */
/************************************************************/
/**@{*/

/** Switch indicating momentum exchange between particles and fluid */
extern LB_parameters_gpu lbpar_gpu;
extern std::vector<LB_rho_v_pi_gpu> host_values;
#ifdef ELECTROKINETICS
extern LB_node_force_density_gpu node_f;
extern bool ek_initialized;
#endif
extern OptionalCounter rng_counter_fluid_gpu;
extern OptionalCounter rng_counter_coupling_gpu;

/**@}*/

/************************************************************/
/** \name Exported Functions */
/************************************************************/
/**@{*/
/** Conserved quantities for the lattice Boltzmann system. */
struct LB_rho_v_gpu {

  /** density of the node */
  float rho;
  /** velocity of the node */

  Utils::Array<float, 3> v;
};
void lb_GPU_sanity_checks();

void lb_get_boundary_force_pointer(float **pointer_address);
void lb_get_para_pointer(LB_parameters_gpu **pointer_address);

/** Perform a full initialization of the lattice Boltzmann system.
 *  All derived parameters and the fluid are reset to their default values.
 */
void lb_init_gpu();

/** (Re-)initialize the derived parameters for the lattice Boltzmann system.
 *  The current state of the fluid is unchanged.
 */
void lb_reinit_parameters_gpu();

/** (Re-)initialize the fluid. */
void lb_reinit_fluid_gpu();

/** Reset the forces on the fluid nodes */
void reset_LB_force_densities_GPU(bool buffer = true);

void lb_init_GPU(const LB_parameters_gpu &lbpar_gpu);

/** Integrate the lattice-Boltzmann system for one time step. */
void lb_integrate_GPU();

void lb_get_values_GPU(LB_rho_v_pi_gpu *host_values);
void lb_print_node_GPU(unsigned single_nodeindex,
                       LB_rho_v_pi_gpu *host_print_values);
#ifdef LB_BOUNDARIES_GPU
void lb_init_boundaries_GPU(std::size_t n_lb_boundaries,
                            unsigned number_of_boundnodes,
                            int *host_boundary_node_list,
                            int *host_boundary_index_list,
                            float *lb_bounday_velocity);
#endif

void lb_set_agrid_gpu(double agrid);

template <std::size_t no_of_neighbours>
void lb_calc_particle_lattice_ia_gpu(bool couple_virtual, double friction,
                                     double time_step);

void lb_calc_fluid_mass_GPU(double *mass);
void lb_calc_fluid_momentum_GPU(double *host_mom);
void lb_get_boundary_flag_GPU(unsigned int single_nodeindex,
                              unsigned int *host_flag);
void lb_get_boundary_flags_GPU(unsigned int *host_bound_array);

void lb_set_node_velocity_GPU(unsigned single_nodeindex, float *host_velocity);
void lb_set_node_rho_GPU(unsigned single_nodeindex, float host_rho);

void reinit_parameters_GPU(LB_parameters_gpu *lbpar_gpu);
void lb_reinit_extern_nodeforce_GPU(LB_parameters_gpu *lbpar_gpu);
void lb_reinit_GPU(LB_parameters_gpu *lbpar_gpu);
void lb_gpu_get_boundary_forces(std::vector<double> &forces);
void lb_save_checkpoint_GPU(float *host_checkpoint_vd);
void lb_load_checkpoint_GPU(float const *host_checkpoint_vd);

void lb_lbfluid_set_population(const Utils::Vector3i &, float[LBQ]);
void lb_lbfluid_get_population(const Utils::Vector3i &, float[LBQ]);

template <std::size_t no_of_neighbours>
void lb_get_interpolated_velocity_gpu(double const *positions,
                                      double *velocities, int length);
void linear_velocity_interpolation(double const *positions, double *velocities,
                                   int length);
void quadratic_velocity_interpolation(double const *positions,
                                      double *velocities, int length);
Utils::Array<float, 6> stress_tensor_GPU();
uint64_t lb_fluid_get_rng_state_gpu();
void lb_fluid_set_rng_state_gpu(uint64_t counter);
uint64_t lb_coupling_get_rng_state_gpu();
void lb_coupling_set_rng_state_gpu(uint64_t counter);

/** Calculate the node index from its coordinates */
inline unsigned int calculate_node_index(LB_parameters_gpu const &lbpar_gpu,
                                         Utils::Vector3i const &coord) {
  return static_cast<unsigned>(
      Utils::get_linear_index(coord, Utils::Vector3i(lbpar_gpu.dim)));
}
/**@}*/

#endif /*  CUDA */

#endif /*  LBGPU_HPP */
