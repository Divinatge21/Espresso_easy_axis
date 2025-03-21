/*
 * Copyright (C) 2022 The ESPResSo project
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

#include "initialize.hpp"

#include "config.hpp"

#ifdef ELECTROSTATICS

#include "Actor_impl.hpp"

#include "CoulombMMM1D.hpp"
#include "CoulombMMM1DGpu.hpp"
#include "CoulombP3M.hpp"
#include "CoulombP3MGPU.hpp"
#include "CoulombScafacos.hpp"
#include "DebyeHueckel.hpp"
#include "ElectrostaticLayerCorrection.hpp"
#include "ICCStar.hpp"
#include "ReactionField.hpp"

#include "core/electrostatics/coulomb.hpp"
#include "core/electrostatics/registration.hpp"

#include "script_interface/auto_parameters/AutoParameter.hpp"

#endif // ELECTROSTATICS

#include <utils/Factory.hpp>

namespace ScriptInterface {
namespace Coulomb {

void initialize(Utils::Factory<ObjectHandle> *om) {
#ifdef ELECTROSTATICS
  om->register_new<DebyeHueckel>("Coulomb::DebyeHueckel");
#ifdef P3M
  om->register_new<CoulombP3M>("Coulomb::CoulombP3M");
#ifdef CUDA
  om->register_new<CoulombP3MGPU>("Coulomb::CoulombP3MGPU");
#endif
  om->register_new<ElectrostaticLayerCorrection>(
      "Coulomb::ElectrostaticLayerCorrection");
#endif // P3M
  om->register_new<ICCStar>("Coulomb::ICCStar");
#ifdef MMM1D_GPU
  om->register_new<CoulombMMM1DGpu>("Coulomb::CoulombMMM1DGpu");
#endif
  om->register_new<CoulombMMM1D>("Coulomb::CoulombMMM1D");
#ifdef SCAFACOS
  om->register_new<CoulombScafacos>("Coulomb::CoulombScafacos");
#endif
  om->register_new<ReactionField>("Coulomb::ReactionField");
#endif // ELECTROSTATICS
}

} // namespace Coulomb
} // namespace ScriptInterface
