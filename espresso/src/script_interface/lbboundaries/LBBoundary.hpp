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
#ifndef SCRIPT_INTERFACE_LBBOUNDARIES_LBBOUNDARY_HPP
#define SCRIPT_INTERFACE_LBBOUNDARIES_LBBOUNDARY_HPP

#include "config.hpp"

#include "script_interface/ScriptInterface.hpp"
#include "script_interface/auto_parameters/AutoParameters.hpp"
#include "script_interface/shapes/Shape.hpp"

#include "core/grid_based_algorithms/lb_interface.hpp"
#include "core/grid_based_algorithms/lbboundaries/LBBoundary.hpp"

#include <memory>
#include <string>

namespace ScriptInterface {
namespace LBBoundaries {
class LBBoundary : public AutoParameters<LBBoundary> {
public:
  LBBoundary() : m_lbboundary(std::make_shared<::LBBoundaries::LBBoundary>()) {
    add_parameters(
        {{"velocity",
          [this](Variant const &value) {
            m_lbboundary->set_velocity(get_value<Utils::Vector3d>(value));
          },
          [this]() { return m_lbboundary->velocity(); }},
         {"shape",
          [this](Variant const &value) {
            m_shape = get_value<std::shared_ptr<Shapes::Shape>>(value);

            if (m_shape) {
              m_lbboundary->set_shape(m_shape->shape());
            }
          },
          [this]() { return m_shape; }}});
#ifdef EK_BOUNDARIES
    add_parameters({{"charge_density",
                     [this](Variant const &value) {
                       m_lbboundary->set_charge_density(
                           get_value<double>(value));
                     },
                     [this]() { return m_lbboundary->charge_density(); }},
                    {"net_charge",
                     [this](Variant const &value) {
                       m_lbboundary->set_net_charge(get_value<double>(value));
                     },
                     [this]() { return m_lbboundary->net_charge(); }}});
#endif
  }

  Variant do_call_method(const std::string &name, const VariantMap &) override {
    if (name == "get_force") {
      // The get force method uses mpi callbacks on lb cpu
      if (context()->is_head_node()) {
        const auto agrid = lb_lbfluid_get_agrid();
        const auto tau = lb_lbfluid_get_tau();
        const double unit_conversion = agrid / tau / tau;
        return m_lbboundary->get_force() * unit_conversion;
      }
      return none;
    }
    return none;
  }

  std::shared_ptr<::LBBoundaries::LBBoundary> lbboundary() {
    return m_lbboundary;
  }

private:
  /* The actual constraint */
  std::shared_ptr<::LBBoundaries::LBBoundary> m_lbboundary;

  /* Keep a reference to the shape */
  std::shared_ptr<Shapes::Shape> m_shape;
};

} // namespace LBBoundaries
} /* namespace ScriptInterface */
#endif
