/*
 * Copyright (C) 2010-2022 The ESPResSo project
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
 *   Max-Planck-Institute for Polymer Research, Theory Group
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
 *
 * Halo scheme for parallelization of lattice algorithms.
 * Implementation of file \ref halo.hpp.
 *
 */

#include "config.hpp"

#include "communication.hpp"
#include "grid.hpp"
#include "grid_based_algorithms/lattice.hpp"
#include "halo.hpp"

#include <utils/Vector.hpp>

#include <cstdlib>
#include <cstring>
#include <memory>

/** Predefined fieldtype for double-precision LB */
static std::shared_ptr<FieldType> fieldtype_double =
    std::make_shared<FieldType>(static_cast<int>(sizeof(double)));

/** Set halo region to a given value
 * @param[out] dest pointer to the halo buffer
 * @param value integer value to write into the halo buffer
 * @param type halo field layout description
 */
void halo_dtset(char *dest, int value, std::shared_ptr<FieldType> type) {
  auto const vblocks = type->vblocks;
  auto const vstride = type->vstride;
  auto const vskip = type->vskip;
  auto const &lens = type->lengths;
  auto const &disps = type->disps;
  auto const extent = type->extent;
  auto const block_size = static_cast<long>(vskip) * static_cast<long>(extent);

  for (int i = 0; i < vblocks; i++) {
    for (int j = 0; j < vstride; j++) {
      for (std::size_t k = 0; k < disps.size(); k++)
        memset(dest + disps[k], value, lens[k]);
    }
    dest += block_size;
  }
}

void halo_dtcopy(char *r_buffer, char *s_buffer, int count,
                 std::shared_ptr<FieldType> type);

void halo_copy_vector(char *r_buffer, char *s_buffer, int count,
                      std::shared_ptr<FieldType> type, bool vflag) {

  auto const vblocks = type->vblocks;
  auto const vstride = type->vstride;
  auto const extent = type->extent;

  auto block_size = static_cast<long>(type->vskip);
  if (vflag) {
    block_size *= static_cast<long>(type->subtype->extent);
  }

  for (int i = 0; i < count; i++, s_buffer += extent, r_buffer += extent) {
    char *dest = r_buffer, *src = s_buffer;
    for (int j = 0; j < vblocks; j++, dest += block_size, src += block_size) {
      halo_dtcopy(dest, src, vstride, type->subtype);
    }
  }
}

/** Copy lattice data with layout described by @p type.
 * @param r_buffer data destination
 * @param s_buffer data source
 * @param count    amount of data to copy
 * @param type     field layout type
 */
void halo_dtcopy(char *r_buffer, char *s_buffer, int count,
                 std::shared_ptr<FieldType> type) {

  if (type->subtype) {
    halo_copy_vector(r_buffer, s_buffer, count, type, type->vflag);
  } else {

    for (int i = 0; i < count;
         i++, s_buffer += type->extent, r_buffer += type->extent) {
      if (!type->count) {
        memmove(r_buffer, s_buffer, type->extent);
      } else {
        for (int j = 0; j < type->count; j++) {
          memmove(r_buffer + type->disps[j], s_buffer + type->disps[j],
                  type->lengths[j]);
        }
      }
    }
  }
}

void prepare_halo_communication(HaloCommunicator &hc, const Lattice &lattice,
                                MPI_Datatype datatype,
                                const Utils::Vector3i &local_node_grid) {

  const auto &grid = lattice.grid;
  const auto &period = lattice.halo_grid;

  for (int n = 0; n < hc.num; n++) {
    MPI_Type_free(&(hc.halo_info[n].datatype));
  }

  int const num = 2 * 3; /* two communications in each space direction */
  hc.num = num;
  hc.halo_info.resize(num);

  auto const extent = static_cast<long>(fieldtype_double->extent);

  auto const node_neighbors = calc_node_neighbors(comm_cart);

  int cnt = 0;
  for (int dir = 0; dir < 3; dir++) {
    for (int lr = 0; lr < 2; lr++) {

      HaloInfo &hinfo = hc.halo_info[cnt];

      int nblocks = 1;
      for (int k = dir + 1; k < 3; k++) {
        nblocks *= period[k];
      }
      int stride = 1;
      for (int k = 0; k < dir; k++) {
        stride *= period[k];
      }
      int skip = 1;
      for (int k = 0; k < dir + 1 && k < 2; k++) {
        skip *= period[k];
      }

      if (lr == 0) {
        /* send to left, recv from right */
        hinfo.s_offset = extent * static_cast<long>(stride * 1);
        hinfo.r_offset = extent * static_cast<long>(stride * (grid[dir] + 1));
      } else {
        /* send to right, recv from left */
        hinfo.s_offset = extent * static_cast<long>(stride * grid[dir]);
        hinfo.r_offset = extent * static_cast<long>(stride * 0);
      }

      hinfo.source_node = node_neighbors[2 * dir + 1 - lr];
      hinfo.dest_node = node_neighbors[2 * dir + lr];

      hinfo.fieldtype = std::make_shared<FieldType>(nblocks, stride, skip, true,
                                                    fieldtype_double);

      MPI_Type_vector(nblocks, stride, skip, datatype, &hinfo.datatype);
      MPI_Type_commit(&hinfo.datatype);

      if (!box_geo.periodic(dir) &&
          (local_geo.boundary()[2 * dir + lr] != 0 ||
           local_geo.boundary()[2 * dir + 1 - lr] != 0)) {
        if (local_node_grid[dir] == 1) {
          hinfo.type = HALO_OPEN;
        } else if (lr == 0) {
          if (local_geo.boundary()[2 * dir + lr] == 1) {
            hinfo.type = HALO_RECV;
          } else {
            hinfo.type = HALO_SEND;
          }
        } else {
          if (local_geo.boundary()[2 * dir + lr] == -1) {
            hinfo.type = HALO_RECV;
          } else {
            hinfo.type = HALO_SEND;
          }
        }
      } else {
        if (local_node_grid[dir] == 1) {
          hc.halo_info[cnt].type = HALO_LOCL;
        } else {
          hc.halo_info[cnt].type = HALO_SENDRECV;
        }
      }
      cnt++;
    }
  }
}

void release_halo_communication(HaloCommunicator &hc) {
  for (int n = 0; n < hc.num; n++) {
    MPI_Type_free(&(hc.halo_info[n].datatype));
  }
}

void halo_communication(const HaloCommunicator &hc, char *const base) {

  std::shared_ptr<FieldType> fieldtype;
  MPI_Datatype datatype;
  MPI_Request request;
  MPI_Status status;

  for (int n = 0; n < hc.num; n++) {
    int s_node, r_node;
    int comm_type = hc.halo_info[n].type;
    char *s_buffer = (char *)base + hc.halo_info[n].s_offset;
    char *r_buffer = (char *)base + hc.halo_info[n].r_offset;

    switch (comm_type) {

    case HALO_LOCL:
      fieldtype = hc.halo_info[n].fieldtype;
      halo_dtcopy(r_buffer, s_buffer, 1, fieldtype);
      break;

    case HALO_SENDRECV:
      datatype = hc.halo_info[n].datatype;
      s_node = hc.halo_info[n].source_node;
      r_node = hc.halo_info[n].dest_node;
      MPI_Sendrecv(s_buffer, 1, datatype, r_node, REQ_HALO_SPREAD, r_buffer, 1,
                   datatype, s_node, REQ_HALO_SPREAD, comm_cart, &status);
      break;

    case HALO_SEND:
      datatype = hc.halo_info[n].datatype;
      fieldtype = hc.halo_info[n].fieldtype;
      r_node = hc.halo_info[n].dest_node;
      MPI_Isend(s_buffer, 1, datatype, r_node, REQ_HALO_SPREAD, comm_cart,
                &request);
      halo_dtset(r_buffer, 0, fieldtype);
      MPI_Wait(&request, &status);
      break;

    case HALO_RECV:
      datatype = hc.halo_info[n].datatype;
      s_node = hc.halo_info[n].source_node;
      MPI_Irecv(r_buffer, 1, datatype, s_node, REQ_HALO_SPREAD, comm_cart,
                &request);
      MPI_Wait(&request, &status);
      break;

    case HALO_OPEN:
      fieldtype = hc.halo_info[n].fieldtype;
      /** \todo this does not work for the n_i - \<n_i\> */
      halo_dtset(r_buffer, 0, fieldtype);
      break;
    }
  }
}
