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

find_package(Sphinx 2.3.0)
if(SPHINX_FOUND)
  # configured documentation tools and intermediate build results
  set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")
  set(SPHINX_PYTHON_DIR "${CMAKE_SOURCE_DIR}/src/python/espressomd")
  # Sphinx cache with pickled ReST documents
  set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

  # HTML output directory
  set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

  # Sphinx configuration file
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/conf.py" @ONLY)

  # Copy the figure directory to the build dir
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/figures
       DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

  # Copy the _static directory to the build dir
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/_static
       DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

  configure_file(samples.py ${CMAKE_CURRENT_BINARY_DIR}/samples.py)

  # Files to be copied to the build directory
  set(FILE_LIST
      "${CMAKE_SOURCE_DIR}/doc/logo/logo_48x48.png"
      "${CMAKE_CURRENT_SOURCE_DIR}/advanced_methods.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/analysis.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/appendix.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/community.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/constraints.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/contributing.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/ek.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/electrostatics.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/index.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/installation.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/integration.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/inter_bonded.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/inter_non-bonded.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/introduction.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/io.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/lb.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/magnetostatics.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/particles.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/reaction_methods.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/running.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/system_manipulation.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/system_setup.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/under_the_hood.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/visualization.rst"
      "${CMAKE_CURRENT_SOURCE_DIR}/bibliography.rst")

  foreach(file ${FILE_LIST})
    get_filename_component(basename ${file} NAME)
    configure_file(${file} ${CMAKE_CURRENT_BINARY_DIR}/${basename} COPYONLY)
  endforeach()
  configure_file(${CMAKE_SOURCE_DIR}/doc/bibliography.bib
                 ${CMAKE_CURRENT_BINARY_DIR}/bibliography.bib COPYONLY)

  # Place files to be excluded from SPHINX documentation
  set(EXCLUDE "${SPHINX_PYTHON_DIR}/gen_code_info.py"
              "${SPHINX_PYTHON_DIR}/gen_pxiconfig*")

  add_custom_target(
    sphinx
    ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/samples.py"
    COMMAND ${SPHINX_API_DOC_EXE} -f -o ${CMAKE_CURRENT_BINARY_DIR}
            ${SPHINX_PYTHON_DIR} ${EXCLUDE}
    COMMAND
      ${SPHINX_EXECUTABLE} -q -W -b html -c "${CMAKE_CURRENT_BINARY_DIR}" -d
      "${SPHINX_CACHE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}" "${SPHINX_HTML_DIR}"
    DEPENDS espressomd pypresso
    COMMENT "Building HTML documentation with Sphinx.")
  add_custom_target(
    doctest ${SPHINX_EXECUTABLE} -b doctest -d "${SPHINX_CACHE_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}" "${SPHINX_HTML_DIR}" DEPENDS sphinx
    COMMENT "Performing doctests within the Sphinx documentation.")
endif(SPHINX_FOUND)
