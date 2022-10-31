/***************************************************************************
 *            parser_submodule.hpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 *  This file is part of Edge Learning.
 *
 *  Edge Learning is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Edge Learning is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Edge Learning.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef EDGE_LEARNING_PYTHON_PARSER_SUBMODULE_HPP
#define EDGE_LEARNING_PYTHON_PARSER_SUBMODULE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace pybind11::literals;


namespace EdgeLearning {

} // namespace EdgeLearning


void parser_submodule(pybind11::module& module);

#endif // EDGE_LEARNING_PYTHON_PARSER_SUBMODULE_HPP