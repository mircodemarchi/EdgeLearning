/***************************************************************************
 *            dnn_submodule.hpp
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

#ifndef EDGE_LEARNING_PYTHON_DNN_SUBMODULE_HPP
#define EDGE_LEARNING_PYTHON_DNN_SUBMODULE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "dnn/dlmath.hpp"
#include "dnn/layer.hpp"

namespace py = pybind11;

using namespace pybind11::literals;


namespace EdgeLearning {

} // namespace EdgeLearning

using namespace EdgeLearning;

DLMath::Shape2d get_shape2d(py::tuple shape2d);
DLMath::Shape3d get_shape3d(py::tuple shape3d);
LayerShape get_shape4d(py::tuple shape4d);
LayerShape get_layer_shape_from(std::vector<py::tuple> shapes);
LayerShape get_layer_shape_from(py::tuple shape);

void dnn_submodule(pybind11::module& module);

#endif // EDGE_LEARNING_PYTHON_DNN_SUBMODULE_HPP