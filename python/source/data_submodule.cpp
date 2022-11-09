/***************************************************************************
 *            data_submodule.cpp
 *
 *  Copyright  2007-20  Mirco De Marchi
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne.
 *
 *  Ariadne is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "data_submodule.hpp"

#include "data/dataset.hpp"

#include <set>

namespace EdgeLearning {

} // namespace EdgeLearning


using namespace EdgeLearning;

using PyDatasetType = double;
using PyDataset = Dataset<PyDatasetType>;

static void dataset_class(pybind11::module& subm)
{
    py::class_<PyDataset> dataset_class(
        subm, "Dataset", py::buffer_protocol());

    // Constructors
    dataset_class
        .def(py::init<>())
        .def(py::init<PyDataset::Vec, SizeType, SizeType, std::set<SizeType>>(),
             "data"_a, "feature_size"_a=1, "sequence_size"_a=1, "label_idx"_a=std::set<SizeType>())
        .def(py::init<PyDataset::Mat, SizeType, std::set<SizeType>>(),
             "data"_a, "sequence_size"_a=1, "label_idx"_a=std::set<SizeType>())
        .def(py::init<PyDataset::Cub, std::set<SizeType>>(),
             "data"_a, "label_idx"_a=std::set<SizeType>());

    // From Numpy
    dataset_class
        .def(py::init([](
                py::array_t<PyDatasetType, py::array::c_style | py::array::forcecast> b,
                std::set<SizeType> labels_idx = {}) {
            py::buffer_info info = b.request();
            std::vector<PyDatasetType> dataset(
                static_cast<std::size_t>(info.size));
            std::copy(
                static_cast<PyDatasetType*>(info.ptr),
                static_cast<PyDatasetType*>(info.ptr) + info.size,
                dataset.begin());

            if (info.ndim == 1)
            {
                return PyDataset(
                    dataset,
                    1,
                    1,
                    labels_idx);
            }

            if (info.ndim == 2)
            {
                return PyDataset(
                    dataset,
                    static_cast<SizeType>(info.shape[1]),
                    1,
                    labels_idx);
            }

            if (info.ndim == 3)
            {
                return PyDataset(
                    dataset,
                    static_cast<SizeType>(info.shape[1]),
                    static_cast<SizeType>(info.shape[2]),
                    labels_idx);
            }

            throw std::runtime_error(
                "The maximum number of dimensions "
                "accepted by the Dataset type is 3");
        }), "np_arr"_a, "label_idx"_a=std::set<SizeType>());

    // To Numpy
    dataset_class.def_buffer([](PyDataset &ds) -> py::buffer_info {
        auto ptr = static_cast<void*>(ds.data().data());
        py::ssize_t itemsize(sizeof(PyDatasetType));
        auto format = py::format_descriptor<PyDatasetType>::format();

        if (ds.feature_size() > 1 && ds.sequence_size() > 1)
        {
            py::ssize_t number_of_dimensions = 3;
            return py::buffer_info(
                ptr, itemsize, format, number_of_dimensions,
                {ds.size(), ds.feature_size(), ds.sequence_size()},
                {sizeof(double) * ds.sequence_size() * ds.feature_size(),
                 sizeof(double) * ds.sequence_size(),
                 sizeof(double)}
            );
        }

        if (ds.feature_size() > 1)
        {
            py::ssize_t number_of_dimensions = 2;
            return py::buffer_info(
                ptr, itemsize, format, number_of_dimensions,
                {ds.size(), ds.feature_size()},
                {sizeof(double) * ds.feature_size(),
                 sizeof(double)}
            );
        }

        py::ssize_t number_of_dimensions = 1;
        return py::buffer_info(
            ptr, itemsize, format, number_of_dimensions,
            {ds.size()},
            {sizeof(double)}
        );
    });

    // Methods
    dataset_class.def("feature_size", &PyDataset::feature_size);
    dataset_class.def_property(
        "sequence_size",
        [](const PyDataset& py_dt){ return py_dt.sequence_size(); },
        [](PyDataset& py_dt, const SizeType& s){ py_dt.sequence_size(s); });
    dataset_class.def("size", &PyDataset::size);
    dataset_class.def("empty", &PyDataset::empty);
    dataset_class.def("entry", &PyDataset::entry, "idx"_a);
    dataset_class.def("entry_seq", &PyDataset::entry_seq, "sequence_idx"_a);
    dataset_class.def("input_idx", &PyDataset::input_idx);
    dataset_class.def("input", [](PyDataset& py_dt, SizeType idx){ return py_dt.input(
        idx); }, "idx"_a);
    dataset_class.def("inputs", [](PyDataset& py_dt){ return py_dt.inputs(); });
    dataset_class.def("inputs_seq", &PyDataset::inputs_seq, "sequence_idx"_a);
    dataset_class.def_property(
        "label_idx",
        [](const PyDataset& py_dt){ return py_dt.label_idx(); },
        [](PyDataset& py_dt, const std::set<SizeType>& set){
            py_dt.label_idx(set); });
    dataset_class.def("label", [](PyDataset& py_dt, SizeType idx){ return py_dt.label(
        idx); }, "idx"_a);
    dataset_class.def("labels", [](PyDataset& py_dt){ return py_dt.labels(); });
    dataset_class.def("inputs_seq", &PyDataset::inputs_seq, "sequence_idx"_a);
    dataset_class.def(
        "subdata",
        [](PyDataset& py_dt, SizeType from, SizeType to = 0) {
            if (from > to) to = py_dt.size();
            return py_dt.subdata(from, to);
        }, "from"_a, "to"_a);
    dataset_class.def(
        "subdata",
        [](PyDataset& py_dt, NumType perc) { return py_dt.subdata(perc); },
        "perc"_a);
    dataset_class.def(
        "split",
        [](PyDataset& py_dt, NumType perc) {
            auto split = py_dt.split(perc);
            return std::pair<PyDataset, PyDataset>(
                split.training_set, split.testing_set);
        },
        "perc"_a);
    dataset_class.def(
        "shuffle",
        [](PyDataset& py_dt, std::random_device::result_type seed) {
            return py_dt.shuffle(RneType(seed));
        }, "seed"_a=std::random_device{}());
    dataset_class.def(
        "min_max_normalization",
        [](PyDataset& py_dt, PyDatasetType min, PyDatasetType max, std::vector<SizeType> idx_vec = {}) {
            return py_dt.min_max_normalization(min, max, idx_vec);
        },
        "min"_a, "max"_a, "apply_to_indexes"_a=std::vector<SizeType>());
    dataset_class.def(
        "min_max_normalization",
        [](PyDataset& py_dt) {
            return py_dt.min_max_normalization();
        });
    dataset_class.def_static(
        "parse", &PyDataset::parse,
        "parser"_a,
        "label_encoding"_a=DatasetParser::LabelEncoding::DEFAULT_ENCODING,
        "sequence_size"_a=1);
}

void data_submodule(pybind11::module& subm)
{
    subm.doc() = "Python Edge Learning submodule for data management";

    dataset_class(subm);
}