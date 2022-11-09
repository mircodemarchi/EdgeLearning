/***************************************************************************
 *            parser_submodule.cpp
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

#include "parser_submodule.hpp"

#include "parser/parser.hpp"
#include "parser/type_checker.hpp"

#include <set>

namespace EdgeLearning {

} // namespace EdgeLearning

using namespace EdgeLearning;

class PyDatasetParser : public DatasetParser {
public:
    using DatasetParser::DatasetParser;

    std::vector<NumType> entry(SizeType i) override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<NumType>, DatasetParser, entry, i
        );
    }

    SizeType entries_amount() const override {
        PYBIND11_OVERRIDE_PURE(
            SizeType, DatasetParser, entries_amount,
        );
    }

    SizeType feature_size() const override {
        PYBIND11_OVERRIDE_PURE(
            SizeType, DatasetParser, feature_size,
        );
    }

    std::set<SizeType> labels_idx() const override {
        PYBIND11_OVERRIDE_PURE(
            std::set<SizeType>, DatasetParser, feature_size,
        );
    }
};

void parser_submodule(pybind11::module& subm)
{
    subm.doc() = "Python Edge Learning submodule for parsing datasets";

    py::class_<Parser> parser_class(subm, "Parser");
    parser_class.def(py::init<>());

    py::class_<DatasetParser, PyDatasetParser, Parser> dataset_parser_class(
        subm, "DatasetParser");
    dataset_parser_class.def(py::init<>());
    dataset_parser_class.def("entry", &DatasetParser::entry, "idx"_a);
    dataset_parser_class.def("entries_amount", &DatasetParser::entries_amount);
    dataset_parser_class.def("feature_size", &DatasetParser::feature_size);
    dataset_parser_class.def("label_idx", &DatasetParser::labels_idx);
    dataset_parser_class.def("unique", &DatasetParser::unique, "idx"_a);
    dataset_parser_class.def("unique_map", &DatasetParser::unique_map, "idx"_a);
    dataset_parser_class.def(
        "encoding_feature_size",
        &DatasetParser::encoding_feature_size, "label_encoding"_a);
    dataset_parser_class.def(
        "encoding_labels_idx",
        &DatasetParser::encoding_labels_idx, "label_encoding"_a);
    dataset_parser_class.def(
        "data_to_encoding",
        &DatasetParser::data_to_encoding, "label_encoding"_a);

    py::enum_<DatasetParser::LabelEncoding> label_encoding(
        dataset_parser_class, "LabelEncoding");
    label_encoding.value("ONE_HOT_ENCODING", DatasetParser::LabelEncoding::ONE_HOT_ENCODING);
    label_encoding.value("DEFAULT_ENCODING", DatasetParser::LabelEncoding::DEFAULT_ENCODING);
    label_encoding.export_values();
}