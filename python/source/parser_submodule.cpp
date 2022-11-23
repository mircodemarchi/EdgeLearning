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
#include "parser/mnist.hpp"
#include "parser/cifar.hpp"
#include "parser/csv.hpp"
#include "parser/json.hpp"
#include "data/path.hpp"
#include "data/dataset.hpp"

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

static py::tuple load_mnist(std::string folder_path)
{
    const std::string MNIST_TRAINING_IMAGES_FN =
        "train-images.idx3-ubyte";
    const std::string MNIST_TRAINING_LABELS_FN =
        "train-labels.idx1-ubyte";
    const std::string MNIST_TESTING_IMAGES_FN =
        "t10k-images.idx3-ubyte";
    const std::string MNIST_TESTING_LABELS_FN =
        "t10k-labels.idx1-ubyte";

    std::filesystem::path mnist_resource_root(folder_path);
    std::filesystem::path mnist_training_images_fp  =
        mnist_resource_root / MNIST_TRAINING_IMAGES_FN;
    std::filesystem::path mnist_training_labels_fp =
        mnist_resource_root / MNIST_TRAINING_LABELS_FN;
    std::filesystem::path mnist_testing_images_fp =
        mnist_resource_root / MNIST_TESTING_IMAGES_FN;
    std::filesystem::path mnist_testing_labels_fp =
        mnist_resource_root / MNIST_TESTING_LABELS_FN;

    auto mnist_training = Mnist(
        mnist_training_images_fp,
        mnist_training_labels_fp);
    auto mnist_testing = Mnist(
        mnist_testing_images_fp,
        mnist_testing_labels_fp);
    auto data_training = Dataset<NumType>::parse(
        mnist_training,
        DatasetParser::LabelEncoding::ONE_HOT_ENCODING);
    auto data_testing = Dataset<NumType>::parse(
        mnist_testing,
        DatasetParser::LabelEncoding::ONE_HOT_ENCODING);
    return py::make_tuple(data_training, data_testing);
}

static py::tuple load_cifar10(std::string folder_path)
{
    const std::string CIFAR10_BATCH1_FN = "data_batch_1.bin";
    const std::string CIFAR10_BATCH2_FN = "data_batch_2.bin";
    const std::string CIFAR10_BATCH3_FN = "data_batch_3.bin";
    const std::string CIFAR10_BATCH4_FN = "data_batch_4.bin";
    const std::string CIFAR10_BATCH5_FN = "data_batch_5.bin";
    const std::string CIFAR10_TEST_FN = "test_batch.bin";
    const std::string CIFAR10_META_FN = "batches.meta.txt";

    const std::vector<std::string> CIFAR10 = {
        CIFAR10_BATCH1_FN,
        CIFAR10_BATCH2_FN,
        CIFAR10_BATCH3_FN,
        CIFAR10_BATCH4_FN,
        CIFAR10_BATCH5_FN
    };

    std::filesystem::path cifar10_resource_root(folder_path);
    std::filesystem::path cifar10_test_fp =
        cifar10_resource_root / CIFAR10_TEST_FN;
    std::filesystem::path cifar10_meta_fp=
        cifar10_resource_root / CIFAR10_META_FN;

    Dataset<NumType> data_training;
    for (const auto& batch_fn: CIFAR10)
    {
        std::filesystem::path batch_fp = cifar10_resource_root / batch_fn;
        auto cifar_batch = Cifar(batch_fp, cifar10_meta_fp,
                                 CifarShapeOrder::CHN_ROW_COL,
                                 CifarDataset::CIFAR_10);
        auto cifar_batch_ds = Dataset<NumType>::parse(
            cifar_batch, DatasetParser::LabelEncoding::ONE_HOT_ENCODING);
        data_training = Dataset<NumType>::concatenate(
            data_training, cifar_batch_ds);
    }

    auto cifar_test = Cifar(cifar10_test_fp, cifar10_meta_fp,
                            CifarShapeOrder::CHN_ROW_COL,
                            CifarDataset::CIFAR_10);
    auto data_testing = Dataset<NumType>::parse(
        cifar_test, DatasetParser::LabelEncoding::ONE_HOT_ENCODING);

    return py::make_tuple(data_training, data_testing);
}

static py::tuple load_cifar100(std::string folder_path)
{
    const std::string CIFAR100_TRAIN_FN = "train.bin";
    const std::string CIFAR100_TEST_FN = "test.bin";
    const std::string CIFAR100_COARSE_META_FN = "coarse_label_names.txt";
    const std::string CIFAR100_FINE_META_FN = "fine_label_names.txt";

    std::filesystem::path cifar_resource_root(folder_path);
    std::filesystem::path cifar100_train_fp =
        cifar_resource_root / CIFAR100_TRAIN_FN;
    std::filesystem::path cifar100_test_fp =
        cifar_resource_root / CIFAR100_TEST_FN;
    std::filesystem::path cifar100_coarse_meta_fp =
        cifar_resource_root / CIFAR100_COARSE_META_FN;
    std::filesystem::path cifar100_fine_meta_fp =
        cifar_resource_root / CIFAR100_FINE_META_FN;

    auto cifar_train = Cifar(cifar100_train_fp, cifar100_coarse_meta_fp,
                             CifarShapeOrder::CHN_ROW_COL,
                             CifarDataset::CIFAR_100,
                             cifar100_fine_meta_fp);
    auto data_training = Dataset<NumType>::parse(
        cifar_train, DatasetParser::LabelEncoding::ONE_HOT_ENCODING);

    auto cifar_test = Cifar(cifar100_test_fp, cifar100_coarse_meta_fp,
                            CifarShapeOrder::CHN_ROW_COL,
                            CifarDataset::CIFAR_100,
                            cifar100_fine_meta_fp);
    auto data_testing = Dataset<NumType>::parse(
        cifar_test, DatasetParser::LabelEncoding::ONE_HOT_ENCODING);

    return py::make_tuple(data_training, data_testing);
}

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

    subm.def("load_mnist", &load_mnist, "folder_path"_a);
    subm.def("load_cifar10", &load_cifar10, "folder_path"_a);
    subm.def("load_cifar100", &load_cifar100, "folder_path"_a);
}