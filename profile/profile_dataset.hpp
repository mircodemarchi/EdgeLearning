/***************************************************************************
 *            profile_dataset.hpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 * This file is part of Opera, under the MIT license.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef EDGE_LEARNING_PROFILE_DATASET_HPP
#define EDGE_LEARNING_PROFILE_DATASET_HPP

#include "profile.hpp"

#include "parser/mnist.hpp"
#include "parser/csv.hpp"
#include "parser/cifar.hpp"

#include <tuple>


using namespace EdgeLearning;
namespace fs = std::filesystem;


class ProfileDataset
{
public:
    const NumType PERCENTAGE_TESTING_DATASET = 0.2;
    const NumType PERCENTAGE_TRAINING_DATASET = 1 - PERCENTAGE_TESTING_DATASET;
    const NumType PERCENTAGE_EVALUATION_DATASET = 0.1;

    struct Info {
        Dataset<NumType> train;
        Dataset<NumType> evaluation;
        Dataset<NumType> test;
        LayerShape input_shape;
    };

    enum class Type {
        MNIST,
        CSV_EXECUTION_TIME,
        CIFAR10,
        CIFAR100,
    };

    ProfileDataset(Type dataset_type)
        : _dataset_type(dataset_type)
    { }

    Info load_dataset()
    {
        switch (_dataset_type) {
            case Type::CIFAR10:
            {
                return _load_cifar10_dataset();
            }
            case Type::CIFAR100:
            {
                return _load_cifar100_dataset();
            }
            case Type::CSV_EXECUTION_TIME:
            {
                return _load_execution_time_dataset();
            }
            case Type::MNIST:
            default:
            {
                return _load_mnist_dataset();
            }
        }
    }

    operator std::string()
    {
        switch (_dataset_type) {
            case Type::CIFAR10:
            {
                return "cifar10";
            }
            case Type::CIFAR100:
            {
                return "cifar100";
            }
            case Type::CSV_EXECUTION_TIME:
            {
                return "execution_time";
            }
            case Type::MNIST:
            {
                return "mnist";
            }
            default:
            {
                return "";
            }
        }
    }

private:

    Info _load_mnist_dataset()
    {
        const std::string MNIST_TRAINING_IMAGES_FN =
            "train-images.idx3-ubyte";
        const std::string MNIST_TRAINING_LABELS_FN =
            "train-labels.idx1-ubyte";
        const std::string MNIST_TESTING_IMAGES_FN =
            "t10k-images.idx3-ubyte";
        const std::string MNIST_TESTING_LABELS_FN =
            "t10k-labels.idx1-ubyte";

        const std::filesystem::path MNIST_RESOURCE_ROOT =
            std::filesystem::path(__FILE__).parent_path() / ".." / "data";
        const std::filesystem::path MNIST_TRAINING_IMAGES_FP =
            MNIST_RESOURCE_ROOT / MNIST_TRAINING_IMAGES_FN;
        const std::filesystem::path MNIST_TRAINING_LABELS_FP =
            MNIST_RESOURCE_ROOT / MNIST_TRAINING_LABELS_FN;
        const std::filesystem::path MNIST_TESTING_IMAGES_FP =
            MNIST_RESOURCE_ROOT / MNIST_TESTING_IMAGES_FN;
        const std::filesystem::path MNIST_TESTING_LABELS_FP =
            MNIST_RESOURCE_ROOT / MNIST_TESTING_LABELS_FN;

        auto mnist_training = Mnist(
            MNIST_TRAINING_IMAGES_FP,
            MNIST_TRAINING_LABELS_FP);
        auto mnist_testing = Mnist(
            MNIST_TESTING_IMAGES_FP,
            MNIST_TESTING_LABELS_FP);
        auto data_training = Dataset<NumType>::parse(
            mnist_training,
            DatasetParser::LabelEncoding::ONE_HOT_ENCODING);
        data_training = data_training.min_max_normalization(
            0, 255, data_training.trainset_idx());
        auto data_testing = Dataset<NumType>::parse(
            mnist_testing,
            DatasetParser::LabelEncoding::ONE_HOT_ENCODING);
        data_testing = data_testing.min_max_normalization(
            0, 255, data_testing.trainset_idx());
        auto data_evaluation = data_training.subdata(
            PERCENTAGE_EVALUATION_DATASET);

        _print_info(data_training, data_testing, data_evaluation);
        return {data_training,
                data_evaluation,
                data_testing,
                DLMath::Shape3d{MnistImage::IMAGE_SIDE, MnistImage::IMAGE_SIDE}};
    }

    Info _load_execution_time_dataset()
    {
        const std::string DATA_TRAINING_FN = "execution-time.csv";
        const std::filesystem::path DATA_TRAINING_FP = std::filesystem::path(
            __FILE__).parent_path() / ".." / "data" / DATA_TRAINING_FN;

        auto csv = CSV(DATA_TRAINING_FP.string(),
                       { TypeChecker::Type::AUTO }, ',', {4});
        auto data = Dataset<NumType>::parse(csv);
        auto data_split = data.split(PERCENTAGE_TRAINING_DATASET);
        auto data_training = data_split.training_set;
        auto data_testing = data_split.testing_set;
        auto data_evaluation = data_split.training_set.subdata(
            PERCENTAGE_EVALUATION_DATASET);

        _print_info(data_training, data_testing, data_evaluation);
        return {data_training,
                data_evaluation,
                data_testing,
                data_split.training_set.trainset_idx().size()};
    }

    Info _load_cifar10_dataset()
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

        const std::filesystem::path CIFAR_RESOURCE_ROOT =
            std::filesystem::path(__FILE__).parent_path() / ".." / "data";
        const std::filesystem::path CIFAR10_TEST_FP =
            CIFAR_RESOURCE_ROOT / CIFAR10_TEST_FN;
        const std::filesystem::path CIFAR10_META_FP =
            CIFAR_RESOURCE_ROOT / CIFAR10_META_FN;

        Dataset<NumType> data_training;
        for (const auto& batch_fn: CIFAR10)
        {
            const std::filesystem::path batch_fp =
                CIFAR_RESOURCE_ROOT / batch_fn;
            auto cifar_batch = Cifar(batch_fp, CIFAR10_META_FP,
                                     CifarShapeOrder::CHN_ROW_COL,
                                     CifarDataset::CIFAR_10);
            auto cifar_batch_ds = Dataset<NumType>::parse(
                cifar_batch, DatasetParser::LabelEncoding::ONE_HOT_ENCODING);
            data_training = Dataset<NumType>::concatenate(
                data_training, cifar_batch_ds);
        }
        data_training = data_training.min_max_normalization(
            0, 255, data_training.trainset_idx());

        auto cifar_test = Cifar(CIFAR10_TEST_FP, CIFAR10_META_FP,
                                CifarShapeOrder::CHN_ROW_COL,
                                CifarDataset::CIFAR_10);
        auto data_testing = Dataset<NumType>::parse(
            cifar_test, DatasetParser::LabelEncoding::ONE_HOT_ENCODING);
        data_testing = data_testing.min_max_normalization(
            0, 255, data_testing.trainset_idx());

        auto data_evaluation = data_training.subdata(
            PERCENTAGE_EVALUATION_DATASET);

        _print_info(data_training, data_testing, data_evaluation);
        return {data_training,
                data_testing,
                data_evaluation,
                DLMath::Shape3d{CifarImage::IMAGE_CHANNELS,
                                CifarImage::IMAGE_SIDE,
                                CifarImage::IMAGE_SIDE}};
    }

    Info _load_cifar100_dataset()
    {

        const std::string CIFAR100_TRAIN_FN = "train.bin";
        const std::string CIFAR100_TEST_FN = "test.bin";
        const std::string CIFAR100_COARSE_META_FN = "coarse_label_names.txt";
        const std::string CIFAR100_FINE_META_FN = "fine_label_names.txt";

        const std::filesystem::path CIFAR_RESOURCE_ROOT =
            std::filesystem::path(__FILE__).parent_path() / ".." / "data";
        const std::filesystem::path CIFAR100_TRAIN_FP =
            CIFAR_RESOURCE_ROOT / CIFAR100_TRAIN_FN;
        const std::filesystem::path CIFAR100_TEST_FP =
            CIFAR_RESOURCE_ROOT / CIFAR100_TEST_FN;
        const std::filesystem::path CIFAR100_COARSE_META_FP =
            CIFAR_RESOURCE_ROOT / CIFAR100_COARSE_META_FN;
        const std::filesystem::path CIFAR100_FINE_META_FP =
            CIFAR_RESOURCE_ROOT / CIFAR100_FINE_META_FN;

        auto cifar_train = Cifar(CIFAR100_TRAIN_FP, CIFAR100_COARSE_META_FP,
                                 CifarShapeOrder::CHN_ROW_COL,
                                 CifarDataset::CIFAR_100,
                                 CIFAR100_FINE_META_FP);
        auto data_training = Dataset<NumType>::parse(
            cifar_train, DatasetParser::LabelEncoding::ONE_HOT_ENCODING);
        data_training = data_training.min_max_normalization(
            0, 255, data_training.trainset_idx());

        auto cifar_test = Cifar(CIFAR100_TEST_FP, CIFAR100_COARSE_META_FP,
                                CifarShapeOrder::CHN_ROW_COL,
                                CifarDataset::CIFAR_100,
                                CIFAR100_FINE_META_FP);
        auto data_testing = Dataset<NumType>::parse(
            cifar_test, DatasetParser::LabelEncoding::ONE_HOT_ENCODING);
        data_testing = data_testing.min_max_normalization(
            0, 255, data_testing.trainset_idx());

        auto data_evaluation = data_training.subdata(
            PERCENTAGE_EVALUATION_DATASET);

        _print_info(data_training, data_testing, data_evaluation);
        return {data_training,
                data_testing,
                data_evaluation,
                DLMath::Shape3d{CifarImage::IMAGE_CHANNELS,
                                CifarImage::IMAGE_SIDE,
                                CifarImage::IMAGE_SIDE}};
    }

    void _print_info(const Dataset<NumType>& data_training,
                     const Dataset<NumType>& data_testing,
                     const Dataset<NumType>& data_evaluation)
    {
        std::cout << "data training shape: ("
                  << data_training.size() << ", "
                  << data_training.feature_size() << ")" << std::endl;
        std::cout << "data evaluation shape: ("
                  << data_evaluation.size() << ", "
                  << data_evaluation.feature_size() << ")" << std::endl;
        std::cout << "data testing shape: ("
                  << data_testing.size() << ", "
                  << data_testing.feature_size() << ")" << std::endl;
    }

    Type _dataset_type;
};

#endif // EDGE_LEARNING_PROFILE_DATASET_HPP