/***************************************************************************
 *            parser/test_cifar.cpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 *  This file is part of EdgeLearning.
 *
 *  EdgeLearning is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  EdgeLearning is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with EdgeLearning.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "test.hpp"
#include "parser/cifar.hpp"

#include <filesystem>
#include <vector>
#include <stdexcept>

using namespace std;
using namespace EdgeLearning;

class TestCifar {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_cifar10_image());
        EDGE_LEARNING_TEST_CALL(test_cifar100_image());
        EDGE_LEARNING_TEST_CALL(test_cifar10_label());
        EDGE_LEARNING_TEST_CALL(test_cifar100_label());
        EDGE_LEARNING_TEST_CALL(test_cifar10());
        EDGE_LEARNING_TEST_CALL(test_cifar100());
    }
private:
    const std::string FIRST10_CIFAR10_BATCH1_FN = "first10_data_batch_1.bin";
    const std::string CIFAR10_META_FN = "batches.meta.txt";
    const std::string FIRST10_CIFAR100_TRAIN_FN = "first10_train.bin";
    const std::string CIFAR100_COARSE_META_FN = "coarse_label_names.txt";
    const std::string CIFAR100_FINE_META_FN = "fine_label_names.txt";

    const std::filesystem::path CIFAR_RESOURCE_ROOT =
        std::filesystem::path(__FILE__).parent_path() / "resource" / "cifar";
    const std::filesystem::path FIRST10_CIFAR10_BATCH1_FP =
        CIFAR_RESOURCE_ROOT / FIRST10_CIFAR10_BATCH1_FN;
    const std::filesystem::path CIFAR10_META_FP =
        CIFAR_RESOURCE_ROOT / CIFAR10_META_FN;
    const std::filesystem::path FIRST10_CIFAR100_TRAIN_FP =
        CIFAR_RESOURCE_ROOT / FIRST10_CIFAR100_TRAIN_FN;
    const std::filesystem::path CIFAR100_COARSE_META_FP =
        CIFAR_RESOURCE_ROOT / CIFAR100_COARSE_META_FN;
    const std::filesystem::path CIFAR100_FINE_META_FP =
        CIFAR_RESOURCE_ROOT / CIFAR100_FINE_META_FN;


    void test_cifar10_image() {
        std::ifstream cifar10_train_images(FIRST10_CIFAR10_BATCH1_FP);
        cifar10_train_images.seekg(1, cifar10_train_images.beg);
        EDGE_LEARNING_TEST_TRY(CifarImage(cifar10_train_images, 0));

        auto cifar_image = CifarImage(cifar10_train_images, 1);
        EDGE_LEARNING_TEST_EQUAL(cifar_image.idx(), 1);
        cifar_image = CifarImage(cifar10_train_images, 2);
        EDGE_LEARNING_TEST_EQUAL(cifar_image.idx(), 2);

        EDGE_LEARNING_TEST_TRY((void) cifar_image.data());
        EDGE_LEARNING_TEST_EQUAL(
            cifar_image.data().size(),
            CifarImage::IMAGE_SIDE * CifarImage::IMAGE_SIDE
                * CifarImage::IMAGE_CHANNELS);

        EDGE_LEARNING_TEST_PRINT(std::string(cifar_image));
        EDGE_LEARNING_TEST_PRINT(cifar_image);

        cifar10_train_images.seekg(1, cifar10_train_images.beg);
        EDGE_LEARNING_TEST_TRY(CifarImage(cifar10_train_images, 0,
                                          CifarShapeOrder::ROW_COL_CHN));

        auto cifar_image_order = CifarImage(cifar10_train_images, 1,
                                            CifarShapeOrder::ROW_COL_CHN);
        EDGE_LEARNING_TEST_EQUAL(cifar_image_order.idx(), 1);
        cifar_image_order = CifarImage(cifar10_train_images, 2,
                                       CifarShapeOrder::ROW_COL_CHN);
        EDGE_LEARNING_TEST_EQUAL(cifar_image_order.idx(), 2);

        EDGE_LEARNING_TEST_TRY((void) cifar_image_order.data());
        EDGE_LEARNING_TEST_EQUAL(
            cifar_image_order.data().size(),
            CifarImage::IMAGE_SIDE * CifarImage::IMAGE_SIDE
                * CifarImage::IMAGE_CHANNELS);

        EDGE_LEARNING_TEST_EQUAL(std::string(cifar_image),
                                 std::string(cifar_image_order));
        EDGE_LEARNING_TEST_PRINT(std::string(cifar_image_order));
        EDGE_LEARNING_TEST_PRINT(cifar_image_order);

        auto image_cpy = cifar_image.data();
        auto image_order_cpy = cifar_image_order.data();
        EDGE_LEARNING_TEST_EQUAL(image_cpy[0], image_order_cpy[0]);
        EDGE_LEARNING_TEST_EQUAL(image_cpy[1],
                                 image_order_cpy[CifarImage::IMAGE_CHANNELS]);

        cifar10_train_images.close();
    }

    void test_cifar100_image() {
        std::ifstream cifar100_train_images(FIRST10_CIFAR100_TRAIN_FP);
        cifar100_train_images.seekg(2, cifar100_train_images.beg);
        EDGE_LEARNING_TEST_TRY(CifarImage(cifar100_train_images, 0));

        auto cifar_image = CifarImage(cifar100_train_images, 1);
        EDGE_LEARNING_TEST_EQUAL(cifar_image.idx(), 1);
        cifar_image = CifarImage(cifar100_train_images, 2);
        EDGE_LEARNING_TEST_EQUAL(cifar_image.idx(), 2);

        EDGE_LEARNING_TEST_TRY((void) cifar_image.data());
        EDGE_LEARNING_TEST_EQUAL(
            cifar_image.data().size(),
            CifarImage::IMAGE_SIDE * CifarImage::IMAGE_SIDE
                * CifarImage::IMAGE_CHANNELS);

        EDGE_LEARNING_TEST_PRINT(std::string(cifar_image));
        EDGE_LEARNING_TEST_PRINT(cifar_image);

        cifar100_train_images.seekg(2, cifar100_train_images.beg);
        EDGE_LEARNING_TEST_TRY(CifarImage(cifar100_train_images, 0,
                                          CifarShapeOrder::ROW_COL_CHN));

        auto cifar_image_order = CifarImage(cifar100_train_images, 1,
                                            CifarShapeOrder::ROW_COL_CHN);
        EDGE_LEARNING_TEST_EQUAL(cifar_image_order.idx(), 1);
        cifar_image_order = CifarImage(cifar100_train_images, 2,
                                       CifarShapeOrder::ROW_COL_CHN);
        EDGE_LEARNING_TEST_EQUAL(cifar_image_order.idx(), 2);

        EDGE_LEARNING_TEST_TRY((void) cifar_image_order.data());
        EDGE_LEARNING_TEST_EQUAL(
            cifar_image_order.data().size(),
            CifarImage::IMAGE_SIDE * CifarImage::IMAGE_SIDE
                * CifarImage::IMAGE_CHANNELS);

        EDGE_LEARNING_TEST_PRINT(std::string(cifar_image_order));
        EDGE_LEARNING_TEST_PRINT(cifar_image_order);

        auto image_cpy = cifar_image.data();
        auto image_order_cpy = cifar_image_order.data();
        EDGE_LEARNING_TEST_EQUAL(image_cpy[0], image_order_cpy[0]);
        EDGE_LEARNING_TEST_EQUAL(image_cpy[1],
                                 image_order_cpy[CifarImage::IMAGE_CHANNELS]);

        cifar100_train_images.close();
    }
    void test_cifar10_label() {

        std::ifstream cifar10_train_labels(FIRST10_CIFAR10_BATCH1_FP);
        EDGE_LEARNING_TEST_TRY(CifarLabel(cifar10_train_labels, 0));

        auto cifar_label = CifarLabel(cifar10_train_labels, 1);
        EDGE_LEARNING_TEST_EQUAL(cifar_label.idx(), 1);
        cifar_label = CifarLabel(cifar10_train_labels, 2);
        EDGE_LEARNING_TEST_EQUAL(cifar_label.idx(), 2);

        EDGE_LEARNING_TEST_TRY(cifar_label.data());
        auto& label = cifar_label.data();
        EDGE_LEARNING_TEST_TRY(cifar_label.data() = 'l');
        EDGE_LEARNING_TEST_EQUAL(cifar_label.data(), label);

        EDGE_LEARNING_TEST_TRY(cifar_label.coarse_label());
        auto& coarse_label = cifar_label.coarse_label();
        EDGE_LEARNING_TEST_TRY(cifar_label.coarse_label() = 'l');
        EDGE_LEARNING_TEST_EQUAL(cifar_label.coarse_label(), coarse_label);

        EDGE_LEARNING_TEST_TRY(cifar_label.fine_label());
        auto& fine_label = cifar_label.fine_label();
        EDGE_LEARNING_TEST_TRY(cifar_label.fine_label() = 'l');
        EDGE_LEARNING_TEST_EQUAL(cifar_label.fine_label(), fine_label);

        EDGE_LEARNING_TEST_PRINT(std::string(cifar_label));
        EDGE_LEARNING_TEST_PRINT(cifar_label);

        cifar10_train_labels.close();
    }

    void test_cifar100_label() {
        std::ifstream cifar100_train_labels(FIRST10_CIFAR100_TRAIN_FP);
        EDGE_LEARNING_TEST_TRY(CifarLabel(cifar100_train_labels, 0,
                                          CifarDataset::CIFAR_100));

        auto cifar_label = CifarLabel(cifar100_train_labels, 1,
                                      CifarDataset::CIFAR_100);
        EDGE_LEARNING_TEST_EQUAL(cifar_label.idx(), 1);
        cifar_label = CifarLabel(cifar100_train_labels, 2,
                                 CifarDataset::CIFAR_100);
        EDGE_LEARNING_TEST_EQUAL(cifar_label.idx(), 2);

        EDGE_LEARNING_TEST_TRY(cifar_label.data());
        auto& label = cifar_label.data();
        EDGE_LEARNING_TEST_TRY(cifar_label.data() = 'l');
        EDGE_LEARNING_TEST_EQUAL(cifar_label.data(), label);

        EDGE_LEARNING_TEST_TRY(cifar_label.coarse_label());
        auto& coarse_label = cifar_label.coarse_label();
        EDGE_LEARNING_TEST_TRY(cifar_label.coarse_label() = 'l');
        EDGE_LEARNING_TEST_EQUAL(cifar_label.coarse_label(), coarse_label);

        EDGE_LEARNING_TEST_TRY(cifar_label.fine_label());
        auto& fine_label = cifar_label.fine_label();
        EDGE_LEARNING_TEST_TRY(cifar_label.fine_label() = 'l');
        EDGE_LEARNING_TEST_EQUAL(cifar_label.fine_label(), fine_label);

        EDGE_LEARNING_TEST_PRINT(std::string(cifar_label));
        EDGE_LEARNING_TEST_PRINT(cifar_label);

        cifar100_train_labels.close();
    }

    void test_cifar10() {
        EDGE_LEARNING_TEST_TRY(Cifar(
            FIRST10_CIFAR10_BATCH1_FP, CIFAR10_META_FP));
        auto cifar = Cifar(
            FIRST10_CIFAR10_BATCH1_FP, CIFAR10_META_FP);

        EDGE_LEARNING_TEST_EQUAL(cifar.size(), 10000);
        EDGE_LEARNING_TEST_EQUAL(cifar.side(), 32);
        EDGE_LEARNING_TEST_EQUAL(cifar.height(), 32);
        EDGE_LEARNING_TEST_EQUAL(cifar.width(), 32);
        EDGE_LEARNING_TEST_EQUAL(cifar.channels(), 3);
        EDGE_LEARNING_TEST_EQUAL(std::get<0>(cifar.shape()), 3);
        EDGE_LEARNING_TEST_EQUAL(std::get<1>(cifar.shape()), 32);
        EDGE_LEARNING_TEST_EQUAL(std::get<2>(cifar.shape()), 32);
        EDGE_LEARNING_TEST_EQUAL(cifar.label_names().size(), 10);
        EDGE_LEARNING_TEST_EQUAL(cifar.coarse_label_names().size(), 10);
        EDGE_LEARNING_TEST_EQUAL(cifar.fine_label_names().size(), 0);

        EDGE_LEARNING_TEST_TRY(cifar.image(0));
        auto first_image = cifar.image(0);
        EDGE_LEARNING_TEST_TRY(cifar.label(0));
        auto first_label = cifar.label(0);
        EDGE_LEARNING_TEST_EQUAL(cifar[0].image.idx(), first_image.idx());
        EDGE_LEARNING_TEST_EQUAL(cifar[0].label.idx(), first_label.idx());
        EDGE_LEARNING_TEST_EQUAL(cifar[0].image.data()[0],
                                 first_image.data()[0]);
        EDGE_LEARNING_TEST_EQUAL(cifar[0].label.data(), first_label.data());

        for (std::size_t i = 0; i < 10; ++i)
        {
            EDGE_LEARNING_TEST_PRINT("test print " + std::to_string(i));
            EDGE_LEARNING_TEST_PRINT(cifar[i].image);
            EDGE_LEARNING_TEST_PRINT(cifar[i].label);
        }

        auto cifar_order = Cifar(
            FIRST10_CIFAR10_BATCH1_FP, CIFAR10_META_FP,
            CifarShapeOrder::ROW_COL_CHN);
        EDGE_LEARNING_TEST_EQUAL(std::get<0>(cifar_order.shape()), 32);
        EDGE_LEARNING_TEST_EQUAL(std::get<1>(cifar_order.shape()), 32);
        EDGE_LEARNING_TEST_EQUAL(std::get<2>(cifar_order.shape()), 3);

        EDGE_LEARNING_TEST_TRY(cifar_order.image(0));
        first_image = cifar_order.image(0);;
        auto first_image_data = first_image.data();
        EDGE_LEARNING_TEST_EQUAL(cifar[0].image.data()[0],
                                 first_image.data()[0]);
        EDGE_LEARNING_TEST_EQUAL(cifar[0].image.data()[1],
                                 first_image_data[CifarImage::IMAGE_CHANNELS]);
    }

    void test_cifar100() {
        EDGE_LEARNING_TEST_TRY(Cifar(
            FIRST10_CIFAR100_TRAIN_FP, CIFAR100_COARSE_META_FP,
            CifarShapeOrder::CHN_ROW_COL, CifarDataset::CIFAR_100,
            CIFAR100_FINE_META_FP));
        auto cifar = Cifar(
            FIRST10_CIFAR100_TRAIN_FP, CIFAR100_COARSE_META_FP,
            CifarShapeOrder::CHN_ROW_COL, CifarDataset::CIFAR_100,
            CIFAR100_FINE_META_FP);

        EDGE_LEARNING_TEST_EQUAL(cifar.size(), 10000);
        EDGE_LEARNING_TEST_EQUAL(cifar.side(), 32);
        EDGE_LEARNING_TEST_EQUAL(cifar.height(), 32);
        EDGE_LEARNING_TEST_EQUAL(cifar.width(), 32);
        EDGE_LEARNING_TEST_EQUAL(cifar.channels(), 3);
        EDGE_LEARNING_TEST_EQUAL(std::get<0>(cifar.shape()), 3);
        EDGE_LEARNING_TEST_EQUAL(std::get<1>(cifar.shape()), 32);
        EDGE_LEARNING_TEST_EQUAL(std::get<2>(cifar.shape()), 32);
        EDGE_LEARNING_TEST_EQUAL(cifar.label_names().size(), 20);
        EDGE_LEARNING_TEST_EQUAL(cifar.coarse_label_names().size(), 20);
        EDGE_LEARNING_TEST_EQUAL(cifar.fine_label_names().size(), 100);

        EDGE_LEARNING_TEST_TRY(cifar.image(0));
        auto first_image = cifar.image(0);
        EDGE_LEARNING_TEST_TRY(cifar.label(0));
        auto first_label = cifar.label(0);
        EDGE_LEARNING_TEST_EQUAL(cifar[0].image.idx(), first_image.idx());
        EDGE_LEARNING_TEST_EQUAL(cifar[0].label.idx(), first_label.idx());
        EDGE_LEARNING_TEST_EQUAL(cifar[0].image.data()[0],
                                 first_image.data()[0]);
        EDGE_LEARNING_TEST_EQUAL(cifar[0].label.data(), first_label.data());

        for (std::size_t i = 0; i < 10; ++i)
        {
            EDGE_LEARNING_TEST_PRINT("test print " + std::to_string(i));
            EDGE_LEARNING_TEST_PRINT(cifar[i].image);
            EDGE_LEARNING_TEST_PRINT(cifar[i].label);
        }

        auto cifar_order = Cifar(
            FIRST10_CIFAR100_TRAIN_FP, CIFAR100_COARSE_META_FP,
            CifarShapeOrder::ROW_COL_CHN, CifarDataset::CIFAR_100,
            CIFAR100_FINE_META_FP);
        EDGE_LEARNING_TEST_EQUAL(std::get<0>(cifar_order.shape()), 32);
        EDGE_LEARNING_TEST_EQUAL(std::get<1>(cifar_order.shape()), 32);
        EDGE_LEARNING_TEST_EQUAL(std::get<2>(cifar_order.shape()), 3);

        EDGE_LEARNING_TEST_TRY(cifar_order.image(0));
        auto first_image_data = cifar.image(0).data();
        auto first_image_order_data = cifar_order.image(0).data();
        EDGE_LEARNING_TEST_EQUAL(first_image_data[0],
                                 first_image_order_data[0]);
        EDGE_LEARNING_TEST_EQUAL(
            first_image_data[1],
            first_image_order_data[CifarImage::IMAGE_CHANNELS]);
    }
};

int main() {
    TestCifar().test();
    return EDGE_LEARNING_TEST_FAILURES;
}



