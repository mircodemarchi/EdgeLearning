/***************************************************************************
 *            parser/test_mnist.cpp
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
#include "parser/mnist.hpp"
#include "data/path.hpp"

#include <vector>
#include <stdexcept>

using namespace std;
using namespace EdgeLearning;

class TestMnist {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_uint32_endian_order());
        EDGE_LEARNING_TEST_CALL(test_mnist_image());
        EDGE_LEARNING_TEST_CALL(test_mnist_label());
        EDGE_LEARNING_TEST_CALL(test_mnist());
        EDGE_LEARNING_TEST_CALL(test_dataset_parser());
    }
private:
    const std::string FIRST10_TRAINING_IMAGES_FN =
        "first10-train-images-idx3-ubyte";
    const std::string FIRST10_TRAINING_LABELS_FN =
        "first10-train-labels-idx1-ubyte";
    const std::string FIRST10_TESTING_IMAGES_FN =
        "first10-t10k-images-idx3-ubyte";
    const std::string FIRST10_TESTING_LABELS_FN =
        "first10-t10k-labels-idx1-ubyte";

    const std::filesystem::path MNIST_RESOURCE_ROOT =
        std::filesystem::path(__FILE__).parent_path() / "resource" / "mnist";
    const std::filesystem::path FIRST10_TRAINING_IMAGES_FP =
        MNIST_RESOURCE_ROOT / FIRST10_TRAINING_IMAGES_FN;
    const std::filesystem::path FIRST10_TRAINING_LABELS_FP =
        MNIST_RESOURCE_ROOT / FIRST10_TRAINING_LABELS_FN;
    const std::filesystem::path FIRST10_TESTING_IMAGES_FP =
        MNIST_RESOURCE_ROOT / FIRST10_TESTING_IMAGES_FN;
    const std::filesystem::path FIRST10_TESTING_LABELS_FP =
        MNIST_RESOURCE_ROOT / FIRST10_TESTING_LABELS_FN;

    void test_uint32_endian_order()
    {
        uint32_t value = 0x12345678;
        uint32_t truth_value = 0x78563412;
        EDGE_LEARNING_TEST_EQUAL(uint32_endian_order(value), truth_value);

        std::ifstream mnist_train_images(FIRST10_TRAINING_IMAGES_FP);
        EDGE_LEARNING_TEST_EQUAL(read_uint32_endian_order(mnist_train_images),
                                 Mnist::IMAGE_MAGIC);
        EDGE_LEARNING_TEST_EQUAL(read_uint32_endian_order(mnist_train_images),
                                 60000);
        EDGE_LEARNING_TEST_EQUAL(read_uint32_endian_order(mnist_train_images),
                                 MnistImage::IMAGE_SIDE);
        EDGE_LEARNING_TEST_EQUAL(read_uint32_endian_order(mnist_train_images),
                                 MnistImage::IMAGE_SIDE);

        std::ifstream mnist_train_labels(FIRST10_TRAINING_LABELS_FP);
        EDGE_LEARNING_TEST_EQUAL(read_uint32_endian_order(mnist_train_labels),
                                 Mnist::LABEL_MAGIC);
        EDGE_LEARNING_TEST_EQUAL(read_uint32_endian_order(mnist_train_labels),
                                 60000);
    }

    void test_mnist_image() {
        std::ifstream mnist_train_images(FIRST10_TRAINING_IMAGES_FP);
        mnist_train_images.seekg(Mnist::IMAGE_HEADER_SIZE,
                                 mnist_train_images.beg);
        EDGE_LEARNING_TEST_TRY(MnistImage(mnist_train_images, 0));

        auto mnist_image = MnistImage(mnist_train_images, 1);
        EDGE_LEARNING_TEST_EQUAL(mnist_image.idx(), 1);
        mnist_image = MnistImage(mnist_train_images, 2);
        EDGE_LEARNING_TEST_EQUAL(mnist_image.idx(), 2);

        EDGE_LEARNING_TEST_TRY(mnist_image.data());
        auto& image = mnist_image.data();
        EDGE_LEARNING_TEST_TRY(mnist_image.data()[0] = 'i');
        EDGE_LEARNING_TEST_EQUAL(mnist_image.data()[0], image[0]);

        EDGE_LEARNING_TEST_PRINT(std::string(mnist_image));
        EDGE_LEARNING_TEST_PRINT(mnist_image);

        mnist_train_images.close();
    }

    void test_mnist_label() {
        std::ifstream mnist_train_labels(FIRST10_TRAINING_LABELS_FP);
        mnist_train_labels.seekg(Mnist::IMAGE_HEADER_SIZE,
                                 mnist_train_labels.beg);
        EDGE_LEARNING_TEST_TRY(MnistLabel(mnist_train_labels, 0));

        auto mnist_label = MnistLabel(mnist_train_labels, 1);
        EDGE_LEARNING_TEST_EQUAL(mnist_label.idx(), 1);
        mnist_label = MnistLabel(mnist_train_labels, 2);
        EDGE_LEARNING_TEST_EQUAL(mnist_label.idx(), 2);

        EDGE_LEARNING_TEST_TRY(mnist_label.data());
        auto& label = mnist_label.data();
        EDGE_LEARNING_TEST_TRY(mnist_label.data() = 'l');
        EDGE_LEARNING_TEST_EQUAL(mnist_label.data(), label);

        EDGE_LEARNING_TEST_PRINT(std::string(mnist_label));
        EDGE_LEARNING_TEST_PRINT(mnist_label);

        mnist_train_labels.close();
    }

    void test_mnist() {
        EDGE_LEARNING_TEST_TRY(Mnist(
            FIRST10_TRAINING_IMAGES_FP, FIRST10_TRAINING_LABELS_FP));
        auto mnist = Mnist(
            FIRST10_TRAINING_IMAGES_FP, FIRST10_TRAINING_LABELS_FP);

        EDGE_LEARNING_TEST_EQUAL(mnist.size(), 60000);
        EDGE_LEARNING_TEST_EQUAL(mnist.side(), 28);
        EDGE_LEARNING_TEST_EQUAL(mnist.height(), 28);
        EDGE_LEARNING_TEST_EQUAL(mnist.width(), 28);
        EDGE_LEARNING_TEST_EQUAL(std::get<0>(mnist.shape()), 28);
        EDGE_LEARNING_TEST_EQUAL(std::get<1>(mnist.shape()), 28);

        EDGE_LEARNING_TEST_TRY(mnist.image(0));
        auto first_image = mnist.image(0);
        EDGE_LEARNING_TEST_TRY(mnist.label(0));
        auto first_label = mnist.label(0);
        EDGE_LEARNING_TEST_EQUAL(mnist[0].image.idx(), first_image.idx());
        EDGE_LEARNING_TEST_EQUAL(mnist[0].label.idx(), first_label.idx());
        EDGE_LEARNING_TEST_EQUAL(mnist[0].image.data()[0],
                                 first_image.data()[0]);
        EDGE_LEARNING_TEST_EQUAL(mnist[0].label.data(), first_label.data());

        for (std::size_t i = 0; i < 10; ++i)
        {
            EDGE_LEARNING_TEST_PRINT("test print " + std::to_string(i));
            EDGE_LEARNING_TEST_PRINT(mnist[i].image);
            EDGE_LEARNING_TEST_PRINT(mnist[i].label);
        }

        auto first10_training_images_fail_fp = fs::path(
            FIRST10_TRAINING_IMAGES_FP.string() + "fail");
        EDGE_LEARNING_TEST_FAIL(Mnist(
            first10_training_images_fail_fp,
            FIRST10_TRAINING_LABELS_FP));
        EDGE_LEARNING_TEST_THROWS(Mnist(
            first10_training_images_fail_fp,
            FIRST10_TRAINING_LABELS_FP), std::runtime_error);
        auto first10_training_labels_fail_fp = fs::path(
            FIRST10_TRAINING_LABELS_FP.string() + "fail");
        EDGE_LEARNING_TEST_FAIL(Mnist(
            FIRST10_TRAINING_IMAGES_FP,
            first10_training_labels_fail_fp));
        EDGE_LEARNING_TEST_THROWS(Mnist(
            FIRST10_TRAINING_IMAGES_FP,
            first10_training_labels_fail_fp), std::runtime_error);

        EDGE_LEARNING_TEST_FAIL(Mnist(
            FIRST10_TRAINING_IMAGES_FP,
            FIRST10_TRAINING_IMAGES_FP));
        EDGE_LEARNING_TEST_THROWS(Mnist(
            FIRST10_TRAINING_IMAGES_FP,
            FIRST10_TRAINING_IMAGES_FP), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(Mnist(
            FIRST10_TRAINING_LABELS_FP,
            FIRST10_TRAINING_LABELS_FP));
        EDGE_LEARNING_TEST_THROWS(Mnist(
            FIRST10_TRAINING_LABELS_FP,
            FIRST10_TRAINING_LABELS_FP), std::runtime_error);

        EDGE_LEARNING_TEST_FAIL(Mnist(
            FIRST10_TESTING_IMAGES_FP,
            FIRST10_TRAINING_LABELS_FP));
        EDGE_LEARNING_TEST_THROWS(Mnist(
            FIRST10_TESTING_IMAGES_FP,
            FIRST10_TRAINING_LABELS_FP), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(Mnist(
            FIRST10_TRAINING_IMAGES_FP,
            FIRST10_TESTING_LABELS_FP));
        EDGE_LEARNING_TEST_THROWS(Mnist(
            FIRST10_TRAINING_IMAGES_FP,
            FIRST10_TESTING_LABELS_FP), std::runtime_error);

        fs::copy(FIRST10_TRAINING_IMAGES_FP,
                 fs::path("first10-train-images-idx3-ubyte-copy"));
        std::ofstream mnist_train_images_copy_out(
            "first10-train-images-idx3-ubyte-copy",
            std::ofstream::out | std::ofstream::in);
        mnist_train_images_copy_out.seekp(8, mnist_train_images_copy_out.beg);
        std::uint32_t new_side = 0x1d000000;
        mnist_train_images_copy_out.write(
            reinterpret_cast<char*>(&new_side), 4);
        mnist_train_images_copy_out.write(
            reinterpret_cast<char*>(&new_side), 4);
        mnist_train_images_copy_out.close();
        EDGE_LEARNING_TEST_FAIL(Mnist(
            fs::path("first10-train-images-idx3-ubyte-copy"),
            FIRST10_TRAINING_LABELS_FP));
        EDGE_LEARNING_TEST_THROWS(Mnist(
            fs::path("first10-train-images-idx3-ubyte-copy"),
            FIRST10_TRAINING_LABELS_FP), std::runtime_error);
        fs::remove(fs::path("first10-train-images-idx3-ubyte-copy"));
    }

    void test_dataset_parser() {
        auto mnist_dp = Mnist(
            FIRST10_TRAINING_IMAGES_FP, FIRST10_TRAINING_LABELS_FP);

        EDGE_LEARNING_TEST_EQUAL(mnist_dp.feature_size(),
                                 mnist_dp.width() * mnist_dp.height() + 1);
        EDGE_LEARNING_TEST_EQUAL(mnist_dp.entries_amount(), mnist_dp.size());
        EDGE_LEARNING_TEST_EQUAL(mnist_dp.entry(0).size(),
                                 mnist_dp.width() * mnist_dp.height() + 1);
        EDGE_LEARNING_TEST_EQUAL(mnist_dp.entry(1).size(),
                                 mnist_dp.width() * mnist_dp.height() + 1);
        EDGE_LEARNING_TEST_EQUAL(mnist_dp.labels_idx().size(), 1);
    }
};

int main() {
    TestMnist().test();
    return EDGE_LEARNING_TEST_FAILURES;
}



