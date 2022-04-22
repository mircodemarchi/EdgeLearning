/***************************************************************************
 *            dnn/test_convolutional.cpp
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
#include "dnn/convolutional.hpp"
#include "dnn/model.hpp"

using namespace std;
using namespace EdgeLearning;


class TestConvolutionalLayer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_layer());
        EDGE_LEARNING_TEST_CALL(test_convolutional_layer());
        EDGE_LEARNING_TEST_CALL(test_getter());
        EDGE_LEARNING_TEST_CALL(test_setter());
    }

private:
    void test_layer() {
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(
                auto l = ConvolutionalLayer(_m, "convolutional_layer_test"));
        EDGE_LEARNING_TEST_TRY(
                auto l = ConvolutionalLayer(_m, "convolutional_layer_test"));
        auto l = ConvolutionalLayer(_m, "convolutional_layer_test");
        EDGE_LEARNING_TEST_TRY(RneType r; l.init(r));
        // TODO: Manage forward with nullptr input.
        // EDGE_LEARNING_TEST_TRY(l.forward(nullptr));
        // EDGE_LEARNING_TEST_TRY(l.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "convolutional_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(ConvolutionalLayer l1_copy{l});
        EDGE_LEARNING_TEST_TRY(ConvolutionalLayer l2_copy{l});
        ConvolutionalLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(RneType r; l_copy.init(r));
        // TODO: Manage forward with nullptr input.
        // EDGE_LEARNING_TEST_TRY(l_copy.forward(nullptr));
        // EDGE_LEARNING_TEST_TRY(l_copy.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "convolutional_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(ConvolutionalLayer l_assign(_m); l_assign = l);
        EDGE_LEARNING_TEST_TRY(ConvolutionalLayer l_assign(_m); l_assign = l);
        ConvolutionalLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_TRY(RneType r; l_assign.init(r));
        // TODO: Manage forward with nullptr input.
        // EDGE_LEARNING_TEST_TRY(l_assign.forward(nullptr));
        // EDGE_LEARNING_TEST_TRY(l_assign.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "convolutional_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = ConvolutionalLayer(_m));
        EDGE_LEARNING_TEST_TRY(auto l2 = ConvolutionalLayer(_m));
        auto l_noname = ConvolutionalLayer(_m);
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());

        DLMath::Shape3d in_shape{3,3,3};
        DLMath::Shape2d k_shape{2,2};
        SizeType filters = 16;
        auto l_shape = ConvolutionalLayer(_m, "convolutional_layer_test",
                                          Layer::Activation::ReLU,
                                          in_shape, k_shape, filters);
        auto truth_output_size = ((in_shape.width - k_shape.width) + 1)
            * ((in_shape.height - k_shape.height) + 1) * filters;
        EDGE_LEARNING_TEST_EQUAL(l_shape.input_size(), in_shape.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape.output_size(), truth_output_size);
        EDGE_LEARNING_TEST_EQUAL(l_shape.last_input(), nullptr);
        EDGE_LEARNING_TEST_NOT_EQUAL(l_shape.last_output(), nullptr);
        ConvolutionalLayer l_shape_copy{l_shape};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), in_shape.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), truth_output_size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_NOT_EQUAL(l_shape_copy.last_output(), nullptr);
        ConvolutionalLayer l_shape_assign(_m); l_shape_assign = l_shape;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), in_shape.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(),
                                 truth_output_size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_NOT_EQUAL(l_shape_assign.last_output(), nullptr);
    }

    void test_convolutional_layer()
    {
        std::vector<NumType> v1{1,1,1, 1,1,1, 1,1,1,
                                1,1,1, 1,1,1, 1,1,1,
                                1,1,1, 1,1,1, 1,1,1};
        DLMath::Shape3d in_shape{3,3,3};
        DLMath::Shape2d k_shape{2,2};
        SizeType filters = 16;
        auto l = ConvolutionalLayer(_m, "convolutional_layer_test",
                                    Layer::Activation::ReLU,
                                    in_shape, k_shape, filters);
        EDGE_LEARNING_TEST_TRY(RneType r; l.init(r));
        EDGE_LEARNING_TEST_TRY(l.forward(v1.data()));
        EDGE_LEARNING_TEST_TRY(l.reverse(v1.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.last_input(), v1.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l.last_output(), nullptr);

        std::vector<NumType> v2{2,2,2, 2,2,2, 2,2,2,
                                2,2,2, 2,2,2, 2,2,2,
                                2,2,2, 2,2,2, 2,2,2};
        ConvolutionalLayer l_copy{l};
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), v1.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_output(), nullptr);
        EDGE_LEARNING_TEST_TRY(l_copy.forward(v2.data()));
        EDGE_LEARNING_TEST_TRY(l_copy.reverse(v2.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), v2.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_output(), nullptr);

        ConvolutionalLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), v1.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_output(), nullptr);
        EDGE_LEARNING_TEST_TRY(l_assign.forward(v2.data()));
        EDGE_LEARNING_TEST_TRY(l_assign.reverse(v2.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), v2.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_output(), nullptr);

        auto l_relu = ConvolutionalLayer(_m, "convolutional_layer_test_relu",
                                    Layer::Activation::ReLU,
                                    in_shape, k_shape, filters);
        EDGE_LEARNING_TEST_TRY(RneType r; l_relu.init(r));
        EDGE_LEARNING_TEST_TRY(l_relu.forward(v1.data()));
        EDGE_LEARNING_TEST_TRY(l_relu.reverse(v1.data()));
        auto l_linear = ConvolutionalLayer(_m, "convolutional_layer_test_linear",
                                         Layer::Activation::Linear,
                                         in_shape, k_shape, filters);
        EDGE_LEARNING_TEST_TRY(RneType r; l_linear.init(r));
        EDGE_LEARNING_TEST_TRY(l_linear.forward(v1.data()));
        EDGE_LEARNING_TEST_TRY(l_linear.reverse(v1.data()));
        auto l_softmax = ConvolutionalLayer(_m, "convolutional_layer_test_softmax",
                                            Layer::Activation::Softmax,
                                            in_shape, k_shape, filters);
        EDGE_LEARNING_TEST_TRY(RneType r; l_softmax.init(r));
        EDGE_LEARNING_TEST_TRY(l_softmax.forward(v1.data()));
        EDGE_LEARNING_TEST_TRY(l_softmax.reverse(v1.data()));
        auto l_tanh = ConvolutionalLayer(_m, "convolutional_layer_test_tanh",
                                         Layer::Activation::TanH,
                                         in_shape, k_shape, filters);
        EDGE_LEARNING_TEST_TRY(RneType r; l_tanh.init(r));
        EDGE_LEARNING_TEST_TRY(l_tanh.forward(v1.data()));
        EDGE_LEARNING_TEST_TRY(l_tanh.reverse(v1.data()));
        auto l_none = ConvolutionalLayer(_m, "convolutional_layer_test_none",
                                         Layer::Activation::None,
                                         in_shape, k_shape, filters);
        EDGE_LEARNING_TEST_TRY(RneType r; l_none.init(r));
        EDGE_LEARNING_TEST_TRY(l_none.forward(v1.data()));
        EDGE_LEARNING_TEST_TRY(l_none.reverse(v1.data()));

        std::vector<NumType> v3{1,2,3, 4,5,6, 7,8,9,
                                1,2,3, 4,5,6, 7,8,9,
                                1,2,3, 4,5,6, 7,8,9};
        DLMath::Shape2d stride{1,1};
        DLMath::Shape2d padding{1,1};
        auto l_complex = ConvolutionalLayer(_m, "convolutional_layer_test",
                                            Layer::Activation::ReLU,
                                            in_shape, k_shape, filters,
                                            stride, padding);
        EDGE_LEARNING_TEST_TRY(RneType r; l_complex.init(r));
        EDGE_LEARNING_TEST_TRY(l_complex.forward(v3.data()));
        EDGE_LEARNING_TEST_TRY(l_complex.reverse(v3.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l_complex.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_complex.last_input(), v3.data())
        EDGE_LEARNING_TEST_NOT_EQUAL(l_complex.last_output(), nullptr);
        EDGE_LEARNING_TEST_TRY(l_complex.print());
        EDGE_LEARNING_TEST_EQUAL(l_complex.param_count(),
                                 k_shape.size() * in_shape.channels * filters
                                 + filters);
        EDGE_LEARNING_TEST_NOT_EQUAL(l_complex.param(0), nullptr);
        EDGE_LEARNING_TEST_NOT_EQUAL(l_complex.gradient(0), nullptr);
        EDGE_LEARNING_TEST_NOT_EQUAL(
            l_complex.param(k_shape.size() * in_shape.channels * filters),
            nullptr);
        EDGE_LEARNING_TEST_NOT_EQUAL(
            l_complex.gradient(k_shape.size() * in_shape.channels * filters),
            nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_complex.name(), "convolutional_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_complex.input_size(), v3.size());
        EDGE_LEARNING_TEST_EQUAL(
            l_complex.output_size(),
            ((in_shape.height - k_shape.height + 2 * padding.height) / stride.height + 1) *
            ((in_shape.width - k_shape.width + 2 * padding.width) / stride.width + 1) *
            filters);
    }

    void test_getter()
    {
        DLMath::Shape3d in_shape{3,3,3};
        DLMath::Shape2d k_shape{2,2};
        SizeType filters = 16;
        auto l = ConvolutionalLayer(_m, "convolutional_layer_test",
                                    Layer::Activation::ReLU,
                                    in_shape, k_shape, filters);

        EDGE_LEARNING_TEST_EQUAL(l.input_shape().height, in_shape.height);
        EDGE_LEARNING_TEST_EQUAL(l.input_shape().width, in_shape.width);
        EDGE_LEARNING_TEST_EQUAL(l.input_shape().channels, in_shape.channels);

        EDGE_LEARNING_TEST_EQUAL(l.output_shape().height,
                                 in_shape.height - k_shape.height + 1);
        EDGE_LEARNING_TEST_EQUAL(l.output_shape().width,
                                 in_shape.width - k_shape.width + 1);
        EDGE_LEARNING_TEST_EQUAL(l.output_shape().channels, filters);

        EDGE_LEARNING_TEST_EQUAL(l.kernel_shape().height, k_shape.height);
        EDGE_LEARNING_TEST_EQUAL(l.kernel_shape().width, k_shape.width);

        EDGE_LEARNING_TEST_EQUAL(l.n_filters(), filters);
    }


    void test_setter()
    {
        DLMath::Shape3d in_shape{3,3,3};
        DLMath::Shape2d k_shape{2,2};
        SizeType filters = 16;
        auto l = ConvolutionalLayer(_m, "convolutional_layer_test",
                                    Layer::Activation::ReLU,
                                    in_shape, k_shape, filters);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), in_shape.size());
        DLMath::Shape3d new_in_shape{5,5,3};
        EDGE_LEARNING_TEST_CALL(l.input_size(new_in_shape));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), new_in_shape.size());
    }

    Model _m = Model("model_convolutional_layer_test");
};

int main() {
    TestConvolutionalLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
