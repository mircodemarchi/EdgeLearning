/***************************************************************************
 *            dnn/test_max_pooling.cpp
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
#include "dnn/max_pooling.hpp"
#include "dnn/model.hpp"

using namespace std;
using namespace EdgeLearning;


class TestMaxPoolingLayer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_layer());
        EDGE_LEARNING_TEST_CALL(test_max_pooling_layer());
        EDGE_LEARNING_TEST_CALL(test_getter());
        EDGE_LEARNING_TEST_CALL(test_setter());
    }

private:
    void test_layer() {
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(
                auto l = MaxPoolingLayer(_m, "max_pooling_layer_test"));
        EDGE_LEARNING_TEST_TRY(
                auto l = MaxPoolingLayer(_m, "max_pooling_layer_test"));
        auto l = MaxPoolingLayer(_m, "max_pooling_layer_test");
        EDGE_LEARNING_TEST_TRY(l.init());
        // TODO: Manage forward with nullptr input.
        // EDGE_LEARNING_TEST_TRY(l.forward(nullptr));
        // EDGE_LEARNING_TEST_TRY(l.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "max_pooling_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(MaxPoolingLayer l1_copy{l});
        EDGE_LEARNING_TEST_TRY(MaxPoolingLayer l2_copy{l});
        MaxPoolingLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(l_copy.init());
        // TODO: Manage forward with nullptr input.
        // EDGE_LEARNING_TEST_TRY(l_copy.forward(nullptr));
        // EDGE_LEARNING_TEST_TRY(l_copy.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "max_pooling_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(MaxPoolingLayer l_assign(_m); l_assign = l);
        EDGE_LEARNING_TEST_TRY(MaxPoolingLayer l_assign(_m); l_assign = l);
        MaxPoolingLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_TRY(l_assign.init());
        // TODO: Manage forward with nullptr input.
        // EDGE_LEARNING_TEST_TRY(l_assign.forward(nullptr));
        // EDGE_LEARNING_TEST_TRY(l_assign.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "max_pooling_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = MaxPoolingLayer(_m));
        EDGE_LEARNING_TEST_TRY(auto l2 = MaxPoolingLayer(_m));
        auto l_noname = MaxPoolingLayer(_m);
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());

        DLMath::Shape3d in_shape{3,3,3};
        DLMath::Shape2d k_shape{2,2};
        auto l_shape = MaxPoolingLayer(_m, "max_pooling_layer_test",
                                       Layer::Activation::ReLU,
                                       in_shape, k_shape);
        auto truth_output_size = ((in_shape.width - k_shape.width) + 1)
            * ((in_shape.height - k_shape.height) + 1) * in_shape.channels;
        EDGE_LEARNING_TEST_EQUAL(l_shape.input_size(), in_shape.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape.output_size(), truth_output_size);
        EDGE_LEARNING_TEST_EQUAL(l_shape.last_input(), nullptr);
        EDGE_LEARNING_TEST_NOT_EQUAL(l_shape.last_output(), nullptr);
        MaxPoolingLayer l_shape_copy{l_shape};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), in_shape.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), truth_output_size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_NOT_EQUAL(l_shape_copy.last_output(), nullptr);
        MaxPoolingLayer l_shape_assign(_m); l_shape_assign = l_shape;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), in_shape.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(),
                                 truth_output_size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_NOT_EQUAL(l_shape_assign.last_output(), nullptr);
    }

    void test_max_pooling_layer()
    {
        std::vector<NumType> v1{1,1,1, 1,1,1, 1,1,1,
                                1,1,1, 1,1,1, 1,1,1,
                                1,1,1, 1,1,1, 1,1,1};
        DLMath::Shape3d in_shape{3,3,3};
        DLMath::Shape2d k_shape{2,2};
        auto l = MaxPoolingLayer(_m, "max_pooling_layer_test",
                                 Layer::Activation::ReLU,
                                 in_shape, k_shape);
        EDGE_LEARNING_TEST_TRY(l.forward(v1.data()));
        EDGE_LEARNING_TEST_TRY(l.reverse(v1.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.last_input(), v1.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l.last_output(), nullptr);

        std::vector<NumType> v2{1,2,3, 4,5,6, 7,8,9,
                                1,2,3, 4,5,6, 7,8,9,
                                1,2,3, 4,5,6, 7,8,9};
        MaxPoolingLayer l_copy{l};
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), v1.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_output(), nullptr);
        EDGE_LEARNING_TEST_TRY(l_copy.forward(v2.data()));
        EDGE_LEARNING_TEST_TRY(l_copy.reverse(v2.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), v2.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_output(), nullptr);

        MaxPoolingLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), v1.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_output(), nullptr);
        EDGE_LEARNING_TEST_TRY(l_assign.forward(v2.data()));
        EDGE_LEARNING_TEST_TRY(l_assign.reverse(v2.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), v2.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_output(), nullptr);
    }

    void test_getter()
    {
        DLMath::Shape3d in_shape{3,3,3};
        DLMath::Shape2d k_shape{2,2};
        auto l = MaxPoolingLayer(_m, "max_pooling_layer_test",
                                 Layer::Activation::ReLU,
                                 in_shape, k_shape);

        EDGE_LEARNING_TEST_EQUAL(l.input_shape().height, in_shape.height);
        EDGE_LEARNING_TEST_EQUAL(l.input_shape().width, in_shape.width);
        EDGE_LEARNING_TEST_EQUAL(l.input_shape().channels, in_shape.channels);

        EDGE_LEARNING_TEST_EQUAL(l.output_shape().height,
                                 in_shape.height - k_shape.height + 1);
        EDGE_LEARNING_TEST_EQUAL(l.output_shape().width,
                                 in_shape.width - k_shape.width + 1);

        EDGE_LEARNING_TEST_EQUAL(l.kernel_shape().height, k_shape.height);
        EDGE_LEARNING_TEST_EQUAL(l.kernel_shape().width, k_shape.width);
    }

    void test_setter()
    {
        DLMath::Shape3d in_shape{3,3,3};
        DLMath::Shape2d k_shape{2,2};
        auto l = MaxPoolingLayer(_m, "max_pooling_layer_test",
                                 Layer::Activation::ReLU,
                                 in_shape, k_shape);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), in_shape.size());
        DLMath::Shape3d new_in_shape{5,5,3};
        EDGE_LEARNING_TEST_CALL(l.input_size(new_in_shape));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), new_in_shape.size());
    }

    Model _m = Model("model_max_pooling_layer_test");
};

int main() {
    TestMaxPoolingLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
