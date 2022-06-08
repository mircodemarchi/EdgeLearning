/***************************************************************************
 *            dnn/test_avg_pooling.cpp
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
#include "dnn/avg_pooling.hpp"
#include "dnn/model.hpp"

using namespace std;
using namespace EdgeLearning;


class TestAvgPoolingLayer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_layer());
        EDGE_LEARNING_TEST_CALL(test_avg_pooling_layer());
        EDGE_LEARNING_TEST_CALL(test_getter());
        EDGE_LEARNING_TEST_CALL(test_setter());
        EDGE_LEARNING_TEST_CALL(test_stream());
    }

private:
    void test_layer() {
        EDGE_LEARNING_TEST_EQUAL(AvgPoolingLayer::TYPE, "AveragePool");
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(
                auto l = AvgPoolingLayer(_m, "avg_pooling_layer_test"));
        EDGE_LEARNING_TEST_TRY(
                auto l = AvgPoolingLayer(_m, "avg_pooling_layer_test"));
        auto l = AvgPoolingLayer(_m, "avg_pooling_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.TYPE, "AveragePool");
        EDGE_LEARNING_TEST_EQUAL(l.type(), "AveragePool");
        EDGE_LEARNING_TEST_TRY(l.init());
        EDGE_LEARNING_TEST_TRY(l.forward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.backward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l.param(0));
        EDGE_LEARNING_TEST_THROWS(l.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "avg_pooling_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());

        EDGE_LEARNING_TEST_EXECUTE(AvgPoolingLayer l1_copy{l});
        EDGE_LEARNING_TEST_TRY(AvgPoolingLayer l2_copy{l});
        AvgPoolingLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(l_copy.init());
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_copy.param(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_copy.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "avg_pooling_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output().size(),
                                 l_copy.output_size());

        EDGE_LEARNING_TEST_EXECUTE(AvgPoolingLayer l_assign(_m); l_assign = l);
        EDGE_LEARNING_TEST_TRY(AvgPoolingLayer l_assign(_m); l_assign = l);
        AvgPoolingLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_TRY(l_assign.init());
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_assign.param(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_assign.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "avg_pooling_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output().size(),
                                 l_assign.output_size());

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = AvgPoolingLayer(_m));
        EDGE_LEARNING_TEST_TRY(auto l2 = AvgPoolingLayer(_m));
        auto l_noname = AvgPoolingLayer(_m);
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());

        DLMath::Shape3d in_shape{3,3,3};
        DLMath::Shape2d k_shape{2,2};
        auto l_shape = AvgPoolingLayer(_m, "avg_pooling_layer_test",
                                       in_shape, k_shape);
        auto truth_output_size = ((in_shape.width - k_shape.width) + 1)
            * ((in_shape.height - k_shape.height) + 1) * in_shape.channels;
        EDGE_LEARNING_TEST_EQUAL(l_shape.input_size(), in_shape.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape.output_size(), truth_output_size);
        EDGE_LEARNING_TEST_ASSERT(l_shape.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape.last_output().size(),
                                 l_shape.output_size());
        AvgPoolingLayer l_shape_copy{l_shape};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), in_shape.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), truth_output_size);
        EDGE_LEARNING_TEST_ASSERT(l_shape_copy.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape_copy.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_output().size(),
                                 l_shape_copy.output_size());
        AvgPoolingLayer l_shape_assign(_m); l_shape_assign = l_shape;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), in_shape.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(),
                                 truth_output_size);
        EDGE_LEARNING_TEST_ASSERT(l_shape_assign.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape_assign.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_output().size(),
                                 l_shape_assign.output_size());
    }

    void test_avg_pooling_layer()
    {
        std::vector<NumType> v1{1,1,1, 1,1,1, 1,1,1,
                                1,1,1, 1,1,1, 1,1,1,
                                1,1,1, 1,1,1, 1,1,1};
        DLMath::Shape3d in_shape{3,3,3};
        DLMath::Shape2d k_shape{2,2};
        auto l = AvgPoolingLayer(_m, "avg_pooling_layer_test",
                                 in_shape, k_shape);
        EDGE_LEARNING_TEST_TRY(l.forward(v1));
        EDGE_LEARNING_TEST_TRY(l.backward(v1));
        EDGE_LEARNING_TEST_ASSERT(!l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v1.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_input()[0], v1[0]);
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());

        std::vector<NumType> v2{1,2,3, 4,5,6, 7,8,9,
                                1,2,3, 4,5,6, 7,8,9,
                                1,2,3, 4,5,6, 7,8,9};
        AvgPoolingLayer l_copy{l};
        EDGE_LEARNING_TEST_ASSERT(!l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v1.size());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input()[0], v1[0]);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output().size(),
                                 l_copy.output_size());
        EDGE_LEARNING_TEST_TRY(l_copy.forward(v2));
        EDGE_LEARNING_TEST_TRY(l_copy.backward(v2));
        EDGE_LEARNING_TEST_ASSERT(!l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v2.size());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input()[0], v2[0]);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output().size(),
                                 l_copy.output_size());

        AvgPoolingLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_ASSERT(!l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v1.size());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input()[0], v1[0]);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output().size(),
                                 l_assign.output_size());
        EDGE_LEARNING_TEST_TRY(l_assign.forward(v2));
        EDGE_LEARNING_TEST_TRY(l_assign.backward(v2));
        EDGE_LEARNING_TEST_ASSERT(!l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v2.size());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input()[0], v2[0]);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output().size(),
                                 l_assign.output_size());
    }

    void test_getter()
    {
        DLMath::Shape3d in_shape{3,3,3};
        DLMath::Shape2d k_shape{2,2};
        auto l = AvgPoolingLayer(_m, "avg_pooling_layer_test",
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
        auto l = AvgPoolingLayer(_m, "avg_pooling_layer_test",
                                 in_shape, k_shape);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), in_shape.size());
        DLMath::Shape3d new_in_shape{5,5,3};
        EDGE_LEARNING_TEST_CALL(l.input_shape(new_in_shape));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), new_in_shape.size());
    }

    void test_stream()
    {
        DLMath::Shape3d in_shape{3,3,3};
        DLMath::Shape2d k_shape{2,2};
        auto l = AvgPoolingLayer(_m, "avg_pooling_layer_test",
                                 in_shape, k_shape);

        Json l_dump;
        EDGE_LEARNING_TEST_TRY(l.dump(l_dump));
        EDGE_LEARNING_TEST_PRINT(l_dump);
        EDGE_LEARNING_TEST_EQUAL(l_dump["type"].as<std::string>(), "AveragePool");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l.name());

        auto input_size_arr = l_dump["input_size"].as_vec<std::size_t>();
        EDGE_LEARNING_TEST_EQUAL(input_size_arr.size(), 3);
        std::size_t input_size = input_size_arr[0]
                                 * input_size_arr[1] * input_size_arr[2];
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[0], l.input_shape().height);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[1], l.input_shape().width);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[2], l.input_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(input_size, l.input_size());

        auto output_size_arr = l_dump["output_size"].as_vec<std::size_t>();
        EDGE_LEARNING_TEST_EQUAL(output_size_arr.size(), 3);
        std::size_t output_size = output_size_arr[0]
                                  * output_size_arr[1] * output_size_arr[2];
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[0], l.output_shape().height);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[1], l.output_shape().width);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[2], l.output_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(output_size, l.output_size());

        EDGE_LEARNING_TEST_EQUAL(l_dump["antecedents"].size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_dump["subsequents"].size(), 0);

        l = AvgPoolingLayer(_m);
        EDGE_LEARNING_TEST_TRY(l.load(l_dump));
        EDGE_LEARNING_TEST_EQUAL(l.type(), "AveragePool");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l.name());
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[0], l.input_shape().height);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[1], l.input_shape().width);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[2], l.input_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(input_size, l.input_size());
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[0], l.output_shape().height);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[1], l.output_shape().width);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[2], l.output_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(output_size, l.output_size());

        Json json_void;
        EDGE_LEARNING_TEST_FAIL(l.load(json_void));
        EDGE_LEARNING_TEST_THROWS(l.load(json_void), std::runtime_error);

        EDGE_LEARNING_TEST_EQUAL(
            l_dump["others"]["kernel_size"].size(), 2);
        EDGE_LEARNING_TEST_EQUAL(
            l_dump["others"]["kernel_size"][0].as<SizeType>(),
            l.kernel_shape().height);
        EDGE_LEARNING_TEST_EQUAL(
            l_dump["others"]["kernel_size"][1].as<SizeType>(),
            l.kernel_shape().width);
        EDGE_LEARNING_TEST_EQUAL(
            l_dump["others"]["stride"].size(), 2);
        EDGE_LEARNING_TEST_EQUAL(
            l_dump["others"]["stride"][0].as<SizeType>(), 1);
        EDGE_LEARNING_TEST_EQUAL(
            l_dump["others"]["stride"][1].as<SizeType>(), 1);
    }

    Model _m = Model("model_avg_pooling_layer_test");
};

int main() {
    TestAvgPoolingLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
