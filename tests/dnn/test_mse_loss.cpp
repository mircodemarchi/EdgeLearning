/***************************************************************************
 *            dnn/testmse_loss.cpp
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
#include "dnn/mse_loss.hpp"
#include "dnn/model.hpp"

using namespace std;
using namespace EdgeLearning;


class TestMSELossLayer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_layer());
        EDGE_LEARNING_TEST_CALL(test_loss_layer());
        EDGE_LEARNING_TEST_CALL(test_score());
        EDGE_LEARNING_TEST_CALL(testse_loss_layer());
        EDGE_LEARNING_TEST_CALL(test_stream());
    }

private:
    void test_layer() {
        EDGE_LEARNING_TEST_EQUAL(MeanSquaredLossLayer::TYPE, "MSELoss");
        EDGE_LEARNING_TEST_EXECUTE(
                auto l1 = MeanSquaredLossLayer("mse_loss_layer_test"));
        EDGE_LEARNING_TEST_TRY(
                auto l2 = MeanSquaredLossLayer("mse_loss_layer_test"));
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        auto l = MeanSquaredLossLayer("mse_loss_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.TYPE, "MSELoss");
        EDGE_LEARNING_TEST_EQUAL(l.type(), "MSELoss");
        EDGE_LEARNING_TEST_TRY(l.init());
        EDGE_LEARNING_TEST_FAIL(l.training_forward(v_empty));
        EDGE_LEARNING_TEST_THROWS(l.training_forward(v_empty),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_TRY(l.backward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l.param(0));
        EDGE_LEARNING_TEST_THROWS(l.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l.param(10));
        EDGE_LEARNING_TEST_THROWS(l.param(10), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l.gradient(10));
        EDGE_LEARNING_TEST_THROWS(l.gradient(10), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "mse_loss_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l.last_output());
        EDGE_LEARNING_TEST_THROWS(l.last_output(), std::runtime_error);
        EDGE_LEARNING_TEST_TRY((void) l.clone());
        EDGE_LEARNING_TEST_EQUAL(l.clone()->name(), l.name());

        EDGE_LEARNING_TEST_EXECUTE(MeanSquaredLossLayer l1_copy{l});
        EDGE_LEARNING_TEST_TRY(MeanSquaredLossLayer l2_copy{l});
        MeanSquaredLossLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(l_copy.init());
        EDGE_LEARNING_TEST_FAIL(l_copy.training_forward(v_empty));
        EDGE_LEARNING_TEST_THROWS(l_copy.training_forward(v_empty),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_TRY(l_copy.backward(v_empty));
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_copy.param(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_copy.param(10));
        EDGE_LEARNING_TEST_THROWS(l_copy.param(10), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_copy.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_copy.gradient(10));
        EDGE_LEARNING_TEST_THROWS(l_copy.gradient(10), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "mse_loss_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_copy.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l_copy.last_output());
        EDGE_LEARNING_TEST_THROWS(l_copy.last_output(), std::runtime_error);

        EDGE_LEARNING_TEST_EXECUTE(MeanSquaredLossLayer l_assign; l_assign = l);
        EDGE_LEARNING_TEST_TRY(MeanSquaredLossLayer l_assign; l_assign = l);
        MeanSquaredLossLayer l_assign; l_assign = l;
        EDGE_LEARNING_TEST_TRY(l_assign.init());
        EDGE_LEARNING_TEST_FAIL(l_assign.training_forward(v_empty));
        EDGE_LEARNING_TEST_THROWS(
            l_assign.training_forward(v_empty), std::runtime_error);
        EDGE_LEARNING_TEST_TRY(l_assign.backward(v_empty));
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_assign.param(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_assign.param(10));
        EDGE_LEARNING_TEST_THROWS(l_assign.param(10), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_assign.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_assign.gradient(10));
        EDGE_LEARNING_TEST_THROWS(l_assign.gradient(10), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "mse_loss_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_assign.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l_assign.last_output());
        EDGE_LEARNING_TEST_THROWS(l_assign.last_output(), std::runtime_error);

        EDGE_LEARNING_TEST_EXECUTE(auto l1_noname = MeanSquaredLossLayer());
        EDGE_LEARNING_TEST_TRY(auto l2_noname = MeanSquaredLossLayer());
        auto l_noname = MeanSquaredLossLayer();
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());
    }

    void test_loss_layer() {
        const std::size_t input_size = 6;
        const std::size_t batch_size = 2;
        std::vector<NumType> v(input_size);
        EDGE_LEARNING_TEST_EXECUTE(
                auto l1 = MeanSquaredLossLayer("mse_loss_layer_test",
                                       0, 0));
        EDGE_LEARNING_TEST_TRY(
                auto l2 = MeanSquaredLossLayer("mse_loss_layer_test",
                                       0, 0));
        auto l = MeanSquaredLossLayer(
                              "mse_loss_layer_test",
                              input_size, batch_size);
        EDGE_LEARNING_TEST_TRY(l.init());
        EDGE_LEARNING_TEST_EXECUTE(l.print());
        std::vector<NumType> target{0,1};
        EDGE_LEARNING_TEST_EXECUTE(l.set_target(target));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);

        auto l1_clone = l.clone();
        auto l2_clone = l.clone();
        EDGE_LEARNING_TEST_EQUAL(
            l1_clone->last_input().size(), l2_clone->last_input().size());
        EDGE_LEARNING_TEST_CALL(l1_clone->training_forward(v));
        EDGE_LEARNING_TEST_NOT_EQUAL(
            l1_clone->last_input().size(), l2_clone->last_input().size());
        EDGE_LEARNING_TEST_TRY(l.training_forward(v));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), v.size());
        EDGE_LEARNING_TEST_ASSERT(!l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v.size());

        MeanSquaredLossLayer l_shape_copy{l};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), input_size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(!l_shape_copy.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l_shape_copy.last_output());
        EDGE_LEARNING_TEST_THROWS(l_shape_copy.last_output(),
                                  std::runtime_error);
        MeanSquaredLossLayer l_shape_assign; l_shape_assign = l;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), input_size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(!l_shape_assign.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l_shape_assign.last_output());
        EDGE_LEARNING_TEST_THROWS(l_shape_assign.last_output(),
                                  std::runtime_error);
    }

    void test_score() {
        auto l = MeanSquaredLossLayer("mse_loss_layer_test", 1);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_EXECUTE(l.reset_score());
        EDGE_LEARNING_TEST_EXECUTE(l.print());
        std::vector<NumType> v{0};
        std::vector<NumType> t2{1};
        EDGE_LEARNING_TEST_EXECUTE(l.set_target(t2));
        for (SizeType i = 0; i < 10; ++i)
        {
            l.training_forward(v);
        }
        EDGE_LEARNING_TEST_ASSERT(!l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_input()[0], v[0]);
        EDGE_LEARNING_TEST_FAIL(l.last_output());
        EDGE_LEARNING_TEST_THROWS(l.last_output(), std::runtime_error);
        EDGE_LEARNING_TEST_EXECUTE(l.print());
        EDGE_LEARNING_TEST_PRINT(l.accuracy());
        EDGE_LEARNING_TEST_PRINT(l.avg_loss());
        EDGE_LEARNING_TEST_EXECUTE(l.reset_score());
        EDGE_LEARNING_TEST_ASSERT(l.accuracy() != l.accuracy());
        EDGE_LEARNING_TEST_ASSERT(l.avg_loss() != l.avg_loss());

        MeanSquaredLossLayer l_shape_copy{l};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(!l_shape_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_input().size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_input()[0], v[0]);
        EDGE_LEARNING_TEST_FAIL(l_shape_copy.last_output());
        EDGE_LEARNING_TEST_THROWS(l_shape_copy.last_output(),
                                  std::runtime_error);
        MeanSquaredLossLayer l_shape_assign; l_shape_assign = l;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(!l_shape_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_input().size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_input()[0], v[0]);
        EDGE_LEARNING_TEST_FAIL(l_shape_assign.last_output());
        EDGE_LEARNING_TEST_THROWS(l_shape_assign.last_output(),
                                  std::runtime_error);
    }

    void testse_loss_layer() {
        SizeType input_size = 1;
        SizeType batch_size = 1;
        auto l = MeanSquaredLossLayer("mse_loss_layer_test",
                                      input_size, batch_size);
        std::vector<NumType> v1{0};
        std::vector<NumType> target{1};
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_FAIL(l.forward(v1));
        EDGE_LEARNING_TEST_THROWS(l.forward(v1), std::runtime_error);
        EDGE_LEARNING_TEST_ASSERT(l.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l.last_output());
        EDGE_LEARNING_TEST_THROWS(l.last_output(), std::runtime_error);

        EDGE_LEARNING_TEST_TRY(l.set_target(target));
        EDGE_LEARNING_TEST_TRY(l.training_forward(v1));
        EDGE_LEARNING_TEST_TRY(l.backward(v1));
        EDGE_LEARNING_TEST_ASSERT(!l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v1.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_input()[0], v1[0]);
        EDGE_LEARNING_TEST_FAIL(l.last_output());
        EDGE_LEARNING_TEST_THROWS(l.last_output(), std::runtime_error);

        std::vector<NumType> v2{10};
        MeanSquaredLossLayer l_copy{l};
        EDGE_LEARNING_TEST_ASSERT(!l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v1.size());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input()[0], v1[0]);
        EDGE_LEARNING_TEST_FAIL(l_copy.last_output());
        EDGE_LEARNING_TEST_THROWS(l_copy.last_output(), std::runtime_error);
        EDGE_LEARNING_TEST_TRY(l_copy.training_forward(v2));
        EDGE_LEARNING_TEST_TRY(l_copy.backward(v2));
        EDGE_LEARNING_TEST_ASSERT(!l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v2.size());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input()[0], v2[0]);
        EDGE_LEARNING_TEST_FAIL(l_copy.last_output());
        EDGE_LEARNING_TEST_THROWS(l_copy.last_output(), std::runtime_error);

        MeanSquaredLossLayer l_assign; l_assign = l;
        EDGE_LEARNING_TEST_ASSERT(!l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v1.size());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input()[0], v1[0]);
        EDGE_LEARNING_TEST_FAIL(l_assign.last_output());
        EDGE_LEARNING_TEST_THROWS(l_assign.last_output(), std::runtime_error);
        EDGE_LEARNING_TEST_TRY(l_assign.training_forward(v2));
        EDGE_LEARNING_TEST_TRY(l_assign.backward(v2));
        EDGE_LEARNING_TEST_ASSERT(!l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v2.size());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input()[0], v2[0]);
        EDGE_LEARNING_TEST_FAIL(l_assign.last_output());
        EDGE_LEARNING_TEST_THROWS(l_assign.last_output(), std::runtime_error);

        input_size = 2;
        NumType e = 0.1;
        auto l_regression = MeanSquaredLossLayer("mse_loss_layer_test",
                                                 input_size, batch_size, e);
        std::vector<NumType> v3{10.2, 10.3};
        std::vector<NumType> target_right{10.21, 10.26};
        std::vector<NumType> target_wrong{10.0, 10.0};
        EDGE_LEARNING_TEST_TRY(l_regression.set_target(target_right));
        EDGE_LEARNING_TEST_TRY(l_regression.training_forward(v3));
        EDGE_LEARNING_TEST_TRY(l_regression.backward(v3));
        EDGE_LEARNING_TEST_TRY(l_regression.set_target(target_wrong));
        EDGE_LEARNING_TEST_TRY(l_regression.training_forward(v3));
        EDGE_LEARNING_TEST_TRY(l_regression.backward(v3));
    }

    void test_stream()
    {
        auto l = MeanSquaredLossLayer("mse_loss_layer_test", 2, 1, 0.1);

        Json l_dump;
        EDGE_LEARNING_TEST_TRY(l_dump = l.dump());
        EDGE_LEARNING_TEST_PRINT(l_dump);
        EDGE_LEARNING_TEST_EQUAL(l_dump["type"].as<std::string>(), "MSELoss");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l.name());

        for (SizeType i = 0; i < l_dump["input_shape"].size(); ++i)
        {
            auto input_size_arr = l_dump["input_shape"][i]
                .as_vec<std::size_t>();
            EDGE_LEARNING_TEST_EQUAL(input_size_arr.size(), 3);
            std::size_t input_size = input_size_arr[0]
                                     * input_size_arr[1] * input_size_arr[2];
            EDGE_LEARNING_TEST_EQUAL(input_size_arr[0],
                                     l.input_shape().height(i));
            EDGE_LEARNING_TEST_EQUAL(input_size_arr[1],
                                     l.input_shape().width(i));
            EDGE_LEARNING_TEST_EQUAL(input_size_arr[2],
                                     l.input_shape().channels(i));
            EDGE_LEARNING_TEST_EQUAL(input_size, l.input_shape().size(i));
        }

        for (SizeType i = 0; i < l_dump["output_shape"].size(); ++i) {
            auto output_size_arr = l_dump["output_shape"][i]
                .as_vec<std::size_t>();
            EDGE_LEARNING_TEST_EQUAL(output_size_arr.size(), 3);
            std::size_t output_size = output_size_arr[0]
                                      * output_size_arr[1] * output_size_arr[2];
            EDGE_LEARNING_TEST_EQUAL(output_size_arr[0],
                                     l.output_shape().height(i));
            EDGE_LEARNING_TEST_EQUAL(output_size_arr[1],
                                     l.output_shape().width(i));
            EDGE_LEARNING_TEST_EQUAL(output_size_arr[2],
                                     l.output_shape().channels(i));
            EDGE_LEARNING_TEST_EQUAL(output_size, l.output_shape().size(i));
        }

        l = MeanSquaredLossLayer();
        EDGE_LEARNING_TEST_TRY(l.load(l_dump));
        EDGE_LEARNING_TEST_EQUAL(l.type(), "MSELoss");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l.name());
        for (SizeType i = 0; i < l_dump["input_shape"].size(); ++i)
        {
            auto input_size_arr = l_dump["input_shape"][i]
                .as_vec<std::size_t>();
            std::size_t input_size = input_size_arr[0]
                                     * input_size_arr[1] * input_size_arr[2];
            EDGE_LEARNING_TEST_EQUAL(input_size_arr[0],
                                     l.input_shape().height(i));
            EDGE_LEARNING_TEST_EQUAL(input_size_arr[1],
                                     l.input_shape().width(i));
            EDGE_LEARNING_TEST_EQUAL(input_size_arr[2],
                                     l.input_shape().channels(i));
            EDGE_LEARNING_TEST_EQUAL(input_size, l.input_shape().size(i));
        }
        for (SizeType i = 0; i < l_dump["output_shape"].size(); ++i) {
            auto output_size_arr = l_dump["output_shape"][i]
                .as_vec<std::size_t>();
            std::size_t output_size = output_size_arr[0]
                                      * output_size_arr[1] * output_size_arr[2];
            EDGE_LEARNING_TEST_EQUAL(output_size_arr[0],
                                     l.output_shape().height(i));
            EDGE_LEARNING_TEST_EQUAL(output_size_arr[1],
                                     l.output_shape().width(i));
            EDGE_LEARNING_TEST_EQUAL(output_size_arr[2],
                                     l.output_shape().channels(i));
            EDGE_LEARNING_TEST_EQUAL(output_size, l.output_shape().size(i));
        }

        Json json_void;
        EDGE_LEARNING_TEST_FAIL(l.load(json_void));
        EDGE_LEARNING_TEST_THROWS(l.load(json_void), std::runtime_error);

        EDGE_LEARNING_TEST_WITHIN(
            l_dump["others"]["loss_tolerance"].as<NumType>(),
            NumType(0.1), 0.00000001);
    }
};

int main() {
    TestMSELossLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
