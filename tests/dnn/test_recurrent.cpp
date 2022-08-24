/***************************************************************************
 *            dnn/test_recurrent.cpp
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
#include "dnn/recurrent.hpp"
#include "dnn/model.hpp"

using namespace std;
using namespace EdgeLearning;


class TestRecurrentLayer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_layer());
        EDGE_LEARNING_TEST_CALL(test_recurrent_layer());
        EDGE_LEARNING_TEST_CALL(test_getter());
        EDGE_LEARNING_TEST_CALL(test_setter());
        EDGE_LEARNING_TEST_CALL(test_stream());
    }

private:
    void test_layer() {
        EDGE_LEARNING_TEST_EQUAL(RecurrentLayer::TYPE, "Recurrent");
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(
                auto l = RecurrentLayer("recurrent_layer_test"));
        EDGE_LEARNING_TEST_TRY(
                auto l = RecurrentLayer("recurrent_layer_test"));
        auto l = RecurrentLayer("recurrent_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.TYPE, "Recurrent");
        EDGE_LEARNING_TEST_EQUAL(l.type(), "Recurrent");
        EDGE_LEARNING_TEST_TRY(
            l.init(Layer::InitializationFunction::KAIMING,
                   Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l.init(Layer::InitializationFunction::KAIMING,
                   Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l.init(Layer::InitializationFunction::XAVIER,
                   Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l.init(Layer::InitializationFunction::XAVIER,
                   Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
        EDGE_LEARNING_TEST_TRY(l.training_forward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.backward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l.param(0));
        EDGE_LEARNING_TEST_THROWS(l.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "recurrent_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());
        EDGE_LEARNING_TEST_TRY((void) l.clone());
        EDGE_LEARNING_TEST_EQUAL(l.clone()->name(), l.name());

        EDGE_LEARNING_TEST_EXECUTE(RecurrentLayer l_copy{l});
        EDGE_LEARNING_TEST_TRY(RecurrentLayer l_copy{l});
        RecurrentLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(
            l_copy.init(Layer::InitializationFunction::KAIMING,
                        Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_copy.init(Layer::InitializationFunction::KAIMING,
                        Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_copy.init(Layer::InitializationFunction::XAVIER,
                        Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_copy.init(Layer::InitializationFunction::XAVIER,
                        Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_copy.param(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_copy.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "recurrent_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output().size(),
                                 l_copy.output_size());

        EDGE_LEARNING_TEST_EXECUTE(
                RecurrentLayer l_assign; l_assign = l);
        EDGE_LEARNING_TEST_TRY(
                RecurrentLayer l_assign; l_assign = l);
        RecurrentLayer l_assign; l_assign = l;
        EDGE_LEARNING_TEST_TRY(
            l_assign.init(
                Layer::InitializationFunction::KAIMING,
                Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_assign.init(
                Layer::InitializationFunction::KAIMING,
                Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_assign.init(
                Layer::InitializationFunction::XAVIER,
                Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_assign.init(
                Layer::InitializationFunction::XAVIER,
                Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_assign.param(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_assign.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "recurrent_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output().size(),
                                 l_assign.output_size());

        auto l1_clone = l.clone();
        auto l2_clone = l.clone();
        EDGE_LEARNING_TEST_EQUAL(
            l1_clone->last_input().size(), l2_clone->last_input().size());
        EDGE_LEARNING_TEST_CALL(l1_clone->training_forward(v));
        EDGE_LEARNING_TEST_NOT_EQUAL(
            l1_clone->last_input().size(), l2_clone->last_input().size());
        EDGE_LEARNING_TEST_TRY(l.training_forward(v));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(!l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = RecurrentLayer());
        EDGE_LEARNING_TEST_TRY(auto l2 = RecurrentLayer());
        auto l_noname = RecurrentLayer();
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());
    }

    void test_recurrent_layer()
    {
        std::vector<NumType> v_empty;
        auto l_fail = RecurrentLayer("recurrent_layer_test");
        EDGE_LEARNING_TEST_FAIL(l_fail.hidden_state({0.0}));
        EDGE_LEARNING_TEST_THROWS(l_fail.hidden_state({0.0}),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_TRY(l_fail.hidden_state({}));
        auto l = RecurrentLayer("recurrent_layer_test",
                                10, 20, 5);
        EDGE_LEARNING_TEST_TRY(l.hidden_state({0.0}));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_TRY(l.time_steps(1));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 20);
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());
        EDGE_LEARNING_TEST_TRY(l.time_steps(2));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 20);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 40);
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());
        RecurrentLayer l_shape_copy{l};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), 20);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), 40);
        EDGE_LEARNING_TEST_ASSERT(l_shape_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_input().size(),
                                 v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_output().size(),
                                 l_shape_copy.output_size());
        RecurrentLayer l_shape_assign; l_shape_assign = l;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), 20);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), 40);
        EDGE_LEARNING_TEST_ASSERT(l_shape_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_input().size(),
                                 v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_output().size(),
                                 l_shape_assign.output_size());
    }

    void test_getter()
    {
        SizeType input_size = 10;
        SizeType output_size = 20;
        SizeType hidden_size = 5;
        auto l = RecurrentLayer("recurrent_layer_test",
                                input_size, output_size, hidden_size);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_TRY(l.time_steps(1));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), output_size);
        EDGE_LEARNING_TEST_TRY(l.time_steps(2));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 2*input_size);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 2*output_size);
    }

    void test_setter()
    {
        SizeType input_size = 1;
        SizeType output_size = 20;
        SizeType hidden_size = 5;
        auto l = RecurrentLayer("recurrent_layer_test",
                                input_size, output_size, hidden_size, 1);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);
        input_size = 10;
        EDGE_LEARNING_TEST_CALL(l.input_shape(input_size));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);

        EDGE_LEARNING_TEST_TRY(l.time_steps(2));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 2*input_size);

        auto l1_clone = l.clone();
        auto l2_clone = l.clone();
        EDGE_LEARNING_TEST_EQUAL(
            l1_clone->input_size(), l2_clone->input_size());
        EDGE_LEARNING_TEST_CALL(l1_clone->input_shape(20));
        EDGE_LEARNING_TEST_NOT_EQUAL(
            l1_clone->input_size(), l2_clone->input_size());

        EDGE_LEARNING_TEST_TRY(l.hidden_state({0, 1, 2, 3, 4}));
    }

    void test_stream()
    {
        SizeType hidden_size = 5;
        SizeType time_steps = 5;
        SizeType input_size = 10;
        SizeType output_size = 20;
        auto l = RecurrentLayer("recurrent_layer_test",
                                input_size, output_size, hidden_size, time_steps);

        Json l_dump;
        EDGE_LEARNING_TEST_TRY(l.dump(l_dump));
        EDGE_LEARNING_TEST_PRINT(l_dump);
        EDGE_LEARNING_TEST_EQUAL(l_dump["type"].as<std::string>(), "Recurrent");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l.name());

        for (SizeType i = 0; i < l_dump["input_shape"].size(); ++i)
        {
            auto input_size_arr = l_dump["input_shape"][i]
                .as_vec<std::size_t>();
            EDGE_LEARNING_TEST_EQUAL(input_size_arr.size(), 3);
            std::size_t i_size =
                input_size_arr[0] * input_size_arr[1] * input_size_arr[2];
            EDGE_LEARNING_TEST_EQUAL(input_size_arr[0],
                                     l.input_shape().height(i));
            EDGE_LEARNING_TEST_EQUAL(input_size_arr[1],
                                     l.input_shape().width(i));
            EDGE_LEARNING_TEST_EQUAL(input_size_arr[2],
                                     l.input_shape().channels(i));
            EDGE_LEARNING_TEST_EQUAL(i_size, l.input_shape().size(i));
        }

        for (SizeType i = 0; i < l_dump["output_shape"].size(); ++i) {
            auto output_size_arr = l_dump["output_shape"][i]
                .as_vec<std::size_t>();
            EDGE_LEARNING_TEST_EQUAL(output_size_arr.size(), 3);
            std::size_t o_size = output_size_arr[0] * output_size_arr[1] * output_size_arr[2];
            EDGE_LEARNING_TEST_EQUAL(output_size_arr[0],
                                     l.output_shape().height(i));
            EDGE_LEARNING_TEST_EQUAL(output_size_arr[1],
                                     l.output_shape().width(i));
            EDGE_LEARNING_TEST_EQUAL(output_size_arr[2],
                                     l.output_shape().channels(i));
            EDGE_LEARNING_TEST_EQUAL(o_size, l.output_shape().size(i));
        }

        l = RecurrentLayer();
        EDGE_LEARNING_TEST_TRY(l.load(l_dump));
        EDGE_LEARNING_TEST_EQUAL(l.type(), "Recurrent");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l.name());
        for (SizeType i = 0; i < l_dump["input_shape"].size(); ++i)
        {
            auto input_size_arr = l_dump["input_shape"][i]
                .as_vec<std::size_t>();
            std::size_t i_size = input_size_arr[0] * input_size_arr[1] * input_size_arr[2];
            EDGE_LEARNING_TEST_EQUAL(input_size_arr[0],
                                     l.input_shape().height(i));
            EDGE_LEARNING_TEST_EQUAL(input_size_arr[1],
                                     l.input_shape().width(i));
            EDGE_LEARNING_TEST_EQUAL(input_size_arr[2],
                                     l.input_shape().channels(i));
            EDGE_LEARNING_TEST_EQUAL(i_size, l.input_shape().size(i));
        }
        for (SizeType i = 0; i < l_dump["output_shape"].size(); ++i) {
            auto output_size_arr = l_dump["output_shape"][i]
                .as_vec<std::size_t>();
            std::size_t o_size =
                output_size_arr[0] * output_size_arr[1] * output_size_arr[2];
            EDGE_LEARNING_TEST_EQUAL(output_size_arr[0],
                                     l.output_shape().height(i));
            EDGE_LEARNING_TEST_EQUAL(output_size_arr[1],
                                     l.output_shape().width(i));
            EDGE_LEARNING_TEST_EQUAL(output_size_arr[2],
                                     l.output_shape().channels(i));
            EDGE_LEARNING_TEST_EQUAL(o_size, l.output_shape().size(i));
        }

        Json json_void;
        EDGE_LEARNING_TEST_FAIL(l.load(json_void));
        EDGE_LEARNING_TEST_THROWS(l.load(json_void), std::runtime_error);

        EDGE_LEARNING_TEST_EQUAL(l_dump["weights"].size(), 3);
        EDGE_LEARNING_TEST_EQUAL(l_dump["weights"][0].size(), hidden_size);
        EDGE_LEARNING_TEST_EQUAL(l_dump["weights"][0][0].size(),
                                 input_size);
        EDGE_LEARNING_TEST_EQUAL(l_dump["weights"][0][hidden_size - 1].size(),
                                 input_size);
        EDGE_LEARNING_TEST_EQUAL(l_dump["weights"][1].size(), hidden_size);
        EDGE_LEARNING_TEST_EQUAL(l_dump["weights"][1][0].size(),
                                 hidden_size);
        EDGE_LEARNING_TEST_EQUAL(l_dump["weights"][1][hidden_size - 1].size(),
                                 hidden_size);
        EDGE_LEARNING_TEST_EQUAL(l_dump["weights"][2].size(), output_size);
        EDGE_LEARNING_TEST_EQUAL(l_dump["weights"][2][0].size(),
                                 hidden_size);
        EDGE_LEARNING_TEST_EQUAL(l_dump["weights"][2][output_size - 1].size(),
                                 hidden_size);
        EDGE_LEARNING_TEST_EQUAL(l_dump["biases"].size(), 2);
        EDGE_LEARNING_TEST_EQUAL(l_dump["biases"][0].size(), hidden_size);
        EDGE_LEARNING_TEST_EQUAL(l_dump["biases"][1].size(), output_size);
        EDGE_LEARNING_TEST_EQUAL(
            l_dump["others"]["hidden_activation"].as<int>(),
            static_cast<int>(RecurrentLayer::HiddenActivation::TanH));
        EDGE_LEARNING_TEST_EQUAL(l_dump["others"]["hidden_size"].as<SizeType>(),
                                 hidden_size);
        EDGE_LEARNING_TEST_EQUAL(l_dump["others"]["time_steps"].as<SizeType>(),
                                 time_steps);
    }
};

int main() {
    TestRecurrentLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
