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
    }

private:
    void test_layer() {
        std::vector<NumType> v_empty;
        EDGE_LEARNING_TEST_EXECUTE(
                auto l = RecurrentLayer(_m, "recurrent_layer_test"));
        EDGE_LEARNING_TEST_TRY(
                auto l = RecurrentLayer(_m, "recurrent_layer_test"));
        auto l = RecurrentLayer(_m, "recurrent_layer_test");
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
        EDGE_LEARNING_TEST_TRY(l.forward(v_empty));
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
                RecurrentLayer l_assign(_m); l_assign = l);
        EDGE_LEARNING_TEST_TRY(
                RecurrentLayer l_assign(_m); l_assign = l);
        RecurrentLayer l_assign(_m); l_assign = l;
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

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = RecurrentLayer(_m));
        EDGE_LEARNING_TEST_TRY(auto l2 = RecurrentLayer(_m));
        auto l_noname = RecurrentLayer(_m);
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());
    }

    void test_recurrent_layer()
    {
        std::vector<NumType> v_empty;
        auto l_fail = RecurrentLayer(_m, "recurrent_layer_test");
        EDGE_LEARNING_TEST_FAIL(l_fail.hidden_state({0.0}));
        EDGE_LEARNING_TEST_THROWS(l_fail.hidden_state({0.0}),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_TRY(l_fail.hidden_state({}));
        auto l = RecurrentLayer(_m, "recurrent_layer_test",
                                10, 20, 5);
        EDGE_LEARNING_TEST_TRY(l.hidden_state({0.0}));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 20);
        EDGE_LEARNING_TEST_ASSERT(l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_TRY(l.time_steps(1));
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());
        EDGE_LEARNING_TEST_TRY(l.time_steps(2));
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), 2*l.output_size());
        RecurrentLayer l_shape_copy{l};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), 20);
        EDGE_LEARNING_TEST_ASSERT(l_shape_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_input().size(),
                                 v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_output().size(),
                                 2*l_shape_copy.output_size());
        RecurrentLayer l_shape_assign(_m); l_shape_assign = l;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), 20);
        EDGE_LEARNING_TEST_ASSERT(l_shape_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_input().size(),
                                 v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_output().size(),
                                 2*l_shape_assign.output_size());
    }

    void test_getter()
    {
        SizeType input_size = 10;
        SizeType output_size = 20;
        SizeType hidden_size = 5;
        auto l = RecurrentLayer(_m, "recurrent_layer_test",
                                input_size, output_size, hidden_size);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), output_size);
    }

    void test_setter()
    {
        SizeType input_size = 1;
        SizeType output_size = 20;
        SizeType hidden_size = 5;
        auto l = RecurrentLayer(_m, "recurrent_layer_test",
                                input_size, output_size, hidden_size);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);
        input_size = 10;
        EDGE_LEARNING_TEST_CALL(l.input_size(input_size));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);

        EDGE_LEARNING_TEST_TRY(l.hidden_state({0, 1, 2, 3, 4}));
    }

    Model _m = Model("model_recurrent_layer_test");
};

int main() {
    TestRecurrentLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
