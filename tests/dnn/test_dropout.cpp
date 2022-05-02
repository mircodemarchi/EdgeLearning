/***************************************************************************
 *            dnn/test_dropout.cpp
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
#include "dnn/dropout.hpp"
#include "dnn/model.hpp"

using namespace std;
using namespace EdgeLearning;


class TestDropoutLayer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_layer());
        EDGE_LEARNING_TEST_CALL(test_dropout_layer());
        EDGE_LEARNING_TEST_CALL(test_getter());
        EDGE_LEARNING_TEST_CALL(test_setter());
    }

private:
    void test_layer() {
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(
                auto l = DropoutLayer(_m, "dropout_layer_test"));
        EDGE_LEARNING_TEST_TRY(
                auto l = DropoutLayer(_m, "dropout_layer_test"));
        auto l = DropoutLayer(_m, "dropout_layer_test");
        EDGE_LEARNING_TEST_TRY(
            l.init(Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l.init(Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
        // TODO: Manage forward with nullptr input.
        // EDGE_LEARNING_TEST_TRY(l.forward(nullptr));
        // EDGE_LEARNING_TEST_TRY(l.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "dropout_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(DropoutLayer l1_copy{l});
        EDGE_LEARNING_TEST_TRY(DropoutLayer l2_copy{l});
        DropoutLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(
            l_copy.init(Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_copy.init(Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
        // TODO: Manage forward with nullptr input.
        // EDGE_LEARNING_TEST_TRY(l_copy.forward(nullptr));
        // EDGE_LEARNING_TEST_TRY(l_copy.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "dropout_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(DropoutLayer l_assign(_m); l_assign = l);
        EDGE_LEARNING_TEST_TRY(DropoutLayer l_assign(_m); l_assign = l);
        DropoutLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_TRY(
            l_assign.init(Layer::ProbabilityDensityFunction::NORMAL,
                          RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_assign.init(Layer::ProbabilityDensityFunction::UNIFORM,
                          RneType()));
        // TODO: Manage forward with nullptr input.
        // EDGE_LEARNING_TEST_TRY(l_assign.forward(nullptr));
        // EDGE_LEARNING_TEST_TRY(l_assign.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "dropout_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = DropoutLayer(_m));
        EDGE_LEARNING_TEST_TRY(auto l2 = DropoutLayer(_m));
        auto l_noname = DropoutLayer(_m);
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());

        auto l_shape = DropoutLayer(_m, "dropout_layer_test",
                                    Layer::Activation::ReLU, 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape.output_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape.last_input(), nullptr);
        EDGE_LEARNING_TEST_NOT_EQUAL(l_shape.last_output(), nullptr);
        DropoutLayer l_shape_copy{l_shape};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_NOT_EQUAL(l_shape_copy.last_output(), nullptr);
        DropoutLayer l_shape_assign(_m); l_shape_assign = l_shape;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_NOT_EQUAL(l_shape_assign.last_output(), nullptr);
    }

    void test_dropout_layer()
    {
        std::vector<NumType> v1{1};
        auto l = DropoutLayer(_m, "dropout_layer_test",
                              Layer::Activation::ReLU, 1);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), l.input_size());
        EDGE_LEARNING_TEST_TRY(l.forward(v1.data()));
        EDGE_LEARNING_TEST_TRY(l.reverse(v1.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.last_input(), v1.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l.last_output(), nullptr);

        std::vector<NumType> v2{2};
        DropoutLayer l_copy{l};
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), l_copy.input_size());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), v1.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_output(), nullptr);
        EDGE_LEARNING_TEST_TRY(l_copy.forward(v2.data()));
        EDGE_LEARNING_TEST_TRY(l_copy.reverse(v2.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), v2.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_output(), nullptr);

        DropoutLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), l_assign.input_size());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), v1.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_output(), nullptr);
        EDGE_LEARNING_TEST_TRY(l_assign.forward(v2.data()));
        EDGE_LEARNING_TEST_TRY(l_assign.reverse(v2.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), v2.data());
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_output(), nullptr);

        NumType p = 0.9;
        std::vector<NumType> v_complex{0,1,2,3,4,5,6,7,8,9};
        auto l_complex = DropoutLayer(_m, "dropout_layer_test",
                                      Layer::Activation::Linear,
                                      v_complex.size(), p);
        EDGE_LEARNING_TEST_TRY(l_complex.forward(v_complex.data()));
        EDGE_LEARNING_TEST_TRY(l_complex.reverse(v_complex.data()));
        p = 1.0;
        l_complex = DropoutLayer(_m, "dropout_layer_test",
                                 Layer::Activation::ReLU,
                                 v_complex.size(), p);
        EDGE_LEARNING_TEST_TRY(l_complex.forward(v_complex.data()));
        EDGE_LEARNING_TEST_TRY(l_complex.reverse(v_complex.data()));
        p = 0.4;
        RneType rne(0);
        l_complex = DropoutLayer(_m, "dropout_layer_test",
                                 Layer::Activation::ReLU,
                                 v_complex.size(), p, rne);
        EDGE_LEARNING_TEST_TRY(l_complex.forward(v_complex.data()));
        EDGE_LEARNING_TEST_TRY(l_complex.reverse(v_complex.data()));
    }

    void test_getter()
    {
        SizeType size = 1;
        auto l = DropoutLayer(_m, "dropout_layer_test",
                              Layer::Activation::ReLU, size);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), size);
    }

    void test_setter()
    {
        SizeType input_size = 1;
        auto l = DropoutLayer(_m, "dropout_layer_test",
                              Layer::Activation::ReLU, input_size, 1);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), input_size);
        input_size = 10;
        EDGE_LEARNING_TEST_CALL(l.input_size(input_size));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), input_size);
    }
    
    Model _m = Model("model_dropout_layer_test");
};

int main() {
    TestDropoutLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
