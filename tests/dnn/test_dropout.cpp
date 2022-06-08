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
        EDGE_LEARNING_TEST_CALL(test_stream());
    }

private:
    void test_layer() {
        EDGE_LEARNING_TEST_EQUAL(DropoutLayer::TYPE, "Dropout");
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        std::vector<NumType> v_diff_size(std::size_t(11));
        EDGE_LEARNING_TEST_EXECUTE(
                auto l = DropoutLayer(_m, "dropout_layer_test"));
        EDGE_LEARNING_TEST_TRY(
                auto l = DropoutLayer(_m, "dropout_layer_test"));
        auto l = DropoutLayer(_m, "dropout_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.TYPE, "Dropout");
        EDGE_LEARNING_TEST_EQUAL(l.type(), "Dropout");
        EDGE_LEARNING_TEST_TRY(l.init());
        EDGE_LEARNING_TEST_TRY(l.training_forward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.forward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.backward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l.param(0));
        EDGE_LEARNING_TEST_THROWS(l.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "dropout_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());
        EDGE_LEARNING_TEST_TRY(l.training_forward(v));
        EDGE_LEARNING_TEST_ASSERT(!l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v.size());
        EDGE_LEARNING_TEST_FAIL(l.training_forward(v_diff_size));
        EDGE_LEARNING_TEST_THROWS(l.training_forward(v_diff_size),
                                  std::runtime_error);

        EDGE_LEARNING_TEST_EXECUTE(DropoutLayer l1_copy{l});
        EDGE_LEARNING_TEST_TRY(DropoutLayer l2_copy{l});
        DropoutLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(l_copy.init());
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_copy.param(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_copy.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "dropout_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), v.size());
        EDGE_LEARNING_TEST_ASSERT(!l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v.size());
        EDGE_LEARNING_TEST_FAIL(l_copy.training_forward(v_diff_size));
        EDGE_LEARNING_TEST_THROWS(l_copy.training_forward(v_diff_size),
                                  std::runtime_error);

        EDGE_LEARNING_TEST_EXECUTE(DropoutLayer l_assign(_m); l_assign = l);
        EDGE_LEARNING_TEST_TRY(DropoutLayer l_assign(_m); l_assign = l);
        DropoutLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_TRY(l_assign.init());
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_assign.param(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_assign.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "dropout_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), v.size());
        EDGE_LEARNING_TEST_ASSERT(!l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v.size());
        EDGE_LEARNING_TEST_FAIL(l_assign.training_forward(v_diff_size));
        EDGE_LEARNING_TEST_THROWS(l_assign.training_forward(v_diff_size),
                                  std::runtime_error);

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = DropoutLayer(_m));
        EDGE_LEARNING_TEST_TRY(auto l2 = DropoutLayer(_m));
        auto l_noname = DropoutLayer(_m);
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());

        auto l_shape = DropoutLayer(_m, "dropout_layer_test", 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape.output_size(), 10);
        EDGE_LEARNING_TEST_ASSERT(l_shape.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape.last_output().size(),
                                 l_shape.output_size());
        EDGE_LEARNING_TEST_FAIL(l_shape.training_forward(v_diff_size));
        EDGE_LEARNING_TEST_THROWS(l_shape.training_forward(v_diff_size),
                                  std::runtime_error);
        DropoutLayer l_shape_copy{l_shape};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), 10);
        EDGE_LEARNING_TEST_ASSERT(l_shape_copy.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape_copy.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_output().size(),
                                 l_shape_copy.output_size());
        EDGE_LEARNING_TEST_FAIL(l_shape_copy.training_forward(v_diff_size));
        EDGE_LEARNING_TEST_THROWS(l_shape_copy.training_forward(v_diff_size),
                                  std::runtime_error);
        DropoutLayer l_shape_assign(_m); l_shape_assign = l_shape;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), 10);
        EDGE_LEARNING_TEST_ASSERT(l_shape_assign.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape_assign.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_output().size(),
                                 l_shape_assign.output_size());
        EDGE_LEARNING_TEST_FAIL(l_shape_assign.training_forward(v_diff_size));
        EDGE_LEARNING_TEST_THROWS(l_shape_assign.training_forward(v_diff_size),
                                  std::runtime_error);
    }

    void test_dropout_layer()
    {
        std::vector<NumType> v1{1};
        auto l = DropoutLayer(_m, "dropout_layer_test", 1);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), l.input_size());
        EDGE_LEARNING_TEST_TRY(l.training_forward(v1));
        EDGE_LEARNING_TEST_TRY(l.forward(v1));
        EDGE_LEARNING_TEST_TRY(l.backward(v1));
        EDGE_LEARNING_TEST_ASSERT(!l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v1.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_input()[0], v1[0]);
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());

        std::vector<NumType> v2{2};
        DropoutLayer l_copy{l};
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), l_copy.input_size());
        EDGE_LEARNING_TEST_ASSERT(!l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v1.size());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input()[0], v1[0]);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output().size(),
                                 l_copy.output_size());
        EDGE_LEARNING_TEST_TRY(l_copy.training_forward(v2));
        EDGE_LEARNING_TEST_TRY(l_copy.forward(v2));
        EDGE_LEARNING_TEST_TRY(l_copy.backward(v2));
        EDGE_LEARNING_TEST_ASSERT(!l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v2.size());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input()[0], v2[0]);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output().size(),
                                 l_copy.output_size());

        DropoutLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), l_assign.input_size());
        EDGE_LEARNING_TEST_ASSERT(!l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v1.size());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input()[0], v1[0]);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output().size(),
                                 l_assign.output_size());
        EDGE_LEARNING_TEST_TRY(l_assign.training_forward(v2));
        EDGE_LEARNING_TEST_TRY(l_assign.forward(v2));
        EDGE_LEARNING_TEST_TRY(l_assign.backward(v2));
        EDGE_LEARNING_TEST_ASSERT(!l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v2.size());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input()[0], v2[0]);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output().size(),
                                 l_assign.output_size());

        NumType p = 0.9;
        std::vector<NumType> v_complex{0,1,2,3,4,5,6,7,8,9};
        auto l_complex = DropoutLayer(_m, "dropout_layer_test",
                                      v_complex.size(), p);
        EDGE_LEARNING_TEST_TRY(l_complex.training_forward(v_complex));
        EDGE_LEARNING_TEST_TRY(l_complex.forward(v_complex));
        EDGE_LEARNING_TEST_TRY(l_complex.backward(v_complex));
        p = 1.0;
        l_complex = DropoutLayer(_m, "dropout_layer_test",
                                 v_complex.size(), p);
        EDGE_LEARNING_TEST_TRY(l_complex.training_forward(v_complex));
        EDGE_LEARNING_TEST_TRY(l_complex.forward(v_complex));
        EDGE_LEARNING_TEST_TRY(l_complex.backward(v_complex));
        p = 0.4;
        RneType rne(0);
        l_complex = DropoutLayer(_m, "dropout_layer_test",
                                 v_complex.size(), p, rne);
        EDGE_LEARNING_TEST_TRY(l_complex.training_forward(v_complex));
        EDGE_LEARNING_TEST_TRY(l_complex.forward(v_complex));
        EDGE_LEARNING_TEST_TRY(l_complex.backward(v_complex));
    }

    void test_getter()
    {
        SizeType size = 1;
        auto l = DropoutLayer(_m, "dropout_layer_test", size);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), size);
    }

    void test_setter()
    {
        SizeType input_size = 1;
        auto l = DropoutLayer(_m, "dropout_layer_test", input_size);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), input_size);
        input_size = 10;
        EDGE_LEARNING_TEST_CALL(l.input_shape(input_size));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), input_size);
    }

    void test_stream()
    {
        NumType drop_probability = 0.1;
        auto l = DropoutLayer(_m, "dropout_layer_test", 1, drop_probability);

        Json l_dump;
        EDGE_LEARNING_TEST_TRY(l.dump(l_dump));
        EDGE_LEARNING_TEST_PRINT(l_dump);
        EDGE_LEARNING_TEST_EQUAL(l_dump["type"].as<std::string>(), "Dropout");
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

        l = DropoutLayer(_m);
        EDGE_LEARNING_TEST_TRY(l.load(l_dump));
        EDGE_LEARNING_TEST_EQUAL(l.type(), "Dropout");
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

        EDGE_LEARNING_TEST_WITHIN(
            l_dump["others"]["drop_probability"].as<NumType>(),
            drop_probability, 0.00000001);
    }
    
    Model _m = Model("model_dropout_layer_test");
};

int main() {
    TestDropoutLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
