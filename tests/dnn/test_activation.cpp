/***************************************************************************
 *            dnn/test_dense.cpp
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
#include "dnn/activation.hpp"
#include "dnn/model.hpp"

using namespace std;
using namespace EdgeLearning;


class TestActivationLayer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_relu());
        EDGE_LEARNING_TEST_CALL(test_softmax());
        EDGE_LEARNING_TEST_CALL(test_tanh());
        EDGE_LEARNING_TEST_CALL(test_linear());
    }

private:
    void test_relu() {
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(
                auto l = ReluLayer(_m, "relu_layer_test"));
        EDGE_LEARNING_TEST_TRY(
                auto l = ReluLayer(_m, "relu_layer_test"));
        auto l = ReluLayer(_m, "relu_layer_test");
        EDGE_LEARNING_TEST_TRY(l.init());
        EDGE_LEARNING_TEST_TRY(l.forward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.backward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l.param(0));
        EDGE_LEARNING_TEST_THROWS(l.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "relu_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());

        EDGE_LEARNING_TEST_EXECUTE(ReluLayer l1_copy{l});
        EDGE_LEARNING_TEST_TRY(ReluLayer l2_copy{l});
        ReluLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(l_copy.init());
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_copy.param(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_copy.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "relu_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output().size(),
                                 l_copy.output_size());

        EDGE_LEARNING_TEST_EXECUTE(ReluLayer l_assign(_m); l_assign = l);
        EDGE_LEARNING_TEST_TRY(ReluLayer l_assign(_m); l_assign = l);
        ReluLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_TRY(l_assign.init());
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_assign.param(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_assign.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "relu_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output().size(),
                                 l_assign.output_size());

        EDGE_LEARNING_TEST_TRY(l.training_forward(v));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), l.input_size());
        EDGE_LEARNING_TEST_ASSERT(!l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = ReluLayer(_m));
        EDGE_LEARNING_TEST_TRY(auto l2 = ReluLayer(_m));
        auto l_noname = ReluLayer(_m);
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());

        SizeType size = 10;
        auto l_shape = ReluLayer(_m, "relu_layer_test", size);
        EDGE_LEARNING_TEST_EQUAL(l_shape.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l_shape.output_size(), size);
        EDGE_LEARNING_TEST_ASSERT(l_shape.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape.last_output().size(),
                                 l_shape.output_size());
        ReluLayer l_shape_copy{l_shape};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), size);
        EDGE_LEARNING_TEST_ASSERT(l_shape_copy.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape_copy.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_output().size(),
                                 l_shape_copy.output_size());
        ReluLayer l_shape_assign(_m); l_shape_assign = l_shape;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), size);
        EDGE_LEARNING_TEST_ASSERT(l_shape_assign.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape_assign.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_output().size(),
                                 l_shape_assign.output_size());
    }

    void test_softmax() {
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(
            auto l = SoftmaxLayer(_m, "softmax_layer_test"));
        EDGE_LEARNING_TEST_TRY(
            auto l = SoftmaxLayer(_m, "softmax_layer_test"));
        auto l = SoftmaxLayer(_m, "softmax_layer_test");
        EDGE_LEARNING_TEST_TRY(l.init());
        EDGE_LEARNING_TEST_TRY(l.forward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.backward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l.param(0));
        EDGE_LEARNING_TEST_THROWS(l.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "softmax_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());

        EDGE_LEARNING_TEST_EXECUTE(SoftmaxLayer l1_copy{l});
        EDGE_LEARNING_TEST_TRY(SoftmaxLayer l2_copy{l});
        SoftmaxLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(l_copy.init());
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_copy.param(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_copy.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "softmax_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output().size(),
                                 l_copy.output_size());

        EDGE_LEARNING_TEST_EXECUTE(SoftmaxLayer l_assign(_m); l_assign = l);
        EDGE_LEARNING_TEST_TRY(SoftmaxLayer l_assign(_m); l_assign = l);
        SoftmaxLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_TRY(l_assign.init());
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_assign.param(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_assign.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "softmax_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output().size(),
                                 l_assign.output_size());

        EDGE_LEARNING_TEST_TRY(l.training_forward(v));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), l.input_size());
        EDGE_LEARNING_TEST_ASSERT(!l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = SoftmaxLayer(_m));
        EDGE_LEARNING_TEST_TRY(auto l2 = SoftmaxLayer(_m));
        auto l_noname = SoftmaxLayer(_m);
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());

        SizeType size = 10;
        auto l_shape = SoftmaxLayer(_m, "softmax_layer_test", size);
        EDGE_LEARNING_TEST_EQUAL(l_shape.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l_shape.output_size(), size);
        EDGE_LEARNING_TEST_ASSERT(l_shape.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape.last_output().size(),
                                 l_shape.output_size());
        SoftmaxLayer l_shape_copy{l_shape};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), size);
        EDGE_LEARNING_TEST_ASSERT(l_shape_copy.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape_copy.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_output().size(),
                                 l_shape_copy.output_size());
        SoftmaxLayer l_shape_assign(_m); l_shape_assign = l_shape;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), size);
        EDGE_LEARNING_TEST_ASSERT(l_shape_assign.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape_assign.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_output().size(),
                                 l_shape_assign.output_size());
    }

    void test_tanh() {
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(
            auto l = TanhLayer(_m, "tanh_layer_test"));
        EDGE_LEARNING_TEST_TRY(
            auto l = TanhLayer(_m, "tanh_layer_test"));
        auto l = TanhLayer(_m, "tanh_layer_test");
        EDGE_LEARNING_TEST_TRY(l.init());
        EDGE_LEARNING_TEST_TRY(l.forward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.backward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l.param(0));
        EDGE_LEARNING_TEST_THROWS(l.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "tanh_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());

        EDGE_LEARNING_TEST_EXECUTE(TanhLayer l1_copy{l});
        EDGE_LEARNING_TEST_TRY(TanhLayer l2_copy{l});
        TanhLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(l_copy.init());
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_copy.param(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_copy.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "tanh_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output().size(),
                                 l_copy.output_size());

        EDGE_LEARNING_TEST_EXECUTE(TanhLayer l_assign(_m); l_assign = l);
        EDGE_LEARNING_TEST_TRY(TanhLayer l_assign(_m); l_assign = l);
        TanhLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_TRY(l_assign.init());
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_assign.param(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_assign.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "tanh_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output().size(),
                                 l_assign.output_size());

        EDGE_LEARNING_TEST_TRY(l.training_forward(v));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), l.input_size());
        EDGE_LEARNING_TEST_ASSERT(!l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = TanhLayer(_m));
        EDGE_LEARNING_TEST_TRY(auto l2 = TanhLayer(_m));
        auto l_noname = TanhLayer(_m);
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());

        SizeType size = 10;
        auto l_shape = TanhLayer(_m, "tanh_layer_test", size);
        EDGE_LEARNING_TEST_EQUAL(l_shape.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l_shape.output_size(), size);
        EDGE_LEARNING_TEST_ASSERT(l_shape.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape.last_output().size(),
                                 l_shape.output_size());
        TanhLayer l_shape_copy{l_shape};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), size);
        EDGE_LEARNING_TEST_ASSERT(l_shape_copy.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape_copy.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_output().size(),
                                 l_shape_copy.output_size());
        TanhLayer l_shape_assign(_m); l_shape_assign = l_shape;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), size);
        EDGE_LEARNING_TEST_ASSERT(l_shape_assign.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape_assign.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_output().size(),
                                 l_shape_assign.output_size());
    }

    void test_linear() {
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(
            auto l = LinearLayer(_m, "linear_layer_test"));
        EDGE_LEARNING_TEST_TRY(
            auto l = LinearLayer(_m, "linear_layer_test"));
        auto l = LinearLayer(_m, "linear_layer_test");
        EDGE_LEARNING_TEST_TRY(l.init());
        EDGE_LEARNING_TEST_TRY(l.forward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.backward(v_empty));
        EDGE_LEARNING_TEST_TRY(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l.param(0));
        EDGE_LEARNING_TEST_THROWS(l.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "linear_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());

        EDGE_LEARNING_TEST_EXECUTE(LinearLayer l1_copy{l});
        EDGE_LEARNING_TEST_TRY(LinearLayer l2_copy{l});
        LinearLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(l_copy.init());
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_copy.param(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_copy.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_copy.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "linear_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output().size(),
                                 l_copy.output_size());

        EDGE_LEARNING_TEST_EXECUTE(LinearLayer l_assign(_m); l_assign = l);
        EDGE_LEARNING_TEST_TRY(LinearLayer l_assign(_m); l_assign = l);
        LinearLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_TRY(l_assign.init());
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_FAIL(l_assign.param(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.param(0), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_assign.gradient(0));
        EDGE_LEARNING_TEST_THROWS(l_assign.gradient(0), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "linear_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v_empty.size());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output().size(),
                                 l_assign.output_size());

        EDGE_LEARNING_TEST_TRY(l.training_forward(v));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), l.input_size());
        EDGE_LEARNING_TEST_ASSERT(!l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v.size());
        EDGE_LEARNING_TEST_EQUAL(l.last_output().size(), l.output_size());

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = LinearLayer(_m));
        EDGE_LEARNING_TEST_TRY(auto l2 = LinearLayer(_m));
        auto l_noname = LinearLayer(_m);
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());

        SizeType size = 10;
        auto l_shape = LinearLayer(_m, "linear_layer_test", size);
        EDGE_LEARNING_TEST_EQUAL(l_shape.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l_shape.output_size(), size);
        EDGE_LEARNING_TEST_ASSERT(l_shape.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape.last_output().size(),
                                 l_shape.output_size());
        LinearLayer l_shape_copy{l_shape};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), size);
        EDGE_LEARNING_TEST_ASSERT(l_shape_copy.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape_copy.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_output().size(),
                                 l_shape_copy.output_size());
        LinearLayer l_shape_assign(_m); l_shape_assign = l_shape;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), size);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), size);
        EDGE_LEARNING_TEST_ASSERT(l_shape_assign.last_input().empty());
        EDGE_LEARNING_TEST_ASSERT(!l_shape_assign.last_output().empty());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_output().size(),
                                 l_shape_assign.output_size());
    }

    Model _m = Model("model_activation_layer_test");
};

int main() {
    TestActivationLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
