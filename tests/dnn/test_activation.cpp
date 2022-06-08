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
        EDGE_LEARNING_TEST_CALL(test_stream());
    }

private:
    void test_relu() {
        EDGE_LEARNING_TEST_EQUAL(ReluLayer::TYPE, "Relu");
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(
                auto l = ReluLayer(_m, "relu_layer_test"));
        EDGE_LEARNING_TEST_TRY(
                auto l = ReluLayer(_m, "relu_layer_test"));
        auto l = ReluLayer(_m, "relu_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.TYPE, "Relu");
        EDGE_LEARNING_TEST_EQUAL(l.type(), "Relu");
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
        EDGE_LEARNING_TEST_EQUAL(SoftmaxLayer::TYPE, "Softmax");
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(
            auto l = SoftmaxLayer(_m, "softmax_layer_test"));
        EDGE_LEARNING_TEST_TRY(
            auto l = SoftmaxLayer(_m, "softmax_layer_test"));
        auto l = SoftmaxLayer(_m, "softmax_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.TYPE, "Softmax");
        EDGE_LEARNING_TEST_EQUAL(l.type(), "Softmax");
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
        EDGE_LEARNING_TEST_EQUAL(TanhLayer::TYPE, "Tanh");
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(
            auto l = TanhLayer(_m, "tanh_layer_test"));
        EDGE_LEARNING_TEST_TRY(
            auto l = TanhLayer(_m, "tanh_layer_test"));
        auto l = TanhLayer(_m, "tanh_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.TYPE, "Tanh");
        EDGE_LEARNING_TEST_EQUAL(l.type(), "Tanh");
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
        EDGE_LEARNING_TEST_EQUAL(LinearLayer::TYPE, "Linear");
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(
            auto l = LinearLayer(_m, "linear_layer_test"));
        EDGE_LEARNING_TEST_TRY(
            auto l = LinearLayer(_m, "linear_layer_test"));
        auto l = LinearLayer(_m, "linear_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.TYPE, "Linear");
        EDGE_LEARNING_TEST_EQUAL(l.type(), "Linear");
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

    void test_stream()
    {
        auto l_relu = ReluLayer(_m, "relu_layer_test", 10);

        Json l_dump;
        EDGE_LEARNING_TEST_TRY(l_relu.dump(l_dump));
        EDGE_LEARNING_TEST_PRINT(l_dump);
        EDGE_LEARNING_TEST_EQUAL(l_dump["type"].as<std::string>(), "Relu");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l_relu.name());

        auto input_size_arr = l_dump["input_size"].as_vec<std::size_t>();
        EDGE_LEARNING_TEST_EQUAL(input_size_arr.size(), 3);
        std::size_t input_size = input_size_arr[0]
                                 * input_size_arr[1] * input_size_arr[2];
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[0], l_relu.input_shape().height);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[1], l_relu.input_shape().width);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[2], l_relu.input_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(input_size, l_relu.input_size());

        auto output_size_arr = l_dump["output_size"].as_vec<std::size_t>();
        EDGE_LEARNING_TEST_EQUAL(output_size_arr.size(), 3);
        std::size_t output_size = output_size_arr[0]
            * output_size_arr[1] * output_size_arr[2];
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[0], l_relu.output_shape().height);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[1], l_relu.output_shape().width);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[2], l_relu.output_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(output_size, l_relu.output_size());

        EDGE_LEARNING_TEST_EQUAL(l_dump["antecedents"].size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_dump["subsequents"].size(), 0);

        l_relu = ReluLayer(_m);
        EDGE_LEARNING_TEST_TRY(l_relu.load(l_dump));
        EDGE_LEARNING_TEST_EQUAL(l_relu.type(), "Relu");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l_relu.name());
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[0], l_relu.input_shape().height);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[1], l_relu.input_shape().width);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[2], l_relu.input_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(input_size, l_relu.input_size());
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[0], l_relu.output_shape().height);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[1], l_relu.output_shape().width);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[2], l_relu.output_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(output_size, l_relu.output_size());

        Json json_void;
        EDGE_LEARNING_TEST_FAIL(l_relu.load(json_void));
        EDGE_LEARNING_TEST_THROWS(l_relu.load(json_void), std::runtime_error);

        auto l_softmax = SoftmaxLayer(_m, "softmax_layer_test", 10);

        EDGE_LEARNING_TEST_TRY(l_softmax.dump(l_dump));
        EDGE_LEARNING_TEST_PRINT(l_dump);
        EDGE_LEARNING_TEST_EQUAL(l_dump["type"].as<std::string>(), "Softmax");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l_softmax.name());

        input_size_arr = l_dump["input_size"].as_vec<std::size_t>();
        EDGE_LEARNING_TEST_EQUAL(input_size_arr.size(), 3);
        input_size = input_size_arr[0]
            * input_size_arr[1] * input_size_arr[2];
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[0], l_softmax.input_shape().height);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[1], l_softmax.input_shape().width);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[2], l_softmax.input_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(input_size, l_softmax.input_size());

        output_size_arr = l_dump["output_size"].as_vec<std::size_t>();
        EDGE_LEARNING_TEST_EQUAL(output_size_arr.size(), 3);
        output_size = output_size_arr[0]
            * output_size_arr[1] * output_size_arr[2];
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[0], l_softmax.output_shape().height);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[1], l_softmax.output_shape().width);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[2], l_softmax.output_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(output_size, l_softmax.output_size());

        EDGE_LEARNING_TEST_EQUAL(l_dump["antecedents"].size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_dump["subsequents"].size(), 0);

        l_softmax = SoftmaxLayer(_m);
        EDGE_LEARNING_TEST_TRY(l_softmax.load(l_dump));
        EDGE_LEARNING_TEST_EQUAL(l_softmax.type(), "Softmax");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l_softmax.name());
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[0], l_softmax.input_shape().height);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[1], l_softmax.input_shape().width);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[2], l_softmax.input_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(input_size, l_softmax.input_size());
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[0], l_softmax.output_shape().height);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[1], l_softmax.output_shape().width);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[2], l_softmax.output_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(output_size, l_softmax.output_size());

        EDGE_LEARNING_TEST_FAIL(l_softmax.load(json_void));
        EDGE_LEARNING_TEST_THROWS(l_softmax.load(json_void), std::runtime_error);

        auto l_tanh = TanhLayer(_m, "tanh_layer_test", 10);

        EDGE_LEARNING_TEST_TRY(l_tanh.dump(l_dump));
        EDGE_LEARNING_TEST_PRINT(l_dump);
        EDGE_LEARNING_TEST_EQUAL(l_dump["type"].as<std::string>(), "Tanh");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l_tanh.name());

        input_size_arr = l_dump["input_size"].as_vec<std::size_t>();
        EDGE_LEARNING_TEST_EQUAL(input_size_arr.size(), 3);
        input_size = input_size_arr[0]
            * input_size_arr[1] * input_size_arr[2];
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[0], l_tanh.input_shape().height);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[1], l_tanh.input_shape().width);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[2], l_tanh.input_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(input_size, l_tanh.input_size());

        output_size_arr = l_dump["output_size"].as_vec<std::size_t>();
        EDGE_LEARNING_TEST_EQUAL(output_size_arr.size(), 3);
        output_size = output_size_arr[0]
            * output_size_arr[1] * output_size_arr[2];
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[0], l_tanh.output_shape().height);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[1], l_tanh.output_shape().width);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[2], l_tanh.output_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(output_size, l_tanh.output_size());

        EDGE_LEARNING_TEST_EQUAL(l_dump["antecedents"].size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_dump["subsequents"].size(), 0);

        l_tanh = TanhLayer(_m);
        EDGE_LEARNING_TEST_TRY(l_tanh.load(l_dump));
        EDGE_LEARNING_TEST_EQUAL(l_tanh.type(), "Tanh");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l_tanh.name());
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[0], l_tanh.input_shape().height);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[1], l_tanh.input_shape().width);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[2], l_tanh.input_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(input_size, l_tanh.input_size());
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[0], l_tanh.output_shape().height);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[1], l_tanh.output_shape().width);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[2], l_tanh.output_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(output_size, l_tanh.output_size());

        EDGE_LEARNING_TEST_FAIL(l_tanh.load(json_void));
        EDGE_LEARNING_TEST_THROWS(l_tanh.load(json_void), std::runtime_error);

        auto l_linear = LinearLayer(_m, "linear_layer_test", 10);

        EDGE_LEARNING_TEST_TRY(l_linear.dump(l_dump));
        EDGE_LEARNING_TEST_PRINT(l_dump);
        EDGE_LEARNING_TEST_EQUAL(l_dump["type"].as<std::string>(), "Linear");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l_linear.name());

        input_size_arr = l_dump["input_size"].as_vec<std::size_t>();
        EDGE_LEARNING_TEST_EQUAL(input_size_arr.size(), 3);
        input_size = input_size_arr[0]
            * input_size_arr[1] * input_size_arr[2];
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[0], l_linear.input_shape().height);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[1], l_linear.input_shape().width);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[2], l_linear.input_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(input_size, l_linear.input_size());

        output_size_arr = l_dump["output_size"].as_vec<std::size_t>();
        EDGE_LEARNING_TEST_EQUAL(output_size_arr.size(), 3);
        output_size = output_size_arr[0]
            * output_size_arr[1] * output_size_arr[2];
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[0], l_linear.output_shape().height);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[1], l_linear.output_shape().width);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[2], l_linear.output_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(output_size, l_linear.output_size());

        EDGE_LEARNING_TEST_EQUAL(l_dump["antecedents"].size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_dump["subsequents"].size(), 0);

        l_linear = LinearLayer(_m);
        EDGE_LEARNING_TEST_TRY(l_linear.load(l_dump));
        EDGE_LEARNING_TEST_EQUAL(l_linear.type(), "Linear");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l_linear.name());
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[0], l_linear.input_shape().height);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[1], l_linear.input_shape().width);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[2], l_linear.input_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(input_size, l_linear.input_size());
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[0], l_linear.output_shape().height);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[1], l_linear.output_shape().width);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[2], l_linear.output_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(output_size, l_linear.output_size());

        EDGE_LEARNING_TEST_FAIL(l_linear.load(json_void));
        EDGE_LEARNING_TEST_THROWS(l_linear.load(json_void), std::runtime_error);
    }

    Model _m = Model("model_activation_layer_test");
};

int main() {
    TestActivationLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
