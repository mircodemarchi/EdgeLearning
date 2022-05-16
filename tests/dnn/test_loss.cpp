/***************************************************************************
 *            dnn/test_loss.cpp
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
#include "dnn/loss.hpp"
#include "dnn/model.hpp"

using namespace std;
using namespace EdgeLearning;

class CustomLossLayer: public LossLayer {
public:
    CustomLossLayer(SizeType input_size = 0, SizeType batch_size = 1)
        : LossLayer(_m, input_size, batch_size, "custom_loss_layer_test")
        , _m{"model_loss_layer_test"}
        , _i{0}
    { }

    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override
    {
        _last_input = inputs.data();
        if (_i++ % 2 == 0) ++_correct; else ++_incorrect;
        _cumulative_loss += 2.0;
        return inputs;
    }

    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override { return gradients; }

private:
    Model _m;
    SizeType _i;
};

class CustomLossLayerNoName: public LossLayer {
public:
    CustomLossLayerNoName()
        : LossLayer(_m)
        , _m{"model_layer_test"}
    { }
    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override
    {
        _last_input = inputs.data();
        return inputs;
    }

    const std::vector<NumType>& backward(
        const std::vector<NumType>&gradients) override
    {
        return gradients;
    }

private:
    Model _m;
};

class TestLossLayer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_layer());
        EDGE_LEARNING_TEST_CALL(test_loss_layer());
        EDGE_LEARNING_TEST_CALL(test_score());
        EDGE_LEARNING_TEST_CALL(test_getter());
        EDGE_LEARNING_TEST_CALL(test_setter());
    }

private:
    void test_layer() {
        EDGE_LEARNING_TEST_EQUAL(LossLayer::TYPE, "Loss");
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        EDGE_LEARNING_TEST_EXECUTE(auto l1 = CustomLossLayer());
        EDGE_LEARNING_TEST_TRY(auto l2 = CustomLossLayer());
        auto l = CustomLossLayer();
        EDGE_LEARNING_TEST_EQUAL(l.TYPE, "Loss");
        EDGE_LEARNING_TEST_EQUAL(l.type(), "Loss");
        EDGE_LEARNING_TEST_TRY(l.init());
        EDGE_LEARNING_TEST_TRY(l.forward(v_empty));
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
        EDGE_LEARNING_TEST_EQUAL(l.name(), "custom_loss_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l.last_output());
        EDGE_LEARNING_TEST_THROWS(l.last_output(), std::runtime_error);
        EDGE_LEARNING_TEST_TRY(l.training_forward(v));
        EDGE_LEARNING_TEST_ASSERT(!l.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l.last_input().size(), v.size());
        EDGE_LEARNING_TEST_FAIL(l.last_output());
        EDGE_LEARNING_TEST_THROWS(l.last_output(), std::runtime_error);

        EDGE_LEARNING_TEST_EXECUTE(CustomLossLayer l_copy{l});
        EDGE_LEARNING_TEST_TRY(CustomLossLayer l_copy{l});
        CustomLossLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(l_copy.init());
        EDGE_LEARNING_TEST_ASSERT(!l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v.size());
        EDGE_LEARNING_TEST_TRY(l_copy.input_size(0));
        EDGE_LEARNING_TEST_TRY(l_copy.training_forward(v_empty));
        EDGE_LEARNING_TEST_TRY(l_copy.forward(v_empty));
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
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "custom_loss_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_copy.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l_copy.last_output());
        EDGE_LEARNING_TEST_THROWS(l_copy.last_output(), std::runtime_error);
        EDGE_LEARNING_TEST_TRY(l_copy.training_forward(v));
        EDGE_LEARNING_TEST_ASSERT(!l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v.size());
        EDGE_LEARNING_TEST_FAIL(l_copy.last_output());
        EDGE_LEARNING_TEST_THROWS(l_copy.last_output(), std::runtime_error);

        EDGE_LEARNING_TEST_EXECUTE(CustomLossLayer l_assign; l_assign = l);
        EDGE_LEARNING_TEST_TRY(CustomLossLayer l_assign; l_assign = l);
        CustomLossLayer l_assign; l_assign = l;
        EDGE_LEARNING_TEST_TRY(l_assign.init());
        EDGE_LEARNING_TEST_ASSERT(!l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v.size());
        EDGE_LEARNING_TEST_TRY(l_assign.input_size(0));
        EDGE_LEARNING_TEST_TRY(l_assign.training_forward(v_empty));
        EDGE_LEARNING_TEST_TRY(l_assign.forward(v_empty));
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
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "custom_loss_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_assign.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l_assign.last_output());
        EDGE_LEARNING_TEST_THROWS(l_assign.last_output(), std::runtime_error);
        EDGE_LEARNING_TEST_TRY(l_assign.training_forward(v));
        EDGE_LEARNING_TEST_ASSERT(!l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v.size());
        EDGE_LEARNING_TEST_FAIL(l_assign.last_output());
        EDGE_LEARNING_TEST_THROWS(l_assign.last_output(), std::runtime_error);

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = CustomLossLayerNoName());
        EDGE_LEARNING_TEST_TRY(auto l2 = CustomLossLayerNoName());
        auto l_noname = CustomLossLayerNoName();
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());
    }

    void test_loss_layer() {
        std::vector<NumType> v_empty;
        EDGE_LEARNING_TEST_EXECUTE(
                auto l1 = CustomLossLayer(0, 0));
        EDGE_LEARNING_TEST_TRY(
                auto l2 = CustomLossLayer(0, 0));
        auto l = CustomLossLayer(6, 2);
        EDGE_LEARNING_TEST_TRY(l.init());
        EDGE_LEARNING_TEST_EXECUTE(l.print());
        EDGE_LEARNING_TEST_EXECUTE(l.set_target(v_empty))
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 6);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);

        CustomLossLayer l_shape_copy{l};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), 6);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_shape_copy.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l_shape_copy.last_output());
        EDGE_LEARNING_TEST_THROWS(l_shape_copy.last_output(),
                                  std::runtime_error);
        CustomLossLayer l_shape_assign; l_shape_assign = l;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), 6);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), 0);
        EDGE_LEARNING_TEST_ASSERT(l_shape_assign.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l_shape_assign.last_output());
        EDGE_LEARNING_TEST_THROWS(l_shape_assign.last_output(),
                                  std::runtime_error);
    }

    void test_score() {
        std::vector<NumType> v_empty;
        auto l = CustomLossLayer(6, 2);
        EDGE_LEARNING_TEST_EXECUTE(l.reset_score());
        EDGE_LEARNING_TEST_EXECUTE(l.print());
        for (SizeType i = 0; i < 10; ++i)
        {
            l.forward(v_empty);
        }
        EDGE_LEARNING_TEST_ASSERT(l.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l.last_output());
        EDGE_LEARNING_TEST_THROWS(l.last_output(), std::runtime_error);
        EDGE_LEARNING_TEST_EXECUTE(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.accuracy(), 0.5);
        EDGE_LEARNING_TEST_EQUAL(l.avg_loss(), 2.0);
        EDGE_LEARNING_TEST_EXECUTE(l.reset_score());
        EDGE_LEARNING_TEST_ASSERT(l.accuracy() != l.accuracy());
        EDGE_LEARNING_TEST_ASSERT(l.avg_loss() != l.avg_loss());
    }

    void test_getter()
    {
        SizeType input_size = 1;
        auto l = CustomLossLayer(input_size);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
    }

    void test_setter()
    {
        SizeType input_size = 1;
        auto l = CustomLossLayer(input_size, 2);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);
        input_size = 10;
        EDGE_LEARNING_TEST_CALL(l.input_size(input_size));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), input_size);
    }
};

int main() {
    TestLossLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
