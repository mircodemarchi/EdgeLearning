/***************************************************************************
 *            tests/dnn/test_loss.cpp
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
    void forward(NumType* inputs) override {
        (void) inputs;
        if (_i % 2 == 0)
        {
            ++_correct;
        }
        else
        {
            ++_incorrect;
        }
        _cumulative_loss += 2.0;
        ++_i;
    }

    void reverse(NumType* gradients) override { (void) gradients; }

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
    void forward(NumType* inputs) override { (void) inputs; }
    void reverse(NumType* gradients) override { (void) gradients; }

private:
    Model _m;
};

class TestLossLayer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_layer());
        EDGE_LEARNING_TEST_CALL(test_loss_layer());
        EDGE_LEARNING_TEST_CALL(test_score());
    }

private:
    void test_layer() {
        EDGE_LEARNING_TEST_EXECUTE(auto l1 = CustomLossLayer());
        EDGE_LEARNING_TEST_TRY(auto l2 = CustomLossLayer());
        auto l = CustomLossLayer();
        EDGE_LEARNING_TEST_TRY(RneType r; l.init(r));
        EDGE_LEARNING_TEST_TRY(l.forward(nullptr));
        EDGE_LEARNING_TEST_TRY(l.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.param(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.gradient(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "custom_loss_layer_test");

        EDGE_LEARNING_TEST_EXECUTE(CustomLossLayer l_copy{l});
        EDGE_LEARNING_TEST_TRY(CustomLossLayer l_copy{l});
        CustomLossLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(RneType r; l_copy.init(r));
        EDGE_LEARNING_TEST_TRY(l_copy.forward(nullptr));
        EDGE_LEARNING_TEST_TRY(l_copy.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.param(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.gradient(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "custom_loss_layer_test");

        EDGE_LEARNING_TEST_EXECUTE(CustomLossLayer l_assign; l_assign = l);
        EDGE_LEARNING_TEST_TRY(CustomLossLayer l_assign; l_assign = l);
        CustomLossLayer l_assign; l_assign = l;
        EDGE_LEARNING_TEST_TRY(RneType r; l_assign.init(r));
        EDGE_LEARNING_TEST_TRY(l_assign.forward(nullptr));
        EDGE_LEARNING_TEST_TRY(l_assign.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.param(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.gradient(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "custom_loss_layer_test");

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = CustomLossLayerNoName());
        EDGE_LEARNING_TEST_TRY(auto l2 = CustomLossLayerNoName());
        auto l_noname = CustomLossLayerNoName();
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());
    }

    void test_loss_layer() {
        EDGE_LEARNING_TEST_EXECUTE(auto l1 = CustomLossLayer(0, 0));
        EDGE_LEARNING_TEST_TRY(auto l2 = CustomLossLayer(0, 0));
        auto l = CustomLossLayer(6, 2);
        EDGE_LEARNING_TEST_EXECUTE(RneType r; l.init(r));
        EDGE_LEARNING_TEST_EXECUTE(l.print());
        EDGE_LEARNING_TEST_EXECUTE(l.set_target(nullptr))
    }

    void test_score() {
        auto l = CustomLossLayer(6, 2);
        EDGE_LEARNING_TEST_EXECUTE(l.reset_score());
        EDGE_LEARNING_TEST_EXECUTE(l.print());
        for (SizeType i = 0; i < 10; ++i)
        {
            l.forward(nullptr);
        }
        EDGE_LEARNING_TEST_EXECUTE(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.accuracy(), 0.5);
        EDGE_LEARNING_TEST_EQUAL(l.avg_loss(), 2.0);
        EDGE_LEARNING_TEST_EXECUTE(l.reset_score());
        EDGE_LEARNING_TEST_ASSERT(l.accuracy() != l.accuracy());
        EDGE_LEARNING_TEST_ASSERT(l.avg_loss() != l.avg_loss());
    }
};

int main() {
    TestLossLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
