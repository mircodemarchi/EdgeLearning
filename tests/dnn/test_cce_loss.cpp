/***************************************************************************
 *            dnn/test_cce_loss.cpp
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
#include "dnn/cce_loss.hpp"
#include "dnn/model.hpp"

using namespace std;
using namespace EdgeLearning;


class TestCCELossLayer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_layer());
        EDGE_LEARNING_TEST_CALL(test_loss_layer());
        EDGE_LEARNING_TEST_CALL(test_score());
        EDGE_LEARNING_TEST_CALL(test_cce_loss_layer());
    }

private:
    void test_layer() {
        EDGE_LEARNING_TEST_EXECUTE(
                auto l1 = CCELossLayer(_m, "cce_loss_layer_test"));
        EDGE_LEARNING_TEST_TRY(
                auto l2 = CCELossLayer(_m, "cce_loss_layer_test"));
        auto l = CCELossLayer(_m, "cce_loss_layer_test");
        EDGE_LEARNING_TEST_TRY(RneType r; l.init(r));
        EDGE_LEARNING_TEST_TRY(l.print());
        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.param(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.gradient(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "cce_loss_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(CCELossLayer l1_copy{l});
        EDGE_LEARNING_TEST_TRY(CCELossLayer l2_copy{l});
        CCELossLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(RneType r; l_copy.init(r));
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.param(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.gradient(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "cce_loss_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(CCELossLayer l_assign(_m); l_assign = l);
        EDGE_LEARNING_TEST_TRY(CCELossLayer l_assign(_m); l_assign = l);
        CCELossLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_TRY(RneType r; l_assign.init(r));
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.param(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.gradient(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "cce_loss_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(auto l1_noname = CCELossLayer(_m));
        EDGE_LEARNING_TEST_TRY(auto l2_noname = CCELossLayer(_m));
        auto l_noname = CCELossLayer(_m);
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());
    }

    void test_loss_layer() {
        EDGE_LEARNING_TEST_EXECUTE(
                auto l1 = CCELossLayer(_m,
                                       "cce_loss_layer_test",
                                       0, 0));
        EDGE_LEARNING_TEST_TRY(
                auto l2 = CCELossLayer(_m,
                                       "cce_loss_layer_test",
                                       0, 0));
        auto l = CCELossLayer(_m,
                              "cce_loss_layer_test",
                              6, 2);
        EDGE_LEARNING_TEST_EXECUTE(RneType r; l.init(r));
        EDGE_LEARNING_TEST_EXECUTE(l.print());
        EDGE_LEARNING_TEST_EXECUTE(l.set_target(nullptr));
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 6);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);

        CCELossLayer l_shape_copy{l};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), 6);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_output(), nullptr);
        CCELossLayer l_shape_assign(_m); l_shape_assign = l;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), 6);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_output(), nullptr);
    }

    void test_score() {
        auto l = CCELossLayer(_m, "cce_loss_layer_test", 1);
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_EXECUTE(l.reset_score());
        EDGE_LEARNING_TEST_EXECUTE(l.print());
        std::vector<NumType> v{0};
        std::vector<NumType> t2{1};
        EDGE_LEARNING_TEST_EXECUTE(l.set_target(t2.data()));
        for (SizeType i = 0; i < 10; ++i)
        {
            l.forward(v.data());
        }
        EDGE_LEARNING_TEST_NOT_EQUAL(l.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.last_input(), v.data());
        EDGE_LEARNING_TEST_EQUAL(l.last_output(), nullptr);
        EDGE_LEARNING_TEST_EXECUTE(l.print());
        EDGE_LEARNING_TEST_PRINT(l.accuracy());
        EDGE_LEARNING_TEST_PRINT(l.avg_loss());
        EDGE_LEARNING_TEST_EXECUTE(l.reset_score());
        EDGE_LEARNING_TEST_ASSERT(l.accuracy() != l.accuracy());
        EDGE_LEARNING_TEST_ASSERT(l.avg_loss() != l.avg_loss());

        CCELossLayer l_shape_copy{l};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), 0);
        EDGE_LEARNING_TEST_NOT_EQUAL(l_shape_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_input(), v.data());
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_output(), nullptr);
        CCELossLayer l_shape_assign(_m); l_shape_assign = l;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), 0);
        EDGE_LEARNING_TEST_NOT_EQUAL(l_shape_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_input(), v.data());
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_output(), nullptr);
    }

    void test_cce_loss_layer() {
        auto l = CCELossLayer(_m, "cce_loss_layer_test",
                              1, 2);
        std::vector<NumType> v1{0};
        std::vector<NumType> target_not_active{0};
        std::vector<NumType> target_active{1};
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 1);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_FAIL(l.forward(v1.data()));
        EDGE_LEARNING_TEST_THROWS(l.forward(v1.data()),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.last_output(), nullptr);

        EDGE_LEARNING_TEST_TRY(l.set_target(target_not_active.data()));
        EDGE_LEARNING_TEST_FAIL(l.forward(v1.data()));
        EDGE_LEARNING_TEST_THROWS(l.forward(v1.data()),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(l.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.last_output(), nullptr);

        EDGE_LEARNING_TEST_TRY(l.set_target(target_active.data()));
        EDGE_LEARNING_TEST_TRY(l.forward(v1.data()));
        EDGE_LEARNING_TEST_TRY(l.reverse(v1.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.last_input(), v1.data());
        EDGE_LEARNING_TEST_EQUAL(l.last_output(), nullptr);

        std::vector<NumType> v2{10};
        CCELossLayer l_copy{l};
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), v1.data());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output(), nullptr);
        EDGE_LEARNING_TEST_TRY(l_copy.forward(v2.data()));
        EDGE_LEARNING_TEST_TRY(l_copy.reverse(v2.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), v2.data());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output(), nullptr);

        CCELossLayer l_assign(_m); l_assign = l;
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), v1.data());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output(), nullptr);
        EDGE_LEARNING_TEST_TRY(l_assign.forward(v2.data()));
        EDGE_LEARNING_TEST_TRY(l_assign.reverse(v2.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), v2.data());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output(), nullptr);
    }

    Model _m = Model("model_cce_loss_layer_test");
};

int main() {
    TestCCELossLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
