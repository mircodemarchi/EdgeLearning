/***************************************************************************
 *            tests/dnn/test_layer.cpp
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


class TestCCELoss {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_layer());
    }

private:
    void test_layer() {
        EDGE_LEARNING_TEST_EXECUTE(
            auto l = CCELossLayer(_m, "cce_loss_layer_test", 1, 1));
        EDGE_LEARNING_TEST_TRY(
            auto l = CCELossLayer(_m, "cce_loss_layer_test", 1, 1));

        auto l = CCELossLayer(_m, "cce_loss_layer_test", 1, 1);
        EDGE_LEARNING_TEST_TRY(RneType r; l.init(r));
        EDGE_LEARNING_TEST_TRY(l.forward(nullptr));
        EDGE_LEARNING_TEST_TRY(l.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l.print());

        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.param(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.gradient(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "cce_loss_layer_test");
    }

    Model _m = Model("model_cce_loss_layer_test");
};

int main() {
    TestCCELoss().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
