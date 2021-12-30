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
#include "dnn/layer.hpp"
#include "dnn/model.hpp"

using namespace std;
using namespace EdgeLearning;


class CustomLayer: public Layer {
public:
    CustomLayer() 
        : _m{"model_layer_test"}
        , Layer(_m, "custom_layer_test")
    { }
    void init(RneType& rne) override { (void) rne; }
    void forward(NumType* inputs) override { (void) inputs; }
    void reverse(NumType* gradients) override { (void) gradients; }
    void print() const override {}

private:
    Model _m;
};


class TestLayer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_layer());
    }

private:
    void test_layer() {
        EDGE_LEARNING_TEST_EXECUTE(auto l = CustomLayer());
        EDGE_LEARNING_TEST_TRY(auto l = CustomLayer());

        auto l = CustomLayer();
        EDGE_LEARNING_TEST_TRY(RneType r; l.init(r));
        EDGE_LEARNING_TEST_TRY(l.forward(nullptr));
        EDGE_LEARNING_TEST_TRY(l.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l.print());

        EDGE_LEARNING_TEST_EQUAL(l.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.param(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.gradient(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.name(), "custom_layer_test");
    }

};

int main() {
    TestLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
