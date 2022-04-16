/***************************************************************************
 *            dnn/test_layer.cpp
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
    CustomLayer(SizeType input_size = 0, SizeType output_size = 0)
        : Layer(_m, input_size, output_size, Activation::None,
                "custom_layer_test")
        , _m{"model_layer_test"}
    { }
    void init(RneType& rne) override { (void) rne; }
    void forward(const NumType *inputs) override { _last_input = inputs; }
    void reverse(const NumType *gradients) override { (void) gradients; }
    void print() const override {}

private:
    Model _m;
};

class CustomLayerNoName: public Layer {
public:
    CustomLayerNoName() 
        : Layer(_m)
        , _m{"model_layer_test"}
    { }
    void init(RneType& rne) override { (void) rne; }
    void forward(const NumType *inputs) override { (void) inputs; }
    void reverse(const NumType *gradients) override { (void) gradients; }
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
        std::vector<NumType> v(std::size_t(10));
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
        EDGE_LEARNING_TEST_EQUAL(l.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.last_output(), nullptr);
        EDGE_LEARNING_TEST_TRY(l.forward(v.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l.last_input(), v.data());
        EDGE_LEARNING_TEST_EQUAL(l.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(CustomLayer l1_copy{l});
        EDGE_LEARNING_TEST_TRY(CustomLayer l2_copy{l});
        CustomLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(RneType r; l_copy.init(r));
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), v.data());
        EDGE_LEARNING_TEST_TRY(l_copy.forward(nullptr));
        EDGE_LEARNING_TEST_TRY(l_copy.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l_copy.print());
        EDGE_LEARNING_TEST_EQUAL(l_copy.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.param(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.gradient(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "custom_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_copy.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output(), nullptr);
        EDGE_LEARNING_TEST_TRY(l_copy.forward(v.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input(), v.data());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(CustomLayer l_assign; l_assign = l);
        EDGE_LEARNING_TEST_TRY(CustomLayer l_assign; l_assign = l);
        CustomLayer l_assign; l_assign = l;
        EDGE_LEARNING_TEST_TRY(RneType r; l_assign.init(r));
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), v.data());
        EDGE_LEARNING_TEST_TRY(l_assign.forward(nullptr));
        EDGE_LEARNING_TEST_TRY(l_assign.reverse(nullptr));
        EDGE_LEARNING_TEST_TRY(l_assign.print());
        EDGE_LEARNING_TEST_EQUAL(l_assign.param_count(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.param(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.param(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.gradient(0), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.gradient(10), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "custom_layer_test");
        EDGE_LEARNING_TEST_EQUAL(l_assign.input_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.output_size(), 0);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output(), nullptr);
        EDGE_LEARNING_TEST_TRY(l_assign.forward(v.data()));
        EDGE_LEARNING_TEST_NOT_EQUAL(l_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input(), v.data());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_output(), nullptr);

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = CustomLayerNoName());
        EDGE_LEARNING_TEST_TRY(auto l2 = CustomLayerNoName());
        auto l_noname = CustomLayerNoName();
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());

        auto l_shape = CustomLayer(10, 20);
        EDGE_LEARNING_TEST_EQUAL(l_shape.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape.output_size(), 20);
        EDGE_LEARNING_TEST_EQUAL(l_shape.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_shape.last_output(), nullptr);
        CustomLayer l_shape_copy{l_shape};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), 20);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.last_output(), nullptr);
        CustomLayer l_shape_assign; l_shape_assign = l_shape;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), 20);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_input(), nullptr);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.last_output(), nullptr);
    }

};

int main() {
    TestLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
