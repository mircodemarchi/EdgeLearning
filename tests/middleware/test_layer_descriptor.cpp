/***************************************************************************
 *            middleware/test_layer_descriptor.cpp
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
#include "middleware/layer_descriptor.hpp"

using namespace std;
using namespace EdgeLearning;


class TestFNN {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_layer_setting());
        EDGE_LEARNING_TEST_CALL(test_layer_descriptor());
        EDGE_LEARNING_TEST_CALL(test_layer_descriptor_implementations());
    }

private:
    void test_layer_setting() {
        EDGE_LEARNING_TEST_TRY(LayerSetting());
        auto ls = LayerSetting();
        EDGE_LEARNING_TEST_EQUAL(ls.units().size(), LayerShape(0).size());
        EDGE_LEARNING_TEST_EQUAL(ls.n_filters(), 0);
        EDGE_LEARNING_TEST_EQUAL(ls.kernel_shape().size(),
                                 DLMath::Shape2d(0).size());
        EDGE_LEARNING_TEST_EQUAL(ls.stride().size(),
                                 DLMath::Shape2d(0).size());
        EDGE_LEARNING_TEST_EQUAL(ls.padding().size(),
                                 DLMath::Shape2d(0).size());
        EDGE_LEARNING_TEST_EQUAL(ls.drop_probability(), 0.0);

        EDGE_LEARNING_TEST_TRY(ls.units(DLMath::Shape3d{100, 100}));
        EDGE_LEARNING_TEST_EQUAL(ls.units().size(),
                                 LayerShape(DLMath::Shape3d{100, 100}).size());
        EDGE_LEARNING_TEST_TRY(ls.n_filters(10));
        EDGE_LEARNING_TEST_EQUAL(ls.n_filters(), 10);
        EDGE_LEARNING_TEST_TRY(ls.kernel_shape({3, 3}));
        EDGE_LEARNING_TEST_EQUAL(ls.kernel_shape().size(),
                                 DLMath::Shape2d(3, 3).size());
        EDGE_LEARNING_TEST_TRY(ls.stride({2, 2}));
        EDGE_LEARNING_TEST_EQUAL(ls.stride().size(),
                                 DLMath::Shape2d(2, 2).size());
        EDGE_LEARNING_TEST_TRY(ls.padding({1, 1}));
        EDGE_LEARNING_TEST_EQUAL(ls.padding().size(),
                                 DLMath::Shape2d(1, 1).size());
        EDGE_LEARNING_TEST_TRY(ls.drop_probability(0.5));
        EDGE_LEARNING_TEST_EQUAL(ls.drop_probability(), 0.5);

        EDGE_LEARNING_TEST_TRY(LayerSetting(DLMath::Shape3d{100, 100}));
        ls = LayerSetting(DLMath::Shape3d{100, 100});
        EDGE_LEARNING_TEST_EQUAL(ls.units().size(),
                                 LayerShape(DLMath::Shape3d{100, 100}).size());
        EDGE_LEARNING_TEST_EQUAL(ls.n_filters(), 0);
        EDGE_LEARNING_TEST_EQUAL(ls.kernel_shape().size(),
                                 DLMath::Shape2d(0).size());
        EDGE_LEARNING_TEST_EQUAL(ls.stride().size(),
                                 DLMath::Shape2d(0).size());
        EDGE_LEARNING_TEST_EQUAL(ls.padding().size(),
                                 DLMath::Shape2d(0).size());
        EDGE_LEARNING_TEST_EQUAL(ls.drop_probability(), 0.0);

        EDGE_LEARNING_TEST_TRY(LayerSetting(10, {3,3}, {2,2}, {1,1}));
        ls = LayerSetting(10, {3,3}, {2,2}, {1,1});
        EDGE_LEARNING_TEST_EQUAL(ls.n_filters(), 10);
        EDGE_LEARNING_TEST_EQUAL(ls.kernel_shape().size(),
                                 DLMath::Shape2d(3, 3).size());
        EDGE_LEARNING_TEST_EQUAL(ls.stride().size(),
                                 DLMath::Shape2d(2, 2).size());
        EDGE_LEARNING_TEST_EQUAL(ls.padding().size(),
                                 DLMath::Shape2d(1, 1).size());

        EDGE_LEARNING_TEST_TRY(LayerSetting({3,3}, {2,2}));
        ls = LayerSetting({3,3}, {2,2});
        EDGE_LEARNING_TEST_EQUAL(ls.kernel_shape().size(),
                                 DLMath::Shape2d(3, 3).size());
        EDGE_LEARNING_TEST_EQUAL(ls.stride().size(),
                                 DLMath::Shape2d(2, 2).size());

        EDGE_LEARNING_TEST_TRY(LayerSetting(0.5));
        ls = LayerSetting(0.5);
        EDGE_LEARNING_TEST_EQUAL(ls.units().size(), LayerShape(0).size());
        EDGE_LEARNING_TEST_EQUAL(ls.n_filters(), 0);
        EDGE_LEARNING_TEST_EQUAL(ls.kernel_shape().size(),
                                 DLMath::Shape2d(0).size());
        EDGE_LEARNING_TEST_EQUAL(ls.stride().size(),
                                 DLMath::Shape2d(0).size());
        EDGE_LEARNING_TEST_EQUAL(ls.padding().size(),
                                 DLMath::Shape2d(0).size());
        EDGE_LEARNING_TEST_EQUAL(ls.drop_probability(), 0.5);
    }

    void test_layer_descriptor() {
        EDGE_LEARNING_TEST_TRY(LayerDescriptor("test", LayerType::Input));
        auto ld = LayerDescriptor("test", LayerType::Input);
        EDGE_LEARNING_TEST_EQUAL(ld.name(), "test");
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld.type()),
                                 static_cast<int>(LayerType::Input));
        EDGE_LEARNING_TEST_EQUAL(ld.setting().units().size(),
                                 LayerSetting().units().size());
        EDGE_LEARNING_TEST_EQUAL(ld.setting().n_filters(),
                                 LayerSetting().n_filters());
        EDGE_LEARNING_TEST_EQUAL(ld.setting().kernel_shape().size(),
                                 LayerSetting().kernel_shape().size());
        EDGE_LEARNING_TEST_EQUAL(ld.setting().stride().size(),
                                 LayerSetting().stride().size());
        EDGE_LEARNING_TEST_EQUAL(ld.setting().padding().size(),
                                 LayerSetting().padding().size());
        EDGE_LEARNING_TEST_EQUAL(ld.setting().drop_probability(),
                                 LayerSetting().drop_probability());
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld.activation_type()),
                                 static_cast<int>(ActivationType::Linear));

        EDGE_LEARNING_TEST_TRY(ld.name("test_edit"));
        EDGE_LEARNING_TEST_EQUAL(ld.name(), "test_edit");
        EDGE_LEARNING_TEST_TRY(ld.type(LayerType::Dropout));
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld.type()),
                                 static_cast<int>(LayerType::Dropout));
        auto ls = LayerSetting(DLMath::Shape3d{100, 100});
        EDGE_LEARNING_TEST_TRY(ld.setting(ls));
        EDGE_LEARNING_TEST_EQUAL(ld.setting().units().size(),
                                 LayerShape(DLMath::Shape3d{100, 100}).size());
        EDGE_LEARNING_TEST_EQUAL(ld.setting().n_filters(), 0);
        EDGE_LEARNING_TEST_EQUAL(ld.setting().kernel_shape().size(),
                                 DLMath::Shape2d(0).size());
        EDGE_LEARNING_TEST_EQUAL(ld.setting().stride().size(),
                                 DLMath::Shape2d(0).size());
        EDGE_LEARNING_TEST_EQUAL(ld.setting().padding().size(),
                                 DLMath::Shape2d(0).size());
        EDGE_LEARNING_TEST_EQUAL(ld.setting().drop_probability(), 0.0);
        EDGE_LEARNING_TEST_TRY(ld.activation_type(ActivationType::ReLU));
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld.activation_type()),
                                 static_cast<int>(ActivationType::ReLU));
    }

    void test_layer_descriptor_implementations() {
        EDGE_LEARNING_TEST_TRY(Input("test_input", DLMath::Shape3d{100,100}));
        auto ld_input = Input("test_input", DLMath::Shape3d{100,100});
        EDGE_LEARNING_TEST_EQUAL(ld_input.name(), "test_input");
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld_input.type()),
                                 static_cast<int>(LayerType::Input));
        EDGE_LEARNING_TEST_EQUAL(ld_input.setting().units().size(),
                                 LayerSetting(DLMath::Shape3d{100,100})
                                    .units().size());
        EDGE_LEARNING_TEST_EQUAL(ld_input.setting().n_filters(),
                                 LayerSetting().n_filters());
        EDGE_LEARNING_TEST_EQUAL(ld_input.setting().kernel_shape().size(),
                                 LayerSetting().kernel_shape().size());
        EDGE_LEARNING_TEST_EQUAL(ld_input.setting().stride().size(),
                                 LayerSetting().stride().size());
        EDGE_LEARNING_TEST_EQUAL(ld_input.setting().padding().size(),
                                 LayerSetting().padding().size());
        EDGE_LEARNING_TEST_EQUAL(ld_input.setting().drop_probability(),
                                 LayerSetting().drop_probability());
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld_input.activation_type()),
                                 static_cast<int>(ActivationType::Linear));

        EDGE_LEARNING_TEST_TRY(Dense("test_relu", 100, ActivationType::ReLU));
        auto ld_dense = Dense("test_relu", 100, ActivationType::ReLU);
        EDGE_LEARNING_TEST_EQUAL(ld_dense.name(), "test_relu");
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld_dense.type()),
                                 static_cast<int>(LayerType::Dense));
        auto ls = LayerSetting(LayerShape(100));
        EDGE_LEARNING_TEST_EQUAL(ld_dense.setting().units().size(),
                                 ls.units().size());
        EDGE_LEARNING_TEST_EQUAL(ld_dense.setting().n_filters(),
                                 ls.n_filters());
        EDGE_LEARNING_TEST_EQUAL(ld_dense.setting().kernel_shape().size(),
                                 ls.kernel_shape().size());
        EDGE_LEARNING_TEST_EQUAL(ld_dense.setting().stride().size(),
                                 ls.stride().size());
        EDGE_LEARNING_TEST_EQUAL(ld_dense.setting().padding().size(),
                                 ls.padding().size());
        EDGE_LEARNING_TEST_EQUAL(ld_dense.setting().drop_probability(),
                                 ls.drop_probability());
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld_dense.activation_type()),
                                 static_cast<int>(ActivationType::ReLU));

        EDGE_LEARNING_TEST_TRY(
            Conv("test_conv",
                 {16, {3,3}, {2,2}, {1,1}},
                 ActivationType::Softmax));
        auto ld_conv = Conv("test_conv",
                            {16, {3,3}, {2,2}, {1,1}},
                            ActivationType::Softmax);
        EDGE_LEARNING_TEST_EQUAL(ld_conv.name(), "test_conv");
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld_conv.type()),
                                 static_cast<int>(LayerType::Conv));
        ls = LayerSetting(16, {3,3}, {2,2}, {1,1});
        EDGE_LEARNING_TEST_EQUAL(ld_conv.setting().units().size(),
                                 ls.units().size());
        EDGE_LEARNING_TEST_EQUAL(ld_conv.setting().n_filters(),
                                 ls.n_filters());
        EDGE_LEARNING_TEST_EQUAL(ld_conv.setting().kernel_shape().size(),
                                 ls.kernel_shape().size());
        EDGE_LEARNING_TEST_EQUAL(ld_conv.setting().stride().size(),
                                 ls.stride().size());
        EDGE_LEARNING_TEST_EQUAL(ld_conv.setting().padding().size(),
                                 ls.padding().size());
        EDGE_LEARNING_TEST_EQUAL(ld_conv.setting().drop_probability(),
                                 ls.drop_probability());
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld_conv.activation_type()),
                                 static_cast<int>(ActivationType::Softmax));

        EDGE_LEARNING_TEST_TRY(
            MaxPool("test_max_pool", {{3,3}, {2,2}}, ActivationType::ELU));
        auto ld_max_pool = MaxPool("test_max_pool", {{3,3}, {2,2}},
                                   ActivationType::ELU);
        EDGE_LEARNING_TEST_EQUAL(ld_max_pool.name(), "test_max_pool");
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld_max_pool.type()),
                                 static_cast<int>(LayerType::MaxPool));
        ls = LayerSetting({3,3}, {2,2});
        EDGE_LEARNING_TEST_EQUAL(ld_max_pool.setting().units().size(),
                                 ls.units().size());
        EDGE_LEARNING_TEST_EQUAL(ld_max_pool.setting().n_filters(),
                                 ls.n_filters());
        EDGE_LEARNING_TEST_EQUAL(ld_max_pool.setting().kernel_shape().size(),
                                 ls.kernel_shape().size());
        EDGE_LEARNING_TEST_EQUAL(ld_max_pool.setting().stride().size(),
                                 ls.stride().size());
        EDGE_LEARNING_TEST_EQUAL(ld_max_pool.setting().padding().size(),
                                 ls.padding().size());
        EDGE_LEARNING_TEST_EQUAL(ld_max_pool.setting().drop_probability(),
                                 ls.drop_probability());
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld_max_pool.activation_type()),
                                 static_cast<int>(ActivationType::ELU));

        EDGE_LEARNING_TEST_TRY(
            AvgPool("test_avg_pool", {{3,3}, {2,2}}, ActivationType::Sigmoid));
        auto ld_avg_pool = AvgPool("test_avg_pool", {{3,3}, {2,2}},
                                   ActivationType::Sigmoid);
        EDGE_LEARNING_TEST_EQUAL(ld_avg_pool.name(), "test_avg_pool");
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld_avg_pool.type()),
                                 static_cast<int>(LayerType::AvgPool));
        ls = LayerSetting({3,3}, {2,2});
        EDGE_LEARNING_TEST_EQUAL(ld_avg_pool.setting().units().size(),
                                 ls.units().size());
        EDGE_LEARNING_TEST_EQUAL(ld_avg_pool.setting().n_filters(),
                                 ls.n_filters());
        EDGE_LEARNING_TEST_EQUAL(ld_avg_pool.setting().kernel_shape().size(),
                                 ls.kernel_shape().size());
        EDGE_LEARNING_TEST_EQUAL(ld_avg_pool.setting().stride().size(),
                                 ls.stride().size());
        EDGE_LEARNING_TEST_EQUAL(ld_avg_pool.setting().padding().size(),
                                 ls.padding().size());
        EDGE_LEARNING_TEST_EQUAL(ld_avg_pool.setting().drop_probability(),
                                 ls.drop_probability());
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld_avg_pool.activation_type()),
                                 static_cast<int>(ActivationType::Sigmoid));

        EDGE_LEARNING_TEST_TRY(
            Dropout("test_dropout", 0.5, ActivationType::TanH));
        auto ld_dropout = Dropout("test_dropout", 0.5, ActivationType::TanH);
        EDGE_LEARNING_TEST_EQUAL(ld_dropout.name(), "test_dropout");
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld_dropout.type()),
                                 static_cast<int>(LayerType::Dropout));
        ls = LayerSetting(0.5);
        EDGE_LEARNING_TEST_EQUAL(ld_dropout.setting().units().size(),
                                 ls.units().size());
        EDGE_LEARNING_TEST_EQUAL(ld_dropout.setting().n_filters(),
                                 ls.n_filters());
        EDGE_LEARNING_TEST_EQUAL(ld_dropout.setting().kernel_shape().size(),
                                 ls.kernel_shape().size());
        EDGE_LEARNING_TEST_EQUAL(ld_dropout.setting().stride().size(),
                                 ls.stride().size());
        EDGE_LEARNING_TEST_EQUAL(ld_dropout.setting().padding().size(),
                                 ls.padding().size());
        EDGE_LEARNING_TEST_EQUAL(ld_dropout.setting().drop_probability(),
                                 ls.drop_probability());
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ld_dropout.activation_type()),
                                 static_cast<int>(ActivationType::TanH));
    }
};

int main() {
    TestFNN().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
