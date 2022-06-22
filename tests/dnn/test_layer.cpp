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
    CustomLayer(SizeType input_size = 0, SizeType output_size = 0,
                std::string name = std::string(),
                SharedPtr antecedent = nullptr, SharedPtr subsequent = nullptr)
        : Layer(_m, input_size, output_size,
                name.empty() ? "custom_layer_test" : name)
        , _m{"model_layer_test"}
    {
        if (antecedent) _antecedents.push_back(antecedent);
        if (subsequent) _subsequents.push_back(subsequent);
    }

    void init(
        InitializationFunction init = InitializationFunction::KAIMING,
        ProbabilityDensityFunction pdf = ProbabilityDensityFunction::NORMAL,
        RneType rne = RneType())
        override
    {
        (void) init;
        (void) pdf;
        (void) rne;
    }

    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override
    {
        _last_input = inputs.data();
        return inputs;
    }

    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override
    { return gradients; }

    const std::vector<NumType>& last_output() override
    { throw std::runtime_error(""); }

    [[nodiscard]] SizeType param_count() const noexcept override
    { return 0; }

    NumType& param(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("");
    }

    NumType& gradient(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("");
    }

    [[nodiscard]] SharedPtr clone() const override
    { return std::make_shared<CustomLayer>(*this); }

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

    void init(
        InitializationFunction init = InitializationFunction::KAIMING,
        ProbabilityDensityFunction pdf = ProbabilityDensityFunction::NORMAL,
        RneType rne = RneType())
        override
    {
        (void) init;
        (void) pdf;
        (void) rne;
    }
    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override
    { return inputs; }

    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override
    { return gradients; }

    const std::vector<NumType>& last_output() override
    { throw std::runtime_error(""); }

    [[nodiscard]] SizeType param_count() const noexcept override
    { return 0; }

    NumType& param(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("");
    }

    NumType& gradient(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("");
    }

    [[nodiscard]] SharedPtr clone() const override
    { return std::make_shared<CustomLayerNoName>(*this); }

    void print() const override {}

private:
    Model _m;
};

class CustomLayerType : public CustomLayer
{
public:
    static const std::string TYPE;

    CustomLayerType(SizeType input_size = 0, SizeType output_size = 0,
                    std::string name = std::string(),
                    SharedPtr antecedent = nullptr,
                    SharedPtr subsequent = nullptr)
        : CustomLayer(input_size, output_size, name, antecedent, subsequent)
    { }

    [[nodiscard]] inline const std::string& type() const override
    { return TYPE; }

    [[nodiscard]] SharedPtr clone() const override
    { return std::make_shared<CustomLayerType>(*this); }
};

const std::string CustomLayerType::TYPE = "CustomType";


class TestLayer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_layer());
        EDGE_LEARNING_TEST_CALL(test_stream());
    }

private:
    void test_layer() {
        std::vector<NumType> v_empty;
        std::vector<NumType> v(std::size_t(10));
        std::vector<NumType> v_diff_size(std::size_t(11));
        EDGE_LEARNING_TEST_EXECUTE(auto l = CustomLayer());
        EDGE_LEARNING_TEST_TRY(auto l = CustomLayer());
        auto l = CustomLayer();
        EDGE_LEARNING_TEST_EQUAL(l.TYPE, "None");
        EDGE_LEARNING_TEST_EQUAL(l.type(), "None");
        EDGE_LEARNING_TEST_TRY(
            l.init(Layer::InitializationFunction::KAIMING,
                   Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l.init(Layer::InitializationFunction::KAIMING,
                   Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l.init(Layer::InitializationFunction::XAVIER,
                   Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l.init(Layer::InitializationFunction::XAVIER,
                   Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
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
        EDGE_LEARNING_TEST_EQUAL(l.name(), "custom_layer_test");
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
        EDGE_LEARNING_TEST_FAIL(l.training_forward(v_diff_size));
        EDGE_LEARNING_TEST_THROWS(l.training_forward(v_diff_size),
                                  std::runtime_error);

        EDGE_LEARNING_TEST_EXECUTE(CustomLayer l1_copy{l});
        EDGE_LEARNING_TEST_TRY(CustomLayer l2_copy{l});
        CustomLayer l_copy{l};
        EDGE_LEARNING_TEST_TRY(
            l_copy.init(Layer::InitializationFunction::KAIMING,
                        Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_copy.init(Layer::InitializationFunction::KAIMING,
                        Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_copy.init(Layer::InitializationFunction::XAVIER,
                        Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_copy.init(Layer::InitializationFunction::XAVIER,
                        Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
        EDGE_LEARNING_TEST_ASSERT(!l_copy.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_copy.last_input().size(), v.size());
        EDGE_LEARNING_TEST_TRY(l_copy.input_shape(0));
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
        EDGE_LEARNING_TEST_EQUAL(l_copy.name(), "custom_layer_test");
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
        EDGE_LEARNING_TEST_FAIL(l_copy.training_forward(v_diff_size));
        EDGE_LEARNING_TEST_THROWS(l_copy.training_forward(v_diff_size),
                                  std::runtime_error);

        EDGE_LEARNING_TEST_EXECUTE(CustomLayer l_assign; l_assign = l);
        EDGE_LEARNING_TEST_TRY(CustomLayer l_assign; l_assign = l);
        CustomLayer l_assign; l_assign = l;
        EDGE_LEARNING_TEST_TRY(
            l_assign.init(
                Layer::InitializationFunction::KAIMING,
                Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_assign.init(
                Layer::InitializationFunction::KAIMING,
                Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_assign.init(
                Layer::InitializationFunction::XAVIER,
                Layer::ProbabilityDensityFunction::NORMAL, RneType()));
        EDGE_LEARNING_TEST_TRY(
            l_assign.init(
                Layer::InitializationFunction::XAVIER,
                Layer::ProbabilityDensityFunction::UNIFORM, RneType()));
        EDGE_LEARNING_TEST_ASSERT(!l_assign.last_input().empty());
        EDGE_LEARNING_TEST_EQUAL(l_assign.last_input().size(), v.size());
        EDGE_LEARNING_TEST_TRY(l_assign.input_shape(0));
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
        EDGE_LEARNING_TEST_EQUAL(l_assign.name(), "custom_layer_test");
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
        EDGE_LEARNING_TEST_FAIL(l_assign.training_forward(v_diff_size));
        EDGE_LEARNING_TEST_THROWS(l_assign.training_forward(v_diff_size),
                                  std::runtime_error);

        EDGE_LEARNING_TEST_EXECUTE(auto l2 = CustomLayerNoName());
        EDGE_LEARNING_TEST_TRY(auto l2 = CustomLayerNoName());
        auto l_noname = CustomLayerNoName();
        EDGE_LEARNING_TEST_PRINT(l_noname.name());
        EDGE_LEARNING_TEST_ASSERT(!l_noname.name().empty());

        auto l_shape = CustomLayer(10, 20);
        EDGE_LEARNING_TEST_EQUAL(l_shape.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape.output_size(), 20);
        EDGE_LEARNING_TEST_ASSERT(l_shape.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l_shape.last_output());
        EDGE_LEARNING_TEST_THROWS(l_shape.last_output(), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_shape.training_forward(v_diff_size));
        EDGE_LEARNING_TEST_THROWS(l_shape.training_forward(v_diff_size),
                                  std::runtime_error);
        CustomLayer l_shape_copy{l_shape};
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape_copy.output_size(), 20);
        EDGE_LEARNING_TEST_ASSERT(l_shape_copy.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l_shape_copy.last_output());
        EDGE_LEARNING_TEST_THROWS(l_shape_copy.last_output(),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_shape_copy.training_forward(v_diff_size));
        EDGE_LEARNING_TEST_THROWS(l_shape_copy.training_forward(v_diff_size),
                                  std::runtime_error);
        CustomLayer l_shape_assign; l_shape_assign = l_shape;
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.input_size(), 10);
        EDGE_LEARNING_TEST_EQUAL(l_shape_assign.output_size(), 20);
        EDGE_LEARNING_TEST_ASSERT(l_shape_assign.last_input().empty());
        EDGE_LEARNING_TEST_FAIL(l_shape_assign.last_output());
        EDGE_LEARNING_TEST_THROWS(l_shape_assign.last_output(),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(l_shape_assign.training_forward(v_diff_size));
        EDGE_LEARNING_TEST_THROWS(l_shape_assign.training_forward(v_diff_size),
                                  std::runtime_error);
    }

    void test_stream()
    {
        CustomLayer l(10, 11, "layer_test_stream");

        Json l_dump;
        EDGE_LEARNING_TEST_TRY(l.dump(l_dump));
        EDGE_LEARNING_TEST_PRINT(l_dump);
        EDGE_LEARNING_TEST_EQUAL(l_dump["type"].as<std::string>(), "None");
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

        l = CustomLayer();
        EDGE_LEARNING_TEST_TRY(l.load(l_dump));
        EDGE_LEARNING_TEST_EQUAL(l.type(), "None");
        EDGE_LEARNING_TEST_EQUAL(l_dump["name"].as<std::string>(), l.name());
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[0], l.input_shape().height);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[1], l.input_shape().width);
        EDGE_LEARNING_TEST_EQUAL(input_size_arr[2], l.input_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(input_size, l.input_size());
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[0], l.output_shape().height);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[1], l.output_shape().width);
        EDGE_LEARNING_TEST_EQUAL(output_size_arr[2], l.output_shape().channels);
        EDGE_LEARNING_TEST_EQUAL(output_size, l.output_size());

        CustomLayer l_prev(12, 13, "layer_test_stream_prev");
        CustomLayer l_next(14, 15, "layer_test_stream_next");
        CustomLayer l_comp(16, 17, "layer_test_stream_comp",
                           std::make_shared<CustomLayer>(l_prev),
                           std::make_shared<CustomLayer>(l_next));
        Json l_dump_comp;
        EDGE_LEARNING_TEST_TRY(l_comp.dump(l_dump_comp));
        EDGE_LEARNING_TEST_PRINT(l_dump_comp);
        EDGE_LEARNING_TEST_EQUAL(l_dump_comp["antecedents"].size(), 1);
        EDGE_LEARNING_TEST_EQUAL(l_dump_comp["antecedents"][0], l_prev.name());
        EDGE_LEARNING_TEST_EQUAL(l_dump_comp["subsequents"].size(), 1);
        EDGE_LEARNING_TEST_EQUAL(l_dump_comp["subsequents"][0], l_next.name());

        l_comp = CustomLayer();
        EDGE_LEARNING_TEST_TRY(l_comp.load(l_dump_comp));
        auto l_comp1 = CustomLayer(16, 17, "layer_test_stream_comp",
                                   std::make_shared<CustomLayer>(l_prev),
                                   nullptr);
        EDGE_LEARNING_TEST_TRY(l_comp.load(l_dump_comp));
        auto l_comp2 = CustomLayer(16, 17, "layer_test_stream_comp",
                                   nullptr,
                                   std::make_shared<CustomLayer>(l_next));
        EDGE_LEARNING_TEST_TRY(l_comp.load(l_dump_comp));
        Json l_dump_comp_fail;
        l_comp.dump(l_dump_comp_fail);
        CustomLayerType l_comp_type;
        EDGE_LEARNING_TEST_FAIL(l_comp_type.load(l_dump_comp_fail));
        EDGE_LEARNING_TEST_THROWS(l_comp_type.load(l_dump_comp_fail),
                                  std::runtime_error);
        l_comp = CustomLayer(16, 17, "layer_test_stream_comp",
                             std::make_shared<CustomLayer>(l_next),
                             std::make_shared<CustomLayer>(l_next));
        EDGE_LEARNING_TEST_FAIL(l_comp.load(l_dump_comp_fail));
        EDGE_LEARNING_TEST_THROWS(l_comp.load(l_dump_comp_fail),
                                  std::runtime_error);
        l_comp = CustomLayer(16, 17, "layer_test_stream_comp",
                             std::make_shared<CustomLayer>(l_prev),
                             std::make_shared<CustomLayer>(l_prev));
        EDGE_LEARNING_TEST_FAIL(l_comp.load(l_dump_comp_fail));
        EDGE_LEARNING_TEST_THROWS(l_comp.load(l_dump_comp_fail),
                                  std::runtime_error);
        l_comp1.dump(l_dump_comp_fail);
        EDGE_LEARNING_TEST_FAIL(l_comp.load(l_dump_comp_fail));
        EDGE_LEARNING_TEST_THROWS(l_comp.load(l_dump_comp_fail),
                                  std::runtime_error);
        l_comp2.dump(l_dump_comp_fail);
        EDGE_LEARNING_TEST_FAIL(l_comp.load(l_dump_comp_fail));
        EDGE_LEARNING_TEST_THROWS(l_comp.load(l_dump_comp_fail),
                                  std::runtime_error);

        Json json_void;
        EDGE_LEARNING_TEST_FAIL(l.load(json_void));
        EDGE_LEARNING_TEST_THROWS(l.load(json_void), std::runtime_error);
    }
};

int main() {
    TestLayer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
