/***************************************************************************
 *            dnn/layer.cpp
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

#include "layer.hpp"

#include "model.hpp"
#include "dlmath.hpp"

#include <stdexcept>


namespace EdgeLearning {

LayerShape::LayerShape(std::vector<DLMath::Shape3d> shape_vec)
    : _shape_vec{std::move(shape_vec)}
{ }

LayerShape::LayerShape(DLMath::Shape3d shape)
    : LayerShape{std::vector<DLMath::Shape3d>{{shape}}}
{ }

LayerShape::LayerShape(SizeType size)
    : LayerShape{DLMath::Shape3d(size)}
{ }

LayerShape::LayerShape()
    : _shape_vec{}
{ }

const std::vector<DLMath::Shape3d>& LayerShape::shapes() const
{
    return _shape_vec;
}

const DLMath::Shape3d& LayerShape::shape(SizeType idx) const
{
    return _shape_vec.at(idx);
}

SizeType LayerShape::size(SizeType idx) const
{
    return shape(idx).size();
}

SizeType LayerShape::height(SizeType idx) const
{
    return shape(idx).height();
}

SizeType LayerShape::width(SizeType idx) const
{
    return shape(idx).width();
}

SizeType LayerShape::channels(SizeType idx) const
{
    return shape(idx).channels();
}

SizeType LayerShape::amount_shapes() const
{
    return _shape_vec.size();
}

const std::string Layer::TYPE = "None";
const std::map<Layer::DumpFields, std::string> Layer::dump_fields = {
    { DumpFields::TYPE,          "type"          },
    { DumpFields::NAME,          "name"          },
    { DumpFields::INPUT_SIZE,    "input_shape"   },
    { DumpFields::OUTPUT_SIZE,   "output_shape"  },
    { DumpFields::WEIGHTS,       "weights"       },
    { DumpFields::BIASES,        "biases"        },
    { DumpFields::ANTECEDENTS,   "antecedents"   },
    { DumpFields::SUBSEQUENTS,   "subsequents"   },
    { DumpFields::OTHERS,        "others"        }
};

Layer::Layer(std::string name, LayerShape input_shape, LayerShape output_shape,
             std::string prefix_name)
    : _name{std::move(name)}
    , _input_shape{input_shape}
    , _output_shape{output_shape}
    , _last_input{}
    , _last_input_size{0}
{ 
    if (_name.empty())
    {
        if (prefix_name.empty()) prefix_name = "layer_";
        _name = prefix_name + std::to_string(DLMath::unique());
    }
}

Layer::Layer(const Layer& obj)
    : _name{obj._name}
    , _input_shape{obj._input_shape}
    , _output_shape{obj._output_shape}
    , _last_input{obj._last_input}
{

}

Layer& Layer::operator=(const Layer& obj)
{
    if (this == &obj) return *this;
    _name = obj._name;
    _input_shape = obj._input_shape;
    _output_shape = obj._output_shape;
    _last_input = obj._last_input;
    return *this;
}

const std::vector<NumType>& Layer::forward(
    const std::vector<NumType>& activations)
{
    return activations;
}

const std::vector<NumType>& Layer::training_forward(
    const std::vector<NumType>& inputs)
{
    _check_training_input(inputs);
    _last_input = inputs.data();
    return forward(inputs);
}

const std::vector<NumType>& Layer::backward(
    const std::vector<NumType>& gradients)
{
    return gradients;
}

std::vector<NumType> Layer::last_input()
{
    return _last_input
           ? std::vector<NumType>{
            _last_input, _last_input + _last_input_size}
           : std::vector<NumType>{};
};

const LayerShape& Layer::input_shape() const
{
    return _input_shape;
}

void Layer::input_shape(LayerShape input_shape)
{
    _set_input_shape(std::move(input_shape));
}

const LayerShape& Layer::output_shape() const
{
    return _output_shape;
}

const std::vector<DLMath::Shape3d>& Layer::input_shapes() const
{
    return _input_shape.shapes();
}

const std::vector<DLMath::Shape3d>& Layer::output_shapes() const
{
    return _output_shape.shapes();
}

SizeType Layer::input_size() const
{
    return _input_shape.size();
}

SizeType Layer::output_size() const
{
    return _output_shape.size();
}

SizeType Layer::input_layers()
{
    return _input_shape.amount_shapes();
}

SizeType Layer::output_layers()
{
    return _output_shape.amount_shapes();
}

void Layer::dump(Json& out) const
{
    out[dump_fields.at(DumpFields::TYPE)] = type();
    out[dump_fields.at(DumpFields::NAME)] = _name;

    Json input_shape;
    for (const auto& shape: _input_shape.shapes())
    {
        std::vector<SizeType> input_size = {
            shape.height(), shape.width(), shape.channels()
        };
        input_shape.append(Json(input_size));
    }
    out[dump_fields.at(DumpFields::INPUT_SIZE)] = Json(input_shape);

    Json output_shape;
    for (const auto& shape: _output_shape.shapes())
    {
        std::vector<SizeType> output_size = {
            shape.height(), shape.width(), shape.channels()
        };
        output_shape.append(Json(output_size));
    }
    out[dump_fields.at(DumpFields::OUTPUT_SIZE)] = Json(output_shape);
}

void Layer::load(Json& in)
{
    if (in.json_type() == JsonObject::JsonType::NONE)
    {
        throw std::runtime_error("No well-formed JSON");
    }

    auto t = in[dump_fields.at(DumpFields::TYPE)];
    if (t.as<std::string>() != type())
    {
        throw std::runtime_error(
            "The current layer of type " + type() +
            " do not correspond with loaded type " + std::string(t));
    }

    _name = std::string(in[dump_fields.at(DumpFields::NAME)]);

    std::vector<DLMath::Shape3d> input_shapes;
    auto input_shapes_json = in[dump_fields.at(DumpFields::INPUT_SIZE)];
    for (SizeType i = 0; i < input_shapes_json.size(); ++i)
    {
        auto shape = input_shapes_json[i];
        input_shapes.emplace_back(shape[0], shape[1], shape[2]);
    }
    _input_shape = LayerShape(input_shapes);

    std::vector<DLMath::Shape3d> output_shapes;
    auto output_shapes_json = in[dump_fields.at(DumpFields::INPUT_SIZE)];
    for (SizeType i = 0; i < output_shapes_json.size(); ++i)
    {
        auto shape = output_shapes_json[i];
        output_shapes.emplace_back(shape[0], shape[1], shape[2]);
    }
    _output_shape = LayerShape(output_shapes);
}

void Layer::_set_input_shape(LayerShape input_shape)
{
    _input_shape = std::move(input_shape);
}

void Layer::_check_training_input(const std::vector<NumType>& inputs)
{
    if (input_size() == 0)
    {
        input_shape(inputs.size());
    }
    else if (input_size() != inputs.size())
    {
        throw std::runtime_error(
            "Training forward input catch an unpredicted input size: "
            + std::to_string(input_size())
            + " != " + std::to_string(inputs.size()));
    }
}

} // namespace EdgeLearning
