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

#include "dlmath.hpp"
#include "dlgraph.hpp"

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
    , _input_size{input_shape.size()}
    , _output_shape{output_shape}
    , _output_size{output_shape.size()}
    , _last_input{}
{ 
    if (_name.empty())
    {
        if (prefix_name.empty()) prefix_name = "layer_";
        _name = prefix_name + std::to_string(DLMath::unique());
    }
}

const std::vector<NumType>& Layer::forward(
    const std::vector<NumType>& activations)
{
    return activations;
}

const std::vector<NumType>& Layer::training_forward(
    const std::vector<NumType>& inputs)
{
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
           ? std::vector<NumType>{_last_input, _last_input + _input_size}
           : std::vector<NumType>{};
}

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

SizeType Layer::input_size(SizeType input_idx) const
{
    return _input_shape.size(input_idx);
}

SizeType Layer::output_size(SizeType output_idx) const
{
    return _output_shape.size(output_idx);
}

SizeType Layer::input_layers()
{
    return _input_shape.amount_shapes();
}

SizeType Layer::output_layers()
{
    return _output_shape.amount_shapes();
}

Json Layer::dump() const
{
    Json out;
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
    return out;
}

void Layer::load(const Json& in)
{
    if (in.json_type() == JsonObject::JsonType::NONE)
    {
        throw std::runtime_error("No well-formed JSON");
    }

    auto t = in.at(dump_fields.at(DumpFields::TYPE));
    if (t.as<std::string>() != type())
    {
        throw std::runtime_error(
            "The current layer of type " + type() +
            " do not correspond with loaded type " + std::string(t));
    }

    _name = std::string(in.at(dump_fields.at(DumpFields::NAME)));

    std::vector<DLMath::Shape3d> input_shapes;
    auto input_shapes_json = in.at(dump_fields.at(DumpFields::INPUT_SIZE));
    for (SizeType i = 0; i < input_shapes_json.size(); ++i)
    {
        auto shape = input_shapes_json[i];
        input_shapes.emplace_back(shape[0], shape[1], shape[2]);
    }
    _input_shape = LayerShape(input_shapes);
    _input_size = _input_shape.size();

    std::vector<DLMath::Shape3d> output_shapes;
    auto output_shapes_json = in.at(dump_fields.at(DumpFields::OUTPUT_SIZE));
    for (SizeType i = 0; i < output_shapes_json.size(); ++i)
    {
        auto shape = output_shapes_json[i];
        output_shapes.emplace_back(shape[0], shape[1], shape[2]);
    }
    _output_shape = LayerShape(output_shapes);
    _output_size = _output_shape.size();
}

void Layer::_set_input_shape(LayerShape input_shape)
{
    _input_shape = std::move(input_shape);
    _input_size = _input_shape.size();
}

} // namespace EdgeLearning
