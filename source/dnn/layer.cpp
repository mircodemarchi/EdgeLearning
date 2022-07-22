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

const std::string Layer::TYPE = "None";
const std::map<Layer::DumpFields, std::string> Layer::dump_fields = {
    { DumpFields::TYPE,          "type"          },
    { DumpFields::NAME,          "name"          },
    { DumpFields::INPUT_SIZE,    "input_size"    },
    { DumpFields::OUTPUT_SIZE,   "output_size"   },
    { DumpFields::WEIGHTS,       "weights"       },
    { DumpFields::BIASES,        "biases"        },
    { DumpFields::ANTECEDENTS,   "antecedents"   },
    { DumpFields::SUBSEQUENTS,   "subsequents"   },
    { DumpFields::OTHERS,        "others"        }
};

Layer::Layer(DLMath::Shape3d input_shape, DLMath::Shape3d output_shape,
             std::string name, std::string prefix_name)
    : _name{std::move(name)}
    , _antecedents{}
    , _subsequents{}
    , _input_shape{input_shape}
    , _output_shape{output_shape}
    , _last_input{}
{ 
    if (_name.empty())
    {
        if (prefix_name.empty()) prefix_name = "layer_";
        _name = prefix_name + std::to_string(DLMath::unique());
    }
}

Layer::Layer(const Layer& obj)
    : _name{obj._name}
    , _antecedents{obj._antecedents}
    , _subsequents{obj._subsequents}
    , _input_shape{obj._input_shape}
    , _output_shape{obj._output_shape}
    , _last_input{obj._last_input}
{

}

Layer& Layer::operator=(const Layer& obj)
{
    if (this == &obj) return *this;
    _name = obj._name;
    _antecedents = obj._antecedents;
    _subsequents = obj._subsequents;
    _input_shape = obj._input_shape;
    _output_shape = obj._output_shape;
    _last_input = obj._last_input;
    return *this;
}

const std::vector<NumType>& Layer::forward(
    const std::vector<NumType>& activations)
{
    // Forward to the next layers.
    for (const auto& l: this->_subsequents)
    {
        l->forward(activations);
    }
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
    for (const auto& l: _antecedents)
    {
        l->backward(gradients);
    }
    return gradients;
}

const DLMath::Shape3d & Layer::input_shape() const
{
    return _input_shape;
}

void Layer::input_shape(DLMath::Shape3d input_shape)
{
    _input_shape = input_shape;
}

const DLMath::Shape3d& Layer::output_shape() const
{
    return _output_shape;
}

SizeType Layer::input_size() const
{
    return _input_shape.size();
}

SizeType Layer::output_size() const
{
    return _output_shape.size();
}

std::vector<Layer::SharedPtr> Layer::subsequent_layers()
{
    return _subsequents;
}

std::vector<Layer::SharedPtr> Layer::antecedent_layers()
{
    return _antecedents;
}

SizeType Layer::input_layers()
{
    return _antecedents.size();
}

SizeType Layer::output_layers()
{
    return _subsequents.size();
}

void Layer::dump(Json& out) const
{
    out[dump_fields.at(DumpFields::TYPE)] = type();
    out[dump_fields.at(DumpFields::NAME)] = _name;
    std::vector<SizeType> input_size = {
        _input_shape.height,
        _input_shape.width,
        _input_shape.channels
    };
    out[dump_fields.at(DumpFields::INPUT_SIZE)] = Json(input_size);
    std::vector<SizeType> output_size = {
        _output_shape.height,
        _output_shape.width,
        _output_shape.channels
    };
    out[dump_fields.at(DumpFields::OUTPUT_SIZE)] = Json(output_size);

    std::vector<std::string> antecedent_names;
    for (const auto& antecedent : _antecedents)
    {
        antecedent_names.push_back(antecedent->name());
    }
    out[dump_fields.at(DumpFields::ANTECEDENTS)] = Json(antecedent_names);

    std::vector<std::string> subsequent_names;
    for (const auto& subsequent : _subsequents)
    {
        subsequent_names.push_back(subsequent->name());
    }
    out[dump_fields.at(DumpFields::SUBSEQUENTS)] = Json(subsequent_names);
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
    _input_shape = DLMath::Shape3d(
        in[dump_fields.at(DumpFields::INPUT_SIZE)][0].as<SizeType>(),
        in[dump_fields.at(DumpFields::INPUT_SIZE)][1].as<SizeType>(),
        in[dump_fields.at(DumpFields::INPUT_SIZE)][2].as<SizeType>()
    );
    _output_shape = DLMath::Shape3d(
        in[dump_fields.at(DumpFields::OUTPUT_SIZE)][0].as<SizeType>(),
        in[dump_fields.at(DumpFields::OUTPUT_SIZE)][1].as<SizeType>(),
        in[dump_fields.at(DumpFields::OUTPUT_SIZE)][2].as<SizeType>()
    );

    auto antecedent_layer_names =
        in.at(dump_fields.at(DumpFields::ANTECEDENTS)).as_vec<std::string>();
    for (const auto& antecedent : _antecedents)
    {
        if (std::find(antecedent_layer_names.begin(),
                      antecedent_layer_names.end(),
                      antecedent->name()) == antecedent_layer_names.end())
        {
            throw std::runtime_error(
                "The loaded json " + _name + " of type " + type() +
                " do not have antecedent layer " + antecedent->name());
        }
    }

    auto subsequent_layer_names =
        in.at(dump_fields.at(DumpFields::SUBSEQUENTS)).as_vec<std::string>();
    for (const auto& subsequent : _subsequents)
    {
        if (std::find(subsequent_layer_names.begin(),
                      subsequent_layer_names.end(),
                      subsequent->name()) == subsequent_layer_names.end())
        {
            throw std::runtime_error(
                "The loaded json " + _name + " of type " + type() +
                " do not have subsequent layer " + subsequent->name());
        }
    }
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
