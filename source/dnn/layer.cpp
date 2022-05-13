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


namespace EdgeLearning {

const std::string Layer::Type = "None";

Layer::Layer(Model& model, SizeType input_size, SizeType output_size,
             Activation activation, std::string name, std::string prefix_name)
    : _model(model)
    , _name{std::move(name)}
    , _antecedents{}
    , _subsequents{}
    , _input_size{input_size}
    , _output_size{output_size}
    , _activation{activation}
    , _last_input{}
{ 
    if (_name.empty())
    {
        if (prefix_name.empty()) prefix_name = "layer_";
        _name = prefix_name + std::to_string(DLMath::unique());
    }
}

Layer::Layer(const Layer& obj)
    : _model{obj._model}
    , _name{obj._name}
    , _antecedents{obj._antecedents}
    , _subsequents{obj._subsequents}
    , _input_size{obj._input_size}
    , _output_size{obj._output_size}
    , _last_input{obj._last_input}
{

}

Layer& Layer::operator=(const Layer& obj)
{
    if (this == &obj) return *this;
    _model = obj._model;
    _name = obj._name;
    _antecedents = obj._antecedents;
    _subsequents = obj._subsequents;
    _input_size = obj._input_size;
    _output_size = obj._output_size;
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

const std::vector<NumType>& Layer::backward(
    const std::vector<NumType>& gradients)
{
    for (const auto& l: _antecedents)
    {
        l->backward(gradients);
    }
    return gradients;
}

SizeType Layer::input_size() const
{
    return _input_size;
}

void Layer::input_size(DLMath::Shape3d input_size)
{
    _input_size = input_size.size();
}

SizeType Layer::output_size() const
{
    return _output_size;
}

} // namespace EdgeLearning
