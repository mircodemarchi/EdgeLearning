/***************************************************************************
 *            layer.cpp
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

Layer::Layer(Model& model, SizeType input_size, SizeType output_size,
             std::string name, std::string prefix_name)
    : _model(model)
    , _name{std::move(name)}
    , _input_size{input_size}
    , _output_size{output_size}
    , _last_input{nullptr}
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
{

}

Layer& Layer::operator=(const Layer& obj)
{
    if (this == &obj) return *this;
    _model = obj._model;
    _name = obj._name;
    _antecedents = obj._antecedents;
    _subsequents = obj._subsequents;
    return *this;
}

} // namespace EdgeLearning
