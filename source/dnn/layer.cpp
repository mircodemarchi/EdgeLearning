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

Layer::Layer(Model& model, std::string name)
    : _model(model)
    , _name{std::move(name)}
{ 
    if (_name.empty())
    {
        _name = "layer_" + std::to_string(DLMath::unique());
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
