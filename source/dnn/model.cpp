/***************************************************************************
 *            model.cpp
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

#include "model.hpp"

#include "dlmath.hpp"

#include <cstdio>
#include <cassert>

namespace EdgeLearning {

Model::Model(std::string name)
    : _name{std::move(name)}
    , _layers{}
    , _loss_layer{}
{ 
    if (_name.empty())
    {
        _name = "model_" + std::to_string(DLMath::unique());
    }
}

Model::Model(const Model& obj)
    : _name{obj._name}
    , _layers{obj._layers}
    , _loss_layer{obj._loss_layer}
{

}

Model& Model::operator=(Model obj)
{
    swap(*this, obj);
    return *this;
}

void swap(Model& lop, Model& rop)
{
    using std::swap;
    swap(lop._name, rop._name);
    swap(lop._layers, rop._layers);
    swap(lop._loss_layer, rop._loss_layer);
}

void Model::create_edge(Layer* src, Layer* dst)
{
    // NOTE: No validation is done to ensure the edge doesn't already exist
    dst->_antecedents.push_back(src);
    src->_subsequents.push_back(dst);
}

RneType::result_type Model::init(RneType::result_type seed)
{
    if (seed == 0)
    {
        // Generate a new random seed from the host random device.
        std::random_device rd{};
        seed = rd();
    }
    std::cout << "Initializing model parameters with seed: " << seed << "\n";

    RneType rne{seed};  
    for (auto& layer: _layers)
    {
        layer->init(rne);
    }
    _loss_layer->init(rne);

    return seed;
}

void Model::train(Optimizer& optimizer)
{
    for (auto& layer: _layers)
    {
        optimizer.train(*layer);
    }
    optimizer.train(*_loss_layer);
    _loss_layer->reset_score();
}

void Model::step(NumType* input, const NumType* target)
{
    _loss_layer->set_target(target);
    _layers[0]->forward(input); //< TODO: how many input layers there are?
    _loss_layer->reverse();
}

void Model::print() const
{
    for (auto& layer: _layers)
    {
        layer->print();
    }
    _loss_layer->print();
}

NumType Model::accuracy() const
{
    return _loss_layer->accuracy();
}

NumType Model::avg_loss() const
{
    return _loss_layer->avg_loss();
}

void Model::save(std::ofstream& out)
{
    for (auto& layer: _layers)
    {
        SizeType param_count = layer->param_count();
        for (SizeType i = 0; i < param_count; ++i)
        {
            out.write(reinterpret_cast<char const*>(
                layer->param(i)), sizeof(NumType));
        }
    }
    SizeType param_count = _loss_layer->param_count();
    for (SizeType i = 0; i < param_count; ++i)
    {
        out.write(reinterpret_cast<char const*>(
            _loss_layer->param(i)), sizeof(NumType));
    }
}

void Model::load(std::ifstream& in)
{
    for (auto& layer: _layers)
    {
        SizeType param_count = layer->param_count();
        for (SizeType i = 0; i < param_count; ++i)
        {
            in.read(reinterpret_cast<char*>(layer->param(i)), sizeof(NumType));
        }
    }
    SizeType param_count = _loss_layer->param_count();
    for (SizeType i = 0; i < param_count; ++i)
    {
        in.read(reinterpret_cast<char*>(
            _loss_layer->param(i)), sizeof(NumType));
    }
}

} // namespace EdgeLearning
