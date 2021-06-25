/***************************************************************************
 *            model.cpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne.
 *
 *  Ariadne is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "model.hpp"

#include <cstdio>
#include <cassert>

namespace Ariadne {

Model::Model(std::string name)
    : _name{name}
    , _layers{}
{ }

void Model::create_edge(Layer& dst, Layer& src)
{
    // NOTE: No validation is done to ensure the edge doesn't already exist
    dst._antecedents.push_back(&src);
    src._subsequents.push_back(&dst);
}

RneType::result_type Model::init(RneType::result_type seed)
{
    if (seed == 0)
    {
        // Generate a new random seed from the host random device.
        std::random_device rd{};
        seed = rd();
    }
    std::printf("Initializing model parameters with seed: %lu\n", seed);

    RneType rne{seed};
    for (auto* layer: _layers)
    {
        layer->init(rne);
    }

    return seed;
}

void Model::train(Optimizer& optimizer)
{
    for (auto& layer: _layers)
    {
        optimizer.train(*layer);
    }
}

void Model::print() const
{
    for (auto& layer: _layers)
    {
        layer->print();
    }
}

void Model::save(std::ofstream& out)
{
    for (auto& layer: _layers)
    {
        size_t param_count = layer->param_count();
        for (size_t i = 0; i < param_count; ++i)
        {
            out.write(reinterpret_cast<char const*>(layer->param(i)), 
                sizeof(NumType));
        }
    }
}

void Model::load(std::ifstream& in)
{
    for (auto& layer: _layers)
    {
        size_t param_count = layer->param_count();
        for (size_t i = 0; i < param_count; ++i)
        {
            in.read(reinterpret_cast<char*>(layer->param(i)), sizeof(NumType));
        }
    }
}

} // namespace Ariadne
