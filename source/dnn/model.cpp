/***************************************************************************
 *            dnn/model.cpp
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
#include "activation.hpp"
#include "parser/json.hpp"

#include <algorithm>

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
    , _layers{obj._layers.size()}
    , _loss_layer{}
{
    if (obj._loss_layer)
    {
        _loss_layer = std::dynamic_pointer_cast<LossLayer>(
            obj._loss_layer->clone());
    }
    for (std::size_t i = 0; i < obj._layers.size(); ++i)
    {
        _layers[i] = obj._layers[i]->clone();
    }
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

void Model::create_back_arc(
    const Layer::SharedPtr& src, const Layer::SharedPtr& dst)
{
    (void) src;
    (void) dst;
    // dst->_antecedents.push_back(src);
}

void Model::create_front_arc(
    const Layer::SharedPtr& src, const Layer::SharedPtr& dst)
{
    (void) src;
    (void) dst;
    // src->_subsequents.push_back(dst);
}

void Model::create_edge(
    const Layer::SharedPtr& src, const Layer::SharedPtr& dst)
{
    // NOTE: No validation is done to ensure the edge doesn't already exist
    create_back_arc(src, dst);
    create_front_arc(src, dst);
}

RneType::result_type Model::init(InitializationFunction init,
                                 ProbabilityDensityFunction pdf,
                                 RneType::result_type seed)
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
        switch (init)
        {
            case InitializationFunction::KAIMING:
            {
                layer->init(Layer::InitializationFunction::KAIMING, pdf, rne);
                break;
            }
            case InitializationFunction::XAVIER:
            {
                layer->init(Layer::InitializationFunction::XAVIER, pdf, rne);
                break;
            }
            case InitializationFunction::AUTO:
            default:
            {
                bool init_done = false;/*
                for (auto const& next_layer: layer->_subsequents)
                {
                    if (next_layer->is_type<ReluLayer>())
                    {
                        layer->init(Layer::InitializationFunction::KAIMING,
                                    pdf, rne);
                        init_done = true;
                        break;
                    }
                }*/
                if (!init_done)
                {
                    layer->init(Layer::InitializationFunction::XAVIER,
                                pdf, rne);
                }
                break;
            }

        }
    }
    _loss_layer->init(Layer::InitializationFunction::XAVIER, pdf, rne);

    return seed;
}

void Model::train(Optimizer& optimizer)
{
    train(optimizer, *this);
}

void Model::train(Optimizer& optimizer, Model& model_from)
{
    for (const auto& layer: model_from._layers)
    {
        optimizer.train(*layer);
    }
    optimizer.train(*model_from._loss_layer);
    model_from._loss_layer->reset_score();
}

void Model::step(const std::vector<NumType>& input,
                 const std::vector<NumType>& target)
{
    _loss_layer->set_target(target);
    // TODO: how many input layers there are?
    // TODO: manage multiple input layers.
    _layers.front()->training_forward(input);
    /*
    for(const auto& l: _loss_layer->_antecedents)
    {
        _loss_layer->training_forward(l->last_output());
    }*/
    const std::vector<NumType> not_used;
    _loss_layer->backward(not_used);
}

const std::vector<NumType>& Model::predict(const std::vector<NumType>& input)
{
    _layers.front()->forward(input);
    return _layers.back()->last_output();
}

SizeType Model::input_size()
{
    return _layers.front()->input_size();
}

SizeType Model::output_size()
{
    return _layers.back()->output_size();
}

const std::vector<Layer::SharedPtr>& Model::layers() const
{
    return _layers;
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

void Model::dump(std::ofstream& out)
{
    Json model;
    model["name"] = _name;

    Json layers_json;
    for (auto& layer: _layers)
    {
        Json layer_json;
        layer->dump(layer_json);
        layers_json.append(layer_json);
    }
    model["layer"] = layers_json;

    Json loss_layers_json;
    Json loss_layer_json;
    _loss_layer->dump(loss_layer_json);
    loss_layers_json.append(loss_layer_json);
    model["loss_layer"] = loss_layers_json;
    out << model;
}

void Model::load(std::ifstream& in)
{
    Json model;
    in >> model;

    _name = model["name"].as<std::string>();
    for (std::size_t l_i = 0; l_i < _layers.size(); ++l_i)
    {
        auto layer_json = Json(model["layer"][l_i]);
        _layers[l_i]->load(layer_json);
    }

    auto loss_layer_json = Json(model["loss_layer"][0]);
    _loss_layer->load(loss_layer_json);
}

std::int64_t Model::_index_of(Layer::SharedPtr l)
{
    auto itr = std::find(_layers.begin(), _layers.end(), l);
    if (itr != _layers.cend())
    {
        return std::distance(_layers.begin(), itr);
    }
    else
    {
        return -1;
    }
}

} // namespace EdgeLearning
