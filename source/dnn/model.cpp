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
{ 
    if (_name.empty())
    {
        _name = "model_" + std::to_string(DLMath::unique());
    }
}

Model::Model(const Model& obj)
    : _name{obj._name}
    , _state(obj._state)
{}

Model& Model::operator=(Model obj)
{
    swap(*this, obj);
    return *this;
}

void swap(Model& lop, Model& rop)
{
    using std::swap;
    swap(lop._name, rop._name);
    swap(lop._state, rop._state);
}

void Model::create_back_arc(
    const Layer::SharedPtr& src, const Layer::SharedPtr& dst)
{
    _state.graph.add_arc_backward(dst, src);
    _state.update();
}

void Model::create_front_arc(
    const Layer::SharedPtr& src, const Layer::SharedPtr& dst)
{
    _state.graph.add_arc_forward(src, dst);
    _state.update();
}

void Model::create_front_arc(
    const Layer::SharedPtr& src, const std::shared_ptr<LossLayer>& dst)
{
    _state.graph.add_arc_forward(src, dst);
    _state.update();
}

void Model::create_edge(
    const Layer::SharedPtr& src, const Layer::SharedPtr& dst)
{
    // NOTE: No validation is done to ensure the edge doesn't already exist
    create_back_arc(src, dst);
    create_front_arc(src, dst);
}

void Model::create_loss_edge(
    const Layer::SharedPtr& src, const std::shared_ptr<LossLayer>& dst)
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
    // std::cout << "Initializing model parameters with seed: " << seed << "\n";

    RneType rne{seed};
    for (const auto& layer_idx: _state.graph.forward_layers_idx())
    {
        auto layer = _state.layers[layer_idx];
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
                bool init_done = false;
                for (const auto& next_layer_idx: _state.graph.forward(layer_idx))
                {
                    if (_state.layers[next_layer_idx]->is_type<ReluLayer>())
                    {
                        layer->init(Layer::InitializationFunction::KAIMING,
                                    pdf, rne);
                        init_done = true;
                        break;
                    }
                }

                if (!init_done)
                {
                    layer->init(Layer::InitializationFunction::XAVIER,
                                pdf, rne);
                }
                break;
            }

        }
    }

    return seed;
}

void Model::train(Optimizer& optimizer)
{
    train(optimizer, *this);
}

void Model::train(Optimizer& optimizer, Model& model_from)
{
    for (const auto& layer: model_from._state.layers)
    {
        optimizer.train(*layer);
    }
}

void Model::reset_score()
{
    for (const auto& loss_layer: _state.loss_layers)
    {
        loss_layer->reset_score();
    }
}

void Model::step(const std::vector<NumType>& input,
                 const std::vector<NumType>& target)
{
    const std::vector<NumType> not_used;

    // Set target.
    for (const auto& loss_layer: _state.loss_layers)
    {
        loss_layer->set_target(target);
    }

    // Forward.
    for (const auto& input_layer: _state.input_layers)
    {
        input_layer->training_forward(input);
    }
    for (const auto& forward_arc: _state.training_forward_run)
    {
        forward_arc.to->training_forward(forward_arc.from->last_output());
    }

    // Backward.
    for (const auto& loss_layer: _state.loss_layers)
    {
        loss_layer->backward(not_used);
    }
    for (const auto& backward_arc: _state.backward_run)
    {
        backward_arc.to->backward(backward_arc.from->last_input_gradient());
    }
}

const std::vector<NumType>& Model::predict(const std::vector<NumType>& input)
{
    if (_state.output_layers.empty())
    {
        throw std::runtime_error("No output layers in model");
    }
    for (const auto& input_layer: _state.input_layers)
    {
        input_layer->forward(input);
    }
    for (const auto& forward_arc: _state.forward_run)
    {
        forward_arc.to->forward(forward_arc.from->last_output());
    }
    return _state.output_layers.front()->last_output();
}

SizeType Model::input_size(SizeType input_layer_idx)
{
    if (input_layer_idx >= _state.input_layers.size()
        || !_state.input_layers[input_layer_idx])
    {
        return 0;
    }
    return _state.input_layers[input_layer_idx]->input_size();
}

SizeType Model::output_size(SizeType output_layer_idx)
{
    if (output_layer_idx >= _state.output_layers.size()
        || !_state.output_layers[output_layer_idx])
    {
        return 0;
    }
    return _state.output_layers[output_layer_idx]->output_size();
}

const std::vector<Layer::SharedPtr>& Model::layers() const
{
    return _state.layers;
}

const std::vector<Layer::SharedPtr>& Model::input_layers() const
{
    return _state.input_layers;
}

const std::vector<Layer::SharedPtr>& Model::output_layers() const
{
    return _state.output_layers;
}

const std::vector<std::shared_ptr<LossLayer>>& Model::loss_layers() const
{
    return _state.loss_layers;
}

[[nodiscard]] std::string const& Model::name() const noexcept
{
    return _name;
}

void Model::print() const
{
    for (auto& layer: _state.layers)
    {
        layer->print();
    }
}

NumType Model::accuracy() const
{
    NumType sum = 0.0;
    for (const auto& loss_layer: _state.loss_layers) {
        sum += loss_layer->accuracy();
    }
    return sum / _state.loss_layers.size();
}

NumType Model::avg_loss() const
{
    NumType sum = 0.0;
    for (const auto& loss_layer: _state.loss_layers) {
        sum += loss_layer->avg_loss();
    }
    return sum / _state.loss_layers.size();
}

void Model::dump(std::ofstream& out)
{
    Json model;
    model["name"] = _name;

    Json layers_json;
    for (auto& layer: _state.layers)
    {
        Json layer_json;
        layer->dump(layer_json);
        layers_json.append(layer_json);
    }
    model["layer"] = layers_json;
    out << model;
}

void Model::load(std::ifstream& in)
{
    Json model;
    in >> model;

    _name = model["name"].as<std::string>();
    for (std::size_t l_i = 0; l_i < _state.layers.size(); ++l_i)
    {
        auto layer_json = Json(model["layer"][l_i]);
        _state.layers[l_i]->load(layer_json);
    }
}

} // namespace EdgeLearning
