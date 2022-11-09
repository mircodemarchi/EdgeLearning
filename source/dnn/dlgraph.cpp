/***************************************************************************
 *            dnn/dlgraph.cpp
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

#include "dlgraph.hpp"


namespace EdgeLearning {

DLGraph::DLGraph()
    : _layers()
    , _forward_graph(_layers)
    , _training_forward_graph(_layers)
    , _backward_graph(_layers)
    , _loss_layers_idx()
    , _input_layers_idx()
    , _output_layers_idx()
{ }

DLGraph::DLGraph(const DLGraph& obj)
    : _layers()
    , _forward_graph(_layers)
    , _training_forward_graph(_layers)
    , _backward_graph(_layers)
    , _loss_layers_idx(obj._loss_layers_idx)
    , _input_layers_idx()
    , _output_layers_idx()
{
    for(const auto& l: obj._layers)
    {
        _layers.push_back(l->clone());
    }
    _forward_graph._edges = obj._forward_graph._edges;
    _training_forward_graph._edges = obj._training_forward_graph._edges;
    _backward_graph._edges = obj._backward_graph._edges;
    _compute_input_layers();
    _compute_output_layers();
}

DLGraph& DLGraph::operator=(const DLGraph& obj)
{
    if (this == &obj) return *this;
    _loss_layers_idx = obj._loss_layers_idx;

    _layers.clear();
    for(const auto& l: obj._layers)
    {
        _layers.push_back(l->clone());
    }

    _forward_graph = Graph(_layers);
    _training_forward_graph = Graph(_layers);
    _backward_graph = Graph(_layers);
    _forward_graph._edges = obj._forward_graph._edges;
    _training_forward_graph._edges = obj._training_forward_graph._edges;
    _backward_graph._edges = obj._backward_graph._edges;
    _compute_input_layers();
    _compute_output_layers();
    return *this;
}

void DLGraph::add_node(std::shared_ptr<Layer> layer)
{
    if (std::find(_layers.begin(), _layers.end(), layer)
        != _layers.cend())
    {
        return;
    }

    _layers.push_back(layer);
}

void DLGraph::add_edge(std::shared_ptr<Layer> from, std::shared_ptr<Layer> to)
{
    add_edge(std::vector<std::shared_ptr<Layer>>({from}), to);
}

void DLGraph::add_edge(std::shared_ptr<Layer> from, std::shared_ptr<LossLayer> to)
{
    add_edge(std::vector<std::shared_ptr<Layer>>({from}), to);
}

void DLGraph::add_arc_forward(std::shared_ptr<Layer> from, std::shared_ptr<Layer> to)
{
    add_arc_forward(std::vector<std::shared_ptr<Layer>>({from}), to);
}

void DLGraph::add_arc_forward(std::shared_ptr<Layer> from, std::shared_ptr<LossLayer> to)
{
    add_arc_forward(std::vector<std::shared_ptr<Layer>>({from}), to);
}

void DLGraph::add_arc_backward(std::shared_ptr<Layer> from, std::shared_ptr<Layer> to)
{
    add_arc_backward(std::vector<std::shared_ptr<Layer>>({from}), to);
}

void DLGraph::add_edge(
    std::vector<std::shared_ptr<Layer>> froms, std::shared_ptr<Layer> to)
{
    add_arc_forward(froms, to);
    add_arc_backward(to, froms);
}

void DLGraph::add_edge(
    std::vector<std::shared_ptr<Layer>> froms, std::shared_ptr<LossLayer> to)
{
    add_arc_forward(froms, to);
    add_arc_backward(to, froms);
}

void DLGraph::add_arc_forward(
    std::vector<std::shared_ptr<Layer>> froms, std::shared_ptr<Layer> to)
{
    for (const auto& from: froms)
    {
        _forward_graph.add_arc(from, to);
        _training_forward_graph.add_arc(from, to);
    }
    _compute_input_layers();
    _compute_output_layers();
}

void DLGraph::add_arc_forward(
    std::vector<std::shared_ptr<Layer>> froms, std::shared_ptr<LossLayer> to)
{
    for (const auto& from: froms)
    {
        _training_forward_graph.add_arc(from, to);
    }
}

void DLGraph::add_arc_backward(
    std::vector<std::shared_ptr<Layer>> froms, std::shared_ptr<Layer> to)
{
    for (const auto& from: froms)
    {
        _backward_graph.add_arc(from, to);
    }
}

void DLGraph::add_edge(
    std::shared_ptr<Layer> from, std::vector<std::shared_ptr<Layer>> tos)
{
    add_arc_forward(from, tos);
    add_arc_backward(tos, from);
}

void DLGraph::add_arc_forward(
    std::shared_ptr<Layer> from, std::vector<std::shared_ptr<Layer>> tos)
{
    for (const auto& to: tos)
    {
        _forward_graph.add_arc(from, to);
        _training_forward_graph.add_arc(from, to);
    }
    _compute_input_layers();
    _compute_output_layers();
}

void DLGraph::add_arc_backward(
    std::shared_ptr<Layer> from, std::vector<std::shared_ptr<Layer>> tos)
{
    for (const auto& to: tos)
    {
        _backward_graph.add_arc(from, to);
    }
}

void DLGraph::add_loss(std::shared_ptr<LossLayer> layer)
{
    add_node(layer);
    _loss_layers_idx.push_back(_layers.size() - 1);
}

bool DLGraph::has_training_forward(std::size_t layer_idx) const
{
    return _training_forward_graph.has_successors(layer_idx);
}

bool DLGraph::has_training_forward_predecessors(std::size_t layer_idx) const
{
    return _training_forward_graph.has_predecessors(layer_idx);
}

bool DLGraph::has_forward(std::size_t layer_idx) const
{
    return _forward_graph.has_successors(layer_idx);
}

bool DLGraph::has_forward_predecessors(std::size_t layer_idx) const
{
    return _forward_graph.has_predecessors(layer_idx);
}

bool DLGraph::has_backward(std::size_t layer_idx) const
{
    return _backward_graph.has_successors(layer_idx);
}

bool DLGraph::has_backward_predecessors(std::size_t layer_idx) const
{
    return _backward_graph.has_predecessors(layer_idx);
}

std::set<std::size_t> DLGraph::training_forward(std::size_t layer_idx) const
{
    return _training_forward_graph.successors(layer_idx);
}

std::set<std::size_t> DLGraph::training_forward_predecessors(std::size_t layer_idx) const
{
    return _training_forward_graph.predecessors(layer_idx);
}

std::set<std::size_t> DLGraph::forward(std::size_t layer_idx) const
{
    return _forward_graph.successors(layer_idx);
}

std::set<std::size_t> DLGraph::forward_predecessors(std::size_t layer_idx) const
{
    return _forward_graph.predecessors(layer_idx);
}

std::set<std::size_t> DLGraph::backward(std::size_t layer_idx) const
{
    return _backward_graph.successors(layer_idx);
}

std::set<std::size_t> DLGraph::backward_predecessors(std::size_t layer_idx) const
{
    return _backward_graph.predecessors(layer_idx);
}

std::vector<std::int64_t> DLGraph::training_forward_adjacent_matrix() const
{
    return _training_forward_graph.adjacent_matrix();
}

std::vector<std::int64_t> DLGraph::forward_adjacent_matrix() const
{
    return _forward_graph.adjacent_matrix();
}

std::vector<std::int64_t> DLGraph::backward_adjacent_matrix() const
{
    return _backward_graph.adjacent_matrix();
}

const std::vector<std::shared_ptr<Layer>>& DLGraph::layers() const
{
    return _layers;
}

std::vector<SizeType> DLGraph::layers_idx() const
{
    std::vector<SizeType> ret(_layers.size());
    for (std::size_t l_idx = 0; l_idx < _layers.size(); ++l_idx)
    {
        ret[l_idx] = l_idx;
    }
    return ret;
}

std::vector<std::shared_ptr<Layer>> DLGraph::training_forward_layers() const
{
    return layers();
}

std::vector<SizeType> DLGraph::training_forward_layers_idx() const
{
    return layers_idx();
}

std::vector<std::shared_ptr<Layer>> DLGraph::backward_layers() const
{
    return layers();
}

std::vector<SizeType> DLGraph::backward_layers_idx() const
{
    return layers_idx();
}

std::vector<std::shared_ptr<Layer>> DLGraph::input_layers() const
{
    std::vector<std::shared_ptr<Layer>> ret;
    for (const auto& l_idx: _input_layers_idx)
    {
        ret.push_back(_layers[l_idx]);
    }
    return ret;
}

const std::vector<SizeType>& DLGraph::input_layers_idx() const
{
    return _input_layers_idx;
}

std::vector<std::shared_ptr<Layer>> DLGraph::output_layers() const
{
    std::vector<std::shared_ptr<Layer>> ret;
    for (const auto& l_idx: _output_layers_idx)
    {
        ret.push_back(_layers[l_idx]);
    }
    return ret;
}

const std::vector<SizeType>& DLGraph::output_layers_idx() const
{
    return _output_layers_idx;
}

std::vector<std::shared_ptr<Layer>> DLGraph::forward_layers() const
{
    std::vector<std::shared_ptr<Layer>> ret;
    for (std::size_t l_idx = 0; l_idx < _layers.size(); ++l_idx)
    {
        if (std::find(_loss_layers_idx.begin(),
                      _loss_layers_idx.end(),
                      l_idx) == _loss_layers_idx.cend())
        {
            ret.push_back(_layers[l_idx]);
        }
    }
    return ret;
}

std::vector<SizeType> DLGraph::forward_layers_idx() const
{
    std::vector<SizeType> ret;
    for (std::size_t l_idx = 0; l_idx < _layers.size(); ++l_idx)
    {
        if (std::find(_loss_layers_idx.begin(),
                      _loss_layers_idx.end(),
                      l_idx) == _loss_layers_idx.cend())
        {
            ret.push_back(l_idx);
        }
    }
    return ret;
}

std::vector<std::shared_ptr<LossLayer>> DLGraph::loss_layers() const
{
    std::vector<std::shared_ptr<LossLayer>> ret;
    for (const auto& loss_layer_idx: _loss_layers_idx)
    {
        ret.push_back(std::dynamic_pointer_cast<LossLayer>(_layers[loss_layer_idx]));
    }
    return ret;
}

std::vector<SizeType> DLGraph::loss_layers_idx() const
{
    return _loss_layers_idx;
}

std::vector<DLGraph::Arc> DLGraph::training_forward_run() const
{
    std::vector<DLGraph::Arc> ret;
    std::vector<SizeType> layers_idx(_input_layers_idx);
    std::vector<SizeType> done_list;
    while(!layers_idx.empty())
    {
        auto curr_layers_idx = layers_idx;
        layers_idx.clear();
        for (auto from_layer_idx: curr_layers_idx)
        {
            for (const auto& to_layer_idx: training_forward(from_layer_idx))
            {
                DLGraph::Arc arc;
                arc.from = _layers[from_layer_idx];
                arc.to   = _layers[to_layer_idx];
                ret.push_back(arc);

                done_list.push_back(from_layer_idx);
                if (DLMath::index_of(curr_layers_idx, to_layer_idx) == -1
                    && DLMath::index_of(done_list, to_layer_idx) == -1
                    && DLMath::index_of(layers_idx, to_layer_idx) == -1)
                {
                    layers_idx.push_back(to_layer_idx);
                }
            }
        }
    }
    return ret;
}

std::vector<DLGraph::Arc> DLGraph::forward_run() const
{
    std::vector<DLGraph::Arc> ret;
    std::vector<SizeType> layers_idx(_input_layers_idx);
    std::vector<SizeType> done_list;
    while(!layers_idx.empty())
    {
        auto curr_layers_idx = layers_idx;
        layers_idx.clear();
        for (auto from_layer_idx: curr_layers_idx)
        {
            for (const auto& to_layer_idx: forward(from_layer_idx))
            {
                DLGraph::Arc arc;
                arc.from = _layers[from_layer_idx];
                arc.to   = _layers[to_layer_idx];
                ret.push_back(arc);

                done_list.push_back(from_layer_idx);
                if (DLMath::index_of(curr_layers_idx, to_layer_idx) == -1
                    && DLMath::index_of(done_list, to_layer_idx) == -1
                    && DLMath::index_of(layers_idx, to_layer_idx) == -1)
                {
                    layers_idx.push_back(to_layer_idx);
                }
            }
        }
    }
    return ret;
}

std::vector<DLGraph::Arc> DLGraph::backward_run() const
{
    std::vector<DLGraph::Arc> ret;
    std::vector<SizeType> layers_idx(_loss_layers_idx);
    std::vector<SizeType> done_list;
    while(!layers_idx.empty())
    {
        auto curr_layers_idx = layers_idx;
        layers_idx.clear();
        for (auto from_layer_idx: curr_layers_idx)
        {
            for (const auto& to_layer_idx: backward(from_layer_idx))
            {
                DLGraph::Arc arc;
                arc.from = _layers[from_layer_idx];
                arc.to   = _layers[to_layer_idx];
                ret.push_back(arc);

                done_list.push_back(from_layer_idx);
                if (DLMath::index_of(curr_layers_idx, to_layer_idx) == -1
                    && DLMath::index_of(done_list, to_layer_idx) == -1
                    && DLMath::index_of(layers_idx, to_layer_idx) == -1)
                {
                    layers_idx.push_back(to_layer_idx);
                }
            }
        }
    }
    return ret;
}

SizeType DLGraph::size() const
{
    return _layers.size();
}

std::shared_ptr<Layer> DLGraph::layer(SizeType idx)
{
    return _layers[idx];
}

std::shared_ptr<Layer> DLGraph::operator[](SizeType idx)
{
    return layer(idx);
}

std::int64_t DLGraph::index_of(const Layer& l) const
{
    std::vector<const Layer*> layer_pointers;
    for (const auto& l_graph: _layers)
    {
        layer_pointers.push_back(l_graph.get());
    }
    return DLMath::index_of(layer_pointers, &l);
}

void DLGraph::_compute_input_layers()
{
    _input_layers_idx.clear();
    for (std::size_t l_idx = 0; l_idx < _layers.size(); ++l_idx)
    {
        if (!_forward_graph.has_predecessors(l_idx)
            && std::find(_loss_layers_idx.begin(), _loss_layers_idx.end(), l_idx)
               == _loss_layers_idx.cend())
        {
            _input_layers_idx.push_back(l_idx);
        }
    }
}

void DLGraph::_compute_output_layers()
{
    _output_layers_idx.clear();
    for (std::size_t l_idx = 0; l_idx < _layers.size(); ++l_idx)
    {
        if (!_forward_graph.has_successors(l_idx)
            && std::find(_loss_layers_idx.begin(), _loss_layers_idx.end(), l_idx)
               == _loss_layers_idx.cend())
        {
            _output_layers_idx.push_back(l_idx);
        }
    }
}

} // namespace EdgeLearning
