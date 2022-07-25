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
    , _backward_graph(_layers)
{ }

DLGraph::DLGraph(const DLGraph& obj)
    : _layers()
    , _forward_graph(obj._forward_graph)
    , _backward_graph(obj._backward_graph)
{
    _forward_graph._nodes = _layers;
    _backward_graph._nodes = _layers;
    for(const auto& l: obj._layers)
    {
        _layers.push_back(l->clone());
    }
}

DLGraph& DLGraph::operator=(const DLGraph& obj)
{
    if (this == &obj) return *this;
    _forward_graph = obj._forward_graph;
    _backward_graph = obj._backward_graph;

    _layers.clear();
    for(const auto& l: obj._layers)
    {
        _layers.push_back(l->clone());
    }

    _forward_graph._nodes = _layers;
    _backward_graph._nodes = _layers;
    return *this;
}

void DLGraph::add_node(Layer::SharedPtr layer)
{
    _layers.push_back(layer);

}

void DLGraph::add_edge(Layer::SharedPtr from, Layer::SharedPtr to)
{
    add_edge(std::vector<Layer::SharedPtr>({from}), to);
}

void DLGraph::add_edge(
    std::vector<Layer::SharedPtr> froms, Layer::SharedPtr to)
{
    add_front_arc(froms, to);
    add_back_arc(to, froms);
}

void DLGraph::add_front_arc(
    std::vector<Layer::SharedPtr> froms, Layer::SharedPtr to)
{
    for (const auto& from: froms)
    {
        _forward_graph.add_edge(from, to);
    }
}

void DLGraph::add_back_arc(
    std::vector<Layer::SharedPtr> froms, Layer::SharedPtr to)
{
    for (const auto& from: froms)
    {
        _backward_graph.add_edge(from, to);
    }
}

void DLGraph::add_edge(
    Layer::SharedPtr from, std::vector<Layer::SharedPtr> tos)
{
    add_front_arc(from, tos);
    add_back_arc(tos, from);
}

void DLGraph::add_front_arc(
    Layer::SharedPtr from, std::vector<Layer::SharedPtr> tos)
{
    for (const auto& to: tos)
    {
        _forward_graph.add_edge(from, to);
    }
}

void DLGraph::add_back_arc(
    Layer::SharedPtr from, std::vector<Layer::SharedPtr> tos)
{
    for (const auto& to: tos)
    {
        _backward_graph.add_edge(from, to);
    }
}

const std::vector<std::size_t>& DLGraph::forward(std::size_t layer_idx)
{
    return _forward_graph.successors(layer_idx);
}

std::vector<std::size_t> DLGraph::forward_predecessors(std::size_t layer_idx)
{
    return _forward_graph.predecessors(layer_idx);
}

const std::vector<std::size_t>& DLGraph::backward(std::size_t layer_idx)
{
    return _backward_graph.successors(layer_idx);
}

std::vector<std::size_t> DLGraph::backward_predecessors(std::size_t layer_idx)
{
    return _backward_graph.predecessors(layer_idx);
}

std::vector<std::int64_t> DLGraph::forward_adjacent_matrix()
{
    return _forward_graph.adjacent_matrix();
}

std::vector<std::int64_t> DLGraph::backward_adjacent_matrix()
{
    return _backward_graph.adjacent_matrix();
}

} // namespace EdgeLearning
