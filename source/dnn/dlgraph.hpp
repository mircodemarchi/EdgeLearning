/***************************************************************************
 *            dnn/dlgraph.hpp
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

/*! \file  dnn/dlgraph.hpp
 *  \brief Deep Learning Graph structure.
 */

#ifndef EDGE_LEARNING_DNN_GRAPH_HPP
#define EDGE_LEARNING_DNN_GRAPH_HPP

#include "layer.hpp"
#include "loss.hpp"
#include "dlmath.hpp"

#include <vector>
#include <set>
#include <string>
#include <map>


namespace EdgeLearning {

template<typename T>
class Graph
{
public:
    Graph(std::vector<T>& nodes_init)
        : _edges{}
        , _nodes{nodes_init}
    { }

    Graph(const Graph<T>& obj)
        : _edges{obj._edges}
        , _nodes{obj._nodes}
    { }

    Graph& operator=(const Graph& obj)
    {
        if (this == &obj) return *this;
        _edges = obj._edges;
        _nodes = obj._nodes;
        return *this;
    }

    void add_arc(const T& from, const T& to)
    {
        auto from_idx = DLMath::index_of(_nodes, from);
        auto to_idx = DLMath::index_of(_nodes, to);
        if (from_idx == -1 || to_idx == -1)
        {
            throw std::runtime_error(
                "add_arc error: params are not included in nodes");
        }
        _edges[static_cast<std::size_t>(from_idx)].insert(
            static_cast<std::size_t>(to_idx));
    }

    [[nodiscard]] bool has_successors(std::size_t idx) const
    {
        return _edges.count(idx) > 0 && !_edges.at(idx).empty();
    }

    [[nodiscard]] std::set<std::size_t> successors(std::size_t idx) const
    {
        return _edges.count(idx) > 0 ? _edges.at(idx) : std::set<std::size_t>();
    }

    [[nodiscard]] bool has_predecessors(std::size_t idx) const
    {
        for (const auto& edge: _edges)
        {
            if (std::find(edge.second.begin(), edge.second.end(), idx)
                != edge.second.cend())
            {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] std::set<std::size_t> predecessors(std::size_t idx) const
    {
        std::set<std::size_t> ret;
        for (const auto& edge: _edges)
        {
            const auto& node = edge.first;
            const auto& successors = edge.second;
            if (std::find(successors.begin(), successors.end(), idx)
                != successors.cend())
            {
                ret.insert(node);
            }
        }
        return ret;
    }

    [[nodiscard]] const std::vector<T>& nodes() const { return _nodes; }
    [[nodiscard]] const std::map<std::size_t, std::set<std::size_t>>& edges() const
    { return _edges; }

    [[nodiscard]] std::vector<std::int64_t> adjacent_matrix() const
    {
        std::vector<std::int64_t> ret(_nodes.size() * _nodes.size(),
                                      std::int64_t(0));
        for (const auto& edge: _edges)
        {
            auto row = edge.first * _nodes.size();
            const auto& successors = edge.second;
            for (const auto& successor: successors)
            {
                auto col = successor;
                ret[row + col] = std::int64_t(1);
            }
        }
        return ret;
    }

private:
    friend class DLGraph;

    std::map<std::size_t, std::set<std::size_t>> _edges;
    std::vector<T>& _nodes;
};

class DLGraph
{
public:
    using SharedPtr = std::shared_ptr<DLGraph>;

    struct Arc {
        std::shared_ptr<Layer> from;
        std::shared_ptr<Layer> to;
    };

    DLGraph();
    DLGraph(const DLGraph& obj);

    DLGraph& operator=(const DLGraph& obj);

    void add_node(std::shared_ptr<Layer> layer);

    void add_edge(std::shared_ptr<Layer> from, std::shared_ptr<Layer> to);
    void add_edge(std::shared_ptr<Layer> from, std::shared_ptr<LossLayer> to);
    void add_arc_forward(std::shared_ptr<Layer> from, std::shared_ptr<Layer> to);
    void add_arc_forward(std::shared_ptr<Layer> from, std::shared_ptr<LossLayer> to);
    void add_arc_backward(std::shared_ptr<Layer> from, std::shared_ptr<Layer> to);

    void add_edge(std::vector<std::shared_ptr<Layer>> froms, std::shared_ptr<Layer> to);
    void add_edge(std::vector<std::shared_ptr<Layer>> froms, std::shared_ptr<LossLayer> to);
    void add_arc_forward(std::vector<std::shared_ptr<Layer>> froms, std::shared_ptr<Layer> to);
    void add_arc_forward(std::vector<std::shared_ptr<Layer>> froms, std::shared_ptr<LossLayer> to);
    void add_arc_backward(std::vector<std::shared_ptr<Layer>> froms, std::shared_ptr<Layer> to);

    void add_edge(std::shared_ptr<Layer> from, std::vector<std::shared_ptr<Layer>> tos);
    void add_arc_forward(std::shared_ptr<Layer> from, std::vector<std::shared_ptr<Layer>> tos);
    void add_arc_backward(std::shared_ptr<Layer> from, std::vector<std::shared_ptr<Layer>> tos);

    void add_loss(std::shared_ptr<LossLayer> layer);

    [[nodiscard]] bool has_training_forward(std::size_t layer_idx) const;
    [[nodiscard]] bool has_training_forward_predecessors(std::size_t layer_idx) const;
    [[nodiscard]] bool has_forward(std::size_t layer_idx) const;
    [[nodiscard]] bool has_forward_predecessors(std::size_t layer_idx) const;
    [[nodiscard]] bool has_backward(std::size_t layer_idx) const;
    [[nodiscard]] bool has_backward_predecessors(std::size_t layer_idx) const;
    [[nodiscard]] std::set<std::size_t> training_forward(std::size_t layer_idx) const;
    [[nodiscard]] std::set<std::size_t> training_forward_predecessors(std::size_t layer_idx) const;
    [[nodiscard]] std::set<std::size_t> forward(std::size_t layer_idx) const;
    [[nodiscard]] std::set<std::size_t> forward_predecessors(std::size_t layer_idx) const;
    [[nodiscard]] std::set<std::size_t> backward(std::size_t layer_idx) const;
    [[nodiscard]] std::set<std::size_t> backward_predecessors(std::size_t layer_idx) const;

    [[nodiscard]] std::vector<std::int64_t> training_forward_adjacent_matrix() const;
    [[nodiscard]] std::vector<std::int64_t> forward_adjacent_matrix() const;
    [[nodiscard]] std::vector<std::int64_t> backward_adjacent_matrix() const;

    [[nodiscard]] const std::vector<std::shared_ptr<Layer>>& layers() const;
    [[nodiscard]] std::vector<SizeType> layers_idx() const;
    [[nodiscard]] std::vector<std::shared_ptr<Layer>> training_forward_layers() const;
    [[nodiscard]] std::vector<SizeType> training_forward_layers_idx() const;
    [[nodiscard]] std::vector<std::shared_ptr<Layer>> forward_layers() const;
    [[nodiscard]] std::vector<SizeType> forward_layers_idx() const;
    [[nodiscard]] std::vector<std::shared_ptr<Layer>> backward_layers() const;
    [[nodiscard]] std::vector<SizeType> backward_layers_idx() const;
    [[nodiscard]] std::vector<std::shared_ptr<Layer>> input_layers() const;
    [[nodiscard]] const std::vector<SizeType>& input_layers_idx() const;
    [[nodiscard]] std::vector<std::shared_ptr<Layer>> output_layers() const;
    [[nodiscard]] const std::vector<SizeType>& output_layers_idx() const;
    [[nodiscard]] std::vector<std::shared_ptr<LossLayer>> loss_layers() const;
    [[nodiscard]] std::vector<SizeType> loss_layers_idx() const;

    [[nodiscard]] std::vector<Arc> training_forward_run() const;
    [[nodiscard]] std::vector<Arc> forward_run() const;
    [[nodiscard]] std::vector<Arc> backward_run() const;

    [[nodiscard]] SizeType size() const;
    std::shared_ptr<Layer> layer(SizeType idx);
    std::shared_ptr<Layer> operator[](SizeType idx);

    template <typename Layer_T>
    std::shared_ptr<Layer_T> as(SizeType idx) const
    {
        return std::dynamic_pointer_cast<Layer_T>(_layers[idx]);
    }

    std::int64_t index_of(const Layer& l) const;

private:
    void _compute_input_layers();
    void _compute_output_layers();

    std::vector<std::shared_ptr<Layer>> _layers;
    Graph<std::shared_ptr<Layer>> _forward_graph;
    Graph<std::shared_ptr<Layer>> _training_forward_graph;
    Graph<std::shared_ptr<Layer>> _backward_graph;
    std::vector<SizeType> _loss_layers_idx;

    std::vector<SizeType> _input_layers_idx;
    std::vector<SizeType> _output_layers_idx;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_GRAPH_HPP
