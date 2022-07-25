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

#include <vector>
#include <string>
#include <map>


namespace EdgeLearning {

template<typename T>
class Graph
{
public:
    Graph(std::vector<T>& nodes_init)
        : _edges{}
        , _nodes(nodes_init)
    { }

    void add_edge(const T& from, const T& to)
    {
        auto from_idx = _index_of(from);
        auto to_idx = _index_of(to);
        if (from_idx == -1 || to_idx == -1)
        {
            throw std::runtime_error(
                "add_edge error: params are not included in nodes");
        }
        _edges[static_cast<std::size_t>(from_idx)].push_back(
            static_cast<std::size_t>(to_idx));
    }

    std::vector<std::size_t> successors(std::size_t idx)
    {
        return _edges[idx];
    }

    std::vector<std::size_t> predecessors(std::size_t idx)
    {
        std::vector<std::size_t> ret;
        for (const auto& edge: _edges)
        {
            const auto& node = edge.first;
            const auto& successors = edge.second;
            if (std::find(successors.begin(), successors.end(), idx)
                != successors.cend())
            {
                ret.push_back(node);
            }
        }
        return ret;
    }

    [[nodiscard]] const std::vector<T>& nodes() const { return _nodes; }
    [[nodiscard]] const std::map<std::size_t, std::vector<std::size_t>>& edges() const
    { return _edges; }

    std::vector<std::int64_t> adjacent_matrix()
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

    std::int64_t _index_of(const T& n)
    {
        auto itr = std::find(_nodes.begin(), _nodes.end(), n);
        if (itr != _nodes.cend())
        {
            return std::distance(_nodes.begin(), itr);
        }
        else
        {
            return -1;
        }
    }

    std::map<std::size_t, std::vector<std::size_t>> _edges;
    std::vector<T>& _nodes;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_GRAPH_HPP
