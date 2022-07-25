/***************************************************************************
 *            dnn/test_dlgraph.cpp
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

#include "test.hpp"
#include "dnn/dlgraph.hpp"
#include "dnn/dense.hpp"

using namespace std;
using namespace EdgeLearning;

class TestDLGraph {
public:

    void test() {
        EDGE_LEARNING_TEST_CALL(test_graph());
        EDGE_LEARNING_TEST_CALL(test_adjacent_matrix());
        EDGE_LEARNING_TEST_CALL(test_dlgraph());
    }

private:

    void test_graph() {
        std::vector<std::string> nodes(
            {"n0", "n1", "n2", "n3", "n4", "n5"});
        Graph<std::string> graph(nodes);

        EDGE_LEARNING_TEST_EQUAL(graph.nodes().size(), nodes.size());
        for (std::size_t n_idx = 0; n_idx < nodes.size(); ++n_idx)
        {
            EDGE_LEARNING_TEST_EQUAL(graph.nodes()[n_idx], nodes[n_idx]);
        }

        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[0], nodes[1]));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[0], nodes[2]));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[1], nodes[2]));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[2], nodes[3]));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[3], nodes[4]));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[3], nodes[5]));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[4], nodes[5]));

        EDGE_LEARNING_TEST_FAIL(graph.add_edge(nodes[0], "error"));
        EDGE_LEARNING_TEST_THROWS(
            graph.add_edge(nodes[0], "error"), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(graph.add_edge("error", nodes[0]));
        EDGE_LEARNING_TEST_THROWS(
            graph.add_edge("error", nodes[0]), std::runtime_error);

        EDGE_LEARNING_TEST_EQUAL(graph.edges().size(), 5);

        auto successors_n0 = graph.successors(0);
        EDGE_LEARNING_TEST_EQUAL(successors_n0.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(successors_n0[0], 1);
        EDGE_LEARNING_TEST_EQUAL(successors_n0[1], 2);
        auto successors_n1 = graph.successors(1);
        EDGE_LEARNING_TEST_EQUAL(successors_n1.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(successors_n1[0], 2);
        auto successors_n5 = graph.successors(5);
        EDGE_LEARNING_TEST_EQUAL(successors_n5.size(), 0);

        auto predecessors_n0 = graph.predecessors(0);
        EDGE_LEARNING_TEST_EQUAL(predecessors_n0.size(), 0);
        auto predecessors_n2 = graph.predecessors(2);
        EDGE_LEARNING_TEST_EQUAL(predecessors_n2.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(predecessors_n2[0], 0);
        EDGE_LEARNING_TEST_EQUAL(predecessors_n2[1], 1);
        auto predecessors_n3 = graph.predecessors(3);
        EDGE_LEARNING_TEST_EQUAL(predecessors_n3.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(predecessors_n3[0], 2);

        Graph<std::string> graph_copy(graph);
        EDGE_LEARNING_TEST_EQUAL(
            graph_copy.nodes().size(), graph.nodes().size());
        EDGE_LEARNING_TEST_EQUAL(
            graph_copy.edges().size(), graph.edges().size());

        std::vector<std::string> vec_empty;
        Graph<std::string> graph_assign(vec_empty);
        graph_assign = graph;
        EDGE_LEARNING_TEST_EQUAL(
            graph_assign.nodes().size(), graph.nodes().size());
        EDGE_LEARNING_TEST_EQUAL(
            graph_assign.edges().size(), graph.edges().size());
    }

    void test_adjacent_matrix() {
        std::vector<std::string> nodes(
            {"n0", "n1", "n2", "n3", "n4", "n5"});
        Graph<std::string> graph(nodes);

        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[0], nodes[1]));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[0], nodes[2]));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[1], nodes[2]));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[2], nodes[3]));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[3], nodes[4]));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[3], nodes[5]));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(nodes[4], nodes[5]));

        std::vector<std::int64_t> truth_data = {
            0, 1, 1, 0, 0, 0,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 1,
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0,
        };
        auto result = graph.adjacent_matrix();
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(result[i], truth_data[i]);
        }
    }

    void test_dlgraph()
    {
        DLGraph graph;
        auto d0 = std::make_shared<DenseLayer>("d0");
        auto d1 = std::make_shared<DenseLayer>("d1");
        auto d2 = std::make_shared<DenseLayer>("d2");
        auto d3 = std::make_shared<DenseLayer>("d3");
        auto d4 = std::make_shared<DenseLayer>("d4");
        auto d5 = std::make_shared<DenseLayer>("d5");

        EDGE_LEARNING_TEST_TRY(graph.add_node(d0));
        EDGE_LEARNING_TEST_TRY(graph.add_node(d1));
        EDGE_LEARNING_TEST_TRY(graph.add_node(d2));
        EDGE_LEARNING_TEST_TRY(graph.add_node(d3));
        EDGE_LEARNING_TEST_TRY(graph.add_node(d4));
        EDGE_LEARNING_TEST_TRY(graph.add_node(d5));

        EDGE_LEARNING_TEST_EQUAL(graph.layers().size(), 6);

        EDGE_LEARNING_TEST_TRY(graph.add_edge(d0, d1));
        EDGE_LEARNING_TEST_TRY(graph.add_edge({d0, d1}, d2));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(d2, d3));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(d3, {d4, d5}));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(d4, d5));

        auto forward_successors_d0 = graph.forward(0);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d0.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d0[0], 1);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d0[1], 2);
        auto forward_successors_d1 = graph.forward(1);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d1.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d1[0], 2);
        auto forward_successors_d2 = graph.forward(2);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d2.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d2[0], 3);
        auto forward_successors_d3 = graph.forward(3);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d3.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d3[0], 4);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d3[1], 5);
        auto forward_successors_d4 = graph.forward(4);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d4.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d4[0], 5);
        auto forward_successors_d5 = graph.forward(5);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d5.size(), 0);

        auto forward_predecessors_d0 = graph.forward_predecessors(0);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d0.size(), 0);
        auto forward_predecessors_d1 = graph.forward_predecessors(1);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d1.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d1[0], 0);
        auto forward_predecessors_d2 = graph.forward_predecessors(2);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d2.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d2[0], 0);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d2[1], 1);
        auto forward_predecessors_d3 = graph.forward_predecessors(3);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d3.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d3[0], 2);
        auto forward_predecessors_d4 = graph.forward_predecessors(4);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d4.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d4[0], 3);
        auto forward_predecessors_d5 = graph.forward_predecessors(5);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d5.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d5[0], 3);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d5[1], 4);

        auto backward_successors_d5 = graph.backward(5);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d5.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d5[0], 3);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d5[1], 4);
        auto backward_successors_d4 = graph.backward(4);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d4.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d4[0], 3);
        auto backward_successors_d3 = graph.backward(3);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d3.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d3[0], 2);
        auto backward_successors_d2 = graph.backward(2);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d2.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d2[0], 0);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d2[1], 1);
        auto backward_successors_d1 = graph.backward(1);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d1.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d1[0], 0);
        auto backward_successors_d0 = graph.backward(0);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d0.size(), 0);

        auto backward_predecessors_d5 = graph.backward_predecessors(5);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d5.size(), 0);
        auto backward_predecessors_d4 = graph.backward_predecessors(4);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d4.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d4[0], 5);
        auto backward_predecessors_d3 = graph.backward_predecessors(3);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d3.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d3[0], 4);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d3[1], 5);
        auto backward_predecessors_d2 = graph.backward_predecessors(2);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d2.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d2[0], 3);
        auto backward_predecessors_d1 = graph.backward_predecessors(1);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d1.size(), 1);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d1[0], 2);
        auto backward_predecessors_d0 = graph.backward_predecessors(0);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d0.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d0[0], 1);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d0[1], 2);

        std::vector<std::int64_t> forward_truth_data = {
            0, 1, 1, 0, 0, 0,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 1,
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0,
        };
        auto forward_result = graph.forward_adjacent_matrix();
        for (std::size_t i = 0; i < forward_result.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(forward_result[i], forward_truth_data[i]);
        }

        std::vector<std::int64_t> backward_truth_data = {
            0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 1, 1, 0,
        };
        auto backward_result = graph.backward_adjacent_matrix();
        for (std::size_t i = 0; i < backward_result.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(backward_result[i], backward_truth_data[i]);
        }
    }
};

int main() {
    TestDLGraph().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
