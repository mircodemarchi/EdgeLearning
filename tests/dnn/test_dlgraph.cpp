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

using namespace std;
using namespace EdgeLearning;

class TestDLGraph {
public:

    void test() {
        EDGE_LEARNING_TEST_CALL(test_graph());
        EDGE_LEARNING_TEST_CALL(test_adjacent_matrix());
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


};

int main() {
    TestDLGraph().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
