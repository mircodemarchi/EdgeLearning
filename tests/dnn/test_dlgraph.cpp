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
#include "dnn/mse_loss.hpp"


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

        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[0], nodes[1]));
        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[0], nodes[2]));
        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[1], nodes[2]));
        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[2], nodes[3]));
        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[3], nodes[4]));
        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[3], nodes[5]));
        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[4], nodes[5]));

        EDGE_LEARNING_TEST_FAIL(graph.add_arc(nodes[0], "error"));
        EDGE_LEARNING_TEST_THROWS(
            graph.add_arc(nodes[0], "error"), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(graph.add_arc("error", nodes[0]));
        EDGE_LEARNING_TEST_THROWS(
            graph.add_arc("error", nodes[0]), std::runtime_error);

        EDGE_LEARNING_TEST_EQUAL(graph.edges().size(), 5);

        std::vector<std::size_t> check_successors; SizeType i;
        EDGE_LEARNING_TEST_ASSERT(graph.has_successors(0));
        auto successors_n0 = graph.successors(0);
        EDGE_LEARNING_TEST_EQUAL(successors_n0.size(), 2);
        check_successors = {1, 2}; i = 0;
        for (const auto& e: successors_n0)
            EDGE_LEARNING_TEST_EQUAL(e, check_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_successors(1));
        auto successors_n1 = graph.successors(1);
        EDGE_LEARNING_TEST_EQUAL(successors_n1.size(), 1);
        check_successors = {2}; i = 0;
        for (const auto& e: successors_n1)
            EDGE_LEARNING_TEST_EQUAL(e, check_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_successors(2));
        auto successors_n2 = graph.successors(2);
        EDGE_LEARNING_TEST_EQUAL(successors_n2.size(), 1);
        check_successors = {3}; i = 0;
        for (const auto& e: successors_n2)
            EDGE_LEARNING_TEST_EQUAL(e, check_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_successors(3));
        auto successors_n3 = graph.successors(3);
        EDGE_LEARNING_TEST_EQUAL(successors_n3.size(), 2);
        check_successors = {4, 5}; i = 0;
        for (const auto& e: successors_n3)
            EDGE_LEARNING_TEST_EQUAL(e, check_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_successors(4));
        auto successors_n4 = graph.successors(4);
        EDGE_LEARNING_TEST_EQUAL(successors_n4.size(), 1);
        check_successors = {5}; i = 0;
        for (const auto& e: successors_n4)
            EDGE_LEARNING_TEST_EQUAL(e, check_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(!graph.has_successors(5));
        auto successors_n5 = graph.successors(5);
        EDGE_LEARNING_TEST_EQUAL(successors_n5.size(), 0);

        std::vector<std::size_t> check_predecessors;
        EDGE_LEARNING_TEST_ASSERT(!graph.has_predecessors(0));
        auto predecessors_n0 = graph.predecessors(0);
        EDGE_LEARNING_TEST_EQUAL(predecessors_n0.size(), 0);
        EDGE_LEARNING_TEST_ASSERT(graph.has_predecessors(1));
        auto predecessors_n1 = graph.predecessors(1);
        EDGE_LEARNING_TEST_EQUAL(predecessors_n1.size(), 1);
        check_predecessors = {0}; i = 0;
        for (const auto& e: predecessors_n1)
            EDGE_LEARNING_TEST_EQUAL(e, check_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_predecessors(2));
        auto predecessors_n2 = graph.predecessors(2);
        EDGE_LEARNING_TEST_EQUAL(predecessors_n2.size(), 2);
        check_predecessors = {0, 1}; i = 0;
        for (const auto& e: predecessors_n2)
            EDGE_LEARNING_TEST_EQUAL(e, check_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_predecessors(3));
        auto predecessors_n3 = graph.predecessors(3);
        EDGE_LEARNING_TEST_EQUAL(predecessors_n3.size(), 1);
        check_predecessors = {2}; i = 0;
        for (const auto& e: predecessors_n3)
            EDGE_LEARNING_TEST_EQUAL(e, check_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_predecessors(4));
        auto predecessors_n4 = graph.predecessors(4);
        EDGE_LEARNING_TEST_EQUAL(predecessors_n4.size(), 1);
        check_predecessors = {3}; i = 0;
        for (const auto& e: predecessors_n4)
            EDGE_LEARNING_TEST_EQUAL(e, check_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_predecessors(5));
        auto predecessors_n5 = graph.predecessors(5);
        EDGE_LEARNING_TEST_EQUAL(predecessors_n5.size(), 2);
        check_predecessors = {3, 4}; i = 0;
        for (const auto& e: predecessors_n5)
            EDGE_LEARNING_TEST_EQUAL(e, check_predecessors[i++]);

        Graph<std::string> graph_copy(graph);
        EDGE_LEARNING_TEST_EQUAL(
            graph_copy.nodes().size(), graph.nodes().size());
        EDGE_LEARNING_TEST_EQUAL(
            graph_copy.edges().size(), graph.edges().size());

        std::vector<std::string> vec_empty;
        Graph<std::string> graph_assign(vec_empty);
        EDGE_LEARNING_TEST_EQUAL(
            graph_assign.nodes().size(), 0);
        EDGE_LEARNING_TEST_EQUAL(
            graph_assign.edges().size(), 0);
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

        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[0], nodes[1]));
        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[0], nodes[2]));
        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[1], nodes[2]));
        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[2], nodes[3]));
        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[3], nodes[4]));
        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[3], nodes[5]));
        EDGE_LEARNING_TEST_TRY(graph.add_arc(nodes[4], nodes[5]));

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
        DLGraph graph; SizeType i;
        auto d0 = std::make_shared<DenseLayer>("d0");
        auto d1 = std::make_shared<DenseLayer>("d1");
        auto d2 = std::make_shared<DenseLayer>("d2");
        auto d3 = std::make_shared<DenseLayer>("d3");
        auto d4 = std::make_shared<DenseLayer>("d4");
        auto d5 = std::make_shared<DenseLayer>("d5");
        auto l0 = std::dynamic_pointer_cast<LossLayer>(
            std::make_shared<MSELossLayer>("l0"));
        auto l1 = std::dynamic_pointer_cast<LossLayer>(
            std::make_shared<MSELossLayer>("l1"));

        EDGE_LEARNING_TEST_TRY(graph.add_node(d0));
        EDGE_LEARNING_TEST_TRY(graph.add_node(d1));
        EDGE_LEARNING_TEST_TRY(graph.add_node(d2));
        EDGE_LEARNING_TEST_TRY(graph.add_node(d3));
        EDGE_LEARNING_TEST_TRY(graph.add_node(d4));
        EDGE_LEARNING_TEST_TRY(graph.add_node(d5));
        EDGE_LEARNING_TEST_TRY(graph.add_loss(l0));
        EDGE_LEARNING_TEST_TRY(graph.add_loss(l1));

        EDGE_LEARNING_TEST_EQUAL(graph.layers().size(), graph.size());
        for (i = 0; i < graph.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(graph.layers()[i], graph.layer(i));
            EDGE_LEARNING_TEST_EQUAL(graph.layers()[i], graph[i]);
            EDGE_LEARNING_TEST_EQUAL(static_cast<std::int64_t>(i),
                                     graph.index_of(*graph.layer(i)));
        }
        EDGE_LEARNING_TEST_TRY(
            graph.as<DenseLayer>(graph.layers_idx()[0]));
        EDGE_LEARNING_TEST_TRY(
            graph.as<MSELossLayer>(graph.loss_layers_idx()[0]));
        auto d0_converted = graph.as<DenseLayer>(graph.layers_idx()[0]);
        auto l0_converted = graph.as<MSELossLayer>(graph.loss_layers_idx()[0]);
        EDGE_LEARNING_TEST_EQUAL(d0->name(), d0_converted->name());
        EDGE_LEARNING_TEST_EQUAL(l0->name(), l0_converted->name());

        EDGE_LEARNING_TEST_EQUAL(graph.layers().size(), 8);
        EDGE_LEARNING_TEST_EQUAL(graph.layers_idx().size(), 8);
        for (i = 0; i < graph.layers_idx().size(); i++)
            EDGE_LEARNING_TEST_EQUAL(graph.layers_idx()[i], i);
        EDGE_LEARNING_TEST_EQUAL(graph.training_forward_layers().size(),
                                 graph.layers().size());
        EDGE_LEARNING_TEST_EQUAL(graph.training_forward_layers_idx().size(),
                                 graph.layers_idx().size());
        for (i = 0; i < graph.training_forward_layers_idx().size(); i++)
            EDGE_LEARNING_TEST_EQUAL(graph.layers_idx()[i], i);
        EDGE_LEARNING_TEST_EQUAL(graph.backward_layers().size(),
                                 graph.layers().size());
        EDGE_LEARNING_TEST_EQUAL(graph.backward_layers_idx().size(),
                                 graph.layers_idx().size());
        for (i = 0; i < graph.backward_layers_idx().size(); i++)
            EDGE_LEARNING_TEST_EQUAL(graph.layers_idx()[i], i);
        EDGE_LEARNING_TEST_EQUAL(graph.forward_layers().size(), 6);
        EDGE_LEARNING_TEST_EQUAL(graph.forward_layers_idx().size(), 6);
        for (i = 0; i < graph.forward_layers_idx().size(); i++)
            EDGE_LEARNING_TEST_EQUAL(graph.forward_layers_idx()[i], i);
        EDGE_LEARNING_TEST_EQUAL(graph.loss_layers().size(), 2);
        EDGE_LEARNING_TEST_EQUAL(graph.loss_layers_idx().size(), 2);
        EDGE_LEARNING_TEST_EQUAL(graph.loss_layers_idx()[0], 6);
        EDGE_LEARNING_TEST_EQUAL(graph.loss_layers_idx()[1], 7);

        EDGE_LEARNING_TEST_TRY(graph.add_edge(d0, d1));
        EDGE_LEARNING_TEST_TRY(graph.add_edge({d0, d1}, d2));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(d2, d3));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(d3, {d4, d5}));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(d4, d5));
        EDGE_LEARNING_TEST_TRY(graph.add_arc_forward(d4, l0));
        EDGE_LEARNING_TEST_TRY(graph.add_edge(d5, l1));

        EDGE_LEARNING_TEST_EQUAL(graph.input_layers().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(graph.input_layers_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(graph.input_layers_idx()[0], 0);
        EDGE_LEARNING_TEST_EQUAL(graph.output_layers().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(graph.output_layers_idx().size(), 1);
        EDGE_LEARNING_TEST_EQUAL(graph.output_layers_idx()[0], 5);

        std::vector<std::size_t> check_forward_successors;
        EDGE_LEARNING_TEST_ASSERT(graph.has_forward(0));
        auto forward_successors_d0 = graph.forward(0);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d0.size(), 2);
        check_forward_successors = {1, 2}; i = 0;
        for (const auto& e: forward_successors_d0)
            EDGE_LEARNING_TEST_EQUAL(e, check_forward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_forward(1));
        auto forward_successors_d1 = graph.forward(1);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d1.size(), 1);
        check_forward_successors = {2}; i = 0;
        for (const auto& e: forward_successors_d1)
            EDGE_LEARNING_TEST_EQUAL(e, check_forward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_forward(2));
        auto forward_successors_d2 = graph.forward(2);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d2.size(), 1);
        check_forward_successors = {3}; i = 0;
        for (const auto& e: forward_successors_d2)
            EDGE_LEARNING_TEST_EQUAL(e, check_forward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_forward(3));
        auto forward_successors_d3 = graph.forward(3);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d3.size(), 2);
        check_forward_successors = {4, 5}; i = 0;
        for (const auto& e: forward_successors_d3)
            EDGE_LEARNING_TEST_EQUAL(e, check_forward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_forward(4));
        auto forward_successors_d4 = graph.forward(4);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d4.size(), 1);
        check_forward_successors = {5}; i = 0;
        for (const auto& e: forward_successors_d4)
            EDGE_LEARNING_TEST_EQUAL(e, check_forward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(!graph.has_forward(5));
        auto forward_successors_d5 = graph.forward(5);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_d5.size(), 0);
        EDGE_LEARNING_TEST_ASSERT(!graph.has_forward(6));
        auto forward_successors_l0 = graph.forward(6);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_l0.size(), 0);
        EDGE_LEARNING_TEST_ASSERT(!graph.has_forward(7));
        auto forward_successors_l1 = graph.forward(7);
        EDGE_LEARNING_TEST_EQUAL(forward_successors_l1.size(), 0);

        std::vector<std::size_t> check_forward_predecessors;
        EDGE_LEARNING_TEST_ASSERT(!graph.has_forward_predecessors(0));
        auto forward_predecessors_d0 = graph.forward_predecessors(0);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d0.size(), 0);
        EDGE_LEARNING_TEST_ASSERT(graph.has_forward_predecessors(1));
        auto forward_predecessors_d1 = graph.forward_predecessors(1);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d1.size(), 1);
        check_forward_predecessors = {0}; i = 0;
        for (const auto& e: forward_predecessors_d1)
            EDGE_LEARNING_TEST_EQUAL(e, check_forward_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_forward_predecessors(2));
        auto forward_predecessors_d2 = graph.forward_predecessors(2);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d2.size(), 2);
        check_forward_predecessors = {0, 1}; i = 0;
        for (const auto& e: forward_predecessors_d2)
            EDGE_LEARNING_TEST_EQUAL(e, check_forward_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_forward_predecessors(3));
        auto forward_predecessors_d3 = graph.forward_predecessors(3);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d3.size(), 1);
        check_forward_predecessors = {2}; i = 0;
        for (const auto& e: forward_predecessors_d3)
            EDGE_LEARNING_TEST_EQUAL(e, check_forward_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_forward_predecessors(4));
        auto forward_predecessors_d4 = graph.forward_predecessors(4);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d4.size(), 1);
        check_forward_predecessors = {3}; i = 0;
        for (const auto& e: forward_predecessors_d4)
            EDGE_LEARNING_TEST_EQUAL(e, check_forward_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_forward_predecessors(5));
        auto forward_predecessors_d5 = graph.forward_predecessors(5);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_d5.size(), 2);
        check_forward_predecessors = {3, 4}; i = 0;
        for (const auto& e: forward_predecessors_d5)
            EDGE_LEARNING_TEST_EQUAL(e, check_forward_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(!graph.has_forward_predecessors(6));
        auto forward_predecessors_l0 = graph.forward_predecessors(6);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_l0.size(), 0);
        EDGE_LEARNING_TEST_ASSERT(!graph.has_forward_predecessors(7));
        auto forward_predecessors_l1 = graph.forward_predecessors(7);
        EDGE_LEARNING_TEST_EQUAL(forward_predecessors_l1.size(), 0);

        std::vector<std::size_t> check_training_forward_successors;
        EDGE_LEARNING_TEST_ASSERT(graph.has_training_forward(0));
        auto training_forward_successors_d0 = graph.training_forward(0);
        EDGE_LEARNING_TEST_EQUAL(training_forward_successors_d0.size(), 2);
        check_training_forward_successors = {1, 2}; i = 0;
        for (const auto& e: training_forward_successors_d0)
            EDGE_LEARNING_TEST_EQUAL(e, check_training_forward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_training_forward(1));
        auto training_forward_successors_d1 = graph.training_forward(1);
        EDGE_LEARNING_TEST_EQUAL(training_forward_successors_d1.size(), 1);
        check_training_forward_successors = {2}; i = 0;
        for (const auto& e: training_forward_successors_d1)
            EDGE_LEARNING_TEST_EQUAL(e, check_training_forward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_training_forward(2));
        auto training_forward_successors_d2 = graph.training_forward(2);
        EDGE_LEARNING_TEST_EQUAL(training_forward_successors_d2.size(), 1);
        check_training_forward_successors = {3}; i = 0;
        for (const auto& e: training_forward_successors_d2)
            EDGE_LEARNING_TEST_EQUAL(e, check_training_forward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_training_forward(3));
        auto training_forward_successors_d3 = graph.training_forward(3);
        EDGE_LEARNING_TEST_EQUAL(training_forward_successors_d3.size(), 2);
        check_training_forward_successors = {4, 5}; i = 0;
        for (const auto& e: training_forward_successors_d3)
            EDGE_LEARNING_TEST_EQUAL(e, check_training_forward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_training_forward(4));
        auto training_forward_successors_d4 = graph.training_forward(4);
        EDGE_LEARNING_TEST_EQUAL(training_forward_successors_d4.size(), 2);
        check_training_forward_successors = {5, 6}; i = 0;
        for (const auto& e: training_forward_successors_d4)
            EDGE_LEARNING_TEST_EQUAL(e, check_training_forward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_training_forward(5));
        auto training_forward_successors_d5 = graph.training_forward(5);
        EDGE_LEARNING_TEST_EQUAL(training_forward_successors_d5.size(), 1);
        check_training_forward_successors = {7}; i = 0;
        for (const auto& e: training_forward_successors_d5)
            EDGE_LEARNING_TEST_EQUAL(e, check_training_forward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(!graph.has_training_forward(6));
        auto training_forward_successors_l0 = graph.training_forward(6);
        EDGE_LEARNING_TEST_EQUAL(training_forward_successors_l0.size(), 0);
        EDGE_LEARNING_TEST_ASSERT(!graph.has_training_forward(7));
        auto training_forward_successors_l1 = graph.training_forward(7);
        EDGE_LEARNING_TEST_EQUAL(training_forward_successors_l1.size(), 0);

        std::vector<std::size_t> check_training_forward_predecessors;
        EDGE_LEARNING_TEST_ASSERT(!graph.has_training_forward_predecessors(0));
        auto training_forward_predecessors_d0 = graph.training_forward_predecessors(0);
        EDGE_LEARNING_TEST_EQUAL(training_forward_predecessors_d0.size(), 0);
        EDGE_LEARNING_TEST_ASSERT(graph.has_training_forward_predecessors(1));
        auto training_forward_predecessors_d1 = graph.training_forward_predecessors(1);
        EDGE_LEARNING_TEST_EQUAL(training_forward_predecessors_d1.size(), 1);
        check_training_forward_predecessors = {0}; i = 0;
        for (const auto& e: training_forward_predecessors_d1)
            EDGE_LEARNING_TEST_EQUAL(e, check_training_forward_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_training_forward_predecessors(2));
        auto training_forward_predecessors_d2 = graph.training_forward_predecessors(2);
        EDGE_LEARNING_TEST_EQUAL(training_forward_predecessors_d2.size(), 2);
        check_training_forward_predecessors = {0, 1}; i = 0;
        for (const auto& e: training_forward_predecessors_d2)
            EDGE_LEARNING_TEST_EQUAL(e, check_training_forward_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_training_forward_predecessors(3));
        auto training_forward_predecessors_d3 = graph.training_forward_predecessors(3);
        EDGE_LEARNING_TEST_EQUAL(training_forward_predecessors_d3.size(), 1);
        check_training_forward_predecessors = {2}; i = 0;
        for (const auto& e: training_forward_predecessors_d3)
            EDGE_LEARNING_TEST_EQUAL(e, check_training_forward_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_training_forward_predecessors(4));
        auto training_forward_predecessors_d4 = graph.training_forward_predecessors(4);
        EDGE_LEARNING_TEST_EQUAL(training_forward_predecessors_d4.size(), 1);
        check_training_forward_predecessors = {3}; i = 0;
        for (const auto& e: training_forward_predecessors_d4)
            EDGE_LEARNING_TEST_EQUAL(e, check_training_forward_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_training_forward_predecessors(5));
        auto training_forward_predecessors_d5 = graph.training_forward_predecessors(5);
        EDGE_LEARNING_TEST_EQUAL(training_forward_predecessors_d5.size(), 2);
        check_training_forward_predecessors = {3, 4}; i = 0;
        for (const auto& e: training_forward_predecessors_d5)
            EDGE_LEARNING_TEST_EQUAL(e, check_training_forward_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_training_forward_predecessors(6));
        auto training_forward_predecessors_l0 = graph.training_forward_predecessors(6);
        EDGE_LEARNING_TEST_EQUAL(training_forward_predecessors_l0.size(), 1);
        check_training_forward_predecessors = {4}; i = 0;
        for (const auto& e: training_forward_predecessors_l0)
            EDGE_LEARNING_TEST_EQUAL(e, check_training_forward_predecessors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_training_forward_predecessors(7));
        auto training_forward_predecessors_l1 = graph.training_forward_predecessors(7);
        EDGE_LEARNING_TEST_EQUAL(training_forward_predecessors_l1.size(), 1);
        check_training_forward_predecessors = {5}; i = 0;
        for (const auto& e: training_forward_predecessors_l1)
            EDGE_LEARNING_TEST_EQUAL(e, check_training_forward_predecessors[i++]);

        std::vector<std::size_t> check_backward_successors;
        EDGE_LEARNING_TEST_ASSERT(graph.has_backward(7));
        auto backward_successors_l1 = graph.backward(7);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_l1.size(), 1);
        check_backward_successors = {5}; i = 0;
        for (const auto& e: backward_successors_l1)
            EDGE_LEARNING_TEST_EQUAL(e, check_backward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(!graph.has_backward(6));
        auto backward_successors_l0 = graph.backward(6);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_l0.size(), 0);
        EDGE_LEARNING_TEST_ASSERT(graph.has_backward(5));
        auto backward_successors_d5 = graph.backward(5);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d5.size(), 2);
        check_backward_successors = {3, 4}; i = 0;
        for (const auto& e: backward_successors_d5)
            EDGE_LEARNING_TEST_EQUAL(e, check_backward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_backward(4));
        auto backward_successors_d4 = graph.backward(4);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d4.size(), 1);
        check_backward_successors = {3}; i = 0;
        for (const auto& e: backward_successors_d5)
            EDGE_LEARNING_TEST_EQUAL(e, check_backward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_backward(3));
        auto backward_successors_d3 = graph.backward(3);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d3.size(), 1);
        check_backward_successors = {2}; i = 0;
        for (const auto& e: backward_successors_d3)
            EDGE_LEARNING_TEST_EQUAL(e, check_backward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_backward(2));
        auto backward_successors_d2 = graph.backward(2);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d2.size(), 2);
        check_backward_successors = {0, 1}; i = 0;
        for (const auto& e: backward_successors_d2)
            EDGE_LEARNING_TEST_EQUAL(e, check_backward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(graph.has_backward(1));
        auto backward_successors_d1 = graph.backward(1);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d1.size(), 1);
        check_backward_successors = {0}; i = 0;
        for (const auto& e: backward_successors_d1)
            EDGE_LEARNING_TEST_EQUAL(e, check_backward_successors[i++]);
        EDGE_LEARNING_TEST_ASSERT(!graph.has_backward(0));
        auto backward_successors_d0 = graph.backward(0);
        EDGE_LEARNING_TEST_EQUAL(backward_successors_d0.size(), 0);

        std::vector<std::size_t> check_backward_predecessors;
        EDGE_LEARNING_TEST_ASSERT(!graph.has_backward_predecessors(7));
        auto backward_predecessors_l1 = graph.backward_predecessors(7);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_l1.size(), 0);
        EDGE_LEARNING_TEST_ASSERT(!graph.has_backward_predecessors(6));
        auto backward_predecessors_l0 = graph.backward_predecessors(6);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_l0.size(), 0);
        EDGE_LEARNING_TEST_ASSERT(graph.has_backward_predecessors(5));
        auto backward_predecessors_d5 = graph.backward_predecessors(5);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d5.size(), 1);
        check_backward_predecessors = {7}; i = 0;
        for (const auto& e: backward_predecessors_d5)
            EDGE_LEARNING_TEST_EQUAL(e, check_backward_predecessors[i++]);
        auto backward_predecessors_d4 = graph.backward_predecessors(4);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d4.size(), 1);
        check_backward_predecessors = {5}; i = 0;
        for (const auto& e: backward_predecessors_d4)
            EDGE_LEARNING_TEST_EQUAL(e, check_backward_predecessors[i++]);
        auto backward_predecessors_d3 = graph.backward_predecessors(3);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d3.size(), 2);
        check_backward_predecessors = {4, 5}; i = 0;
        for (const auto& e: backward_predecessors_d3)
            EDGE_LEARNING_TEST_EQUAL(e, check_backward_predecessors[i++]);
        auto backward_predecessors_d2 = graph.backward_predecessors(2);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d2.size(), 1);
        check_backward_predecessors = {3}; i = 0;
        for (const auto& e: backward_predecessors_d2)
            EDGE_LEARNING_TEST_EQUAL(e, check_backward_predecessors[i++]);
        auto backward_predecessors_d1 = graph.backward_predecessors(1);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d1.size(), 1);
        check_backward_predecessors = {2}; i = 0;
        for (const auto& e: backward_predecessors_d1)
            EDGE_LEARNING_TEST_EQUAL(e, check_backward_predecessors[i++]);
        auto backward_predecessors_d0 = graph.backward_predecessors(0);
        EDGE_LEARNING_TEST_EQUAL(backward_predecessors_d0.size(), 2);
        check_backward_predecessors = {1, 2}; i = 0;
        for (const auto& e: backward_predecessors_d0)
            EDGE_LEARNING_TEST_EQUAL(e, check_backward_predecessors[i++]);

        std::vector<std::int64_t> forward_truth_data = {
            0, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        };
        auto forward_result = graph.forward_adjacent_matrix();
        for (i = 0; i < forward_result.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(forward_result[i], forward_truth_data[i]);
        }

        std::vector<std::int64_t> training_forward_truth_data = {
            0, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        };
        auto training_forward_result = graph.training_forward_adjacent_matrix();
        for (i = 0; i < training_forward_result.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(
                training_forward_result[i], training_forward_truth_data[i]);
        }

        std::vector<std::int64_t> backward_truth_data = {
            0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0,
        };
        auto backward_result = graph.backward_adjacent_matrix();
        for (i = 0; i < backward_result.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(backward_result[i], backward_truth_data[i]);
        }

        EDGE_LEARNING_TEST_TRY((void) graph.forward_run());
        auto forward_arcs = graph.forward_run();
        EDGE_LEARNING_TEST_EQUAL(forward_arcs.size(), 7);
        EDGE_LEARNING_TEST_TRY((void) graph.training_forward_run());
        auto training_forward_arcs = graph.training_forward_run();
        EDGE_LEARNING_TEST_EQUAL(training_forward_arcs.size(), 9);
        EDGE_LEARNING_TEST_TRY((void) graph.backward_run());
        auto backward_arcs = graph.backward_run();
        EDGE_LEARNING_TEST_EQUAL(backward_arcs.size(), 8);
    }
};

int main() {
    TestDLGraph().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
