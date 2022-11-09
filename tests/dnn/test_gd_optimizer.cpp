/***************************************************************************
 *            dnn/test_gd_optimizer.cpp
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
#include "dnn/gd_optimizer.hpp"

#include "dnn/model.hpp"
#include "dnn/dense.hpp"


using namespace std;
using namespace EdgeLearning;


template<typename T>
T dummy_loss(T v)
{
    return v * v - 2 * v + 1;
}

template<typename T>
T dummy_loss_gradient(T v)
{
    return 2 * v - 2;
}


class TestGDOptimizer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_optimizer());
    }

private:
    std::size_t input_size = 1;
    std::size_t output_size = 1;

    void test_optimizer() {
        std::vector<NumType> eta_to_test({
            0.3,
            0.1,
            0.03,
            0.01,
            0.003,
            0.001
            });

        std::vector<SizeType> num_iterations;
        for (const auto& e: eta_to_test)
        {
            auto num_iter = _test_optimize(e);
            num_iterations.push_back(num_iter);
        }

        std::cout << "Iterations report: " << std::endl;
        for (std::size_t i = 0; i < eta_to_test.size(); ++i)
        {
            std::cout << "GradientDescentOptimizer(" << eta_to_test[i] << ") iterations = ";
            std::cout << (num_iterations[i] != 0
                          ? std::to_string(num_iterations[i])
                          : "inf") << std::endl;
        }
    }

    SizeType _test_optimize(NumType eta)
    {
        EDGE_LEARNING_TEST_PRINT("GradientDescentOptimizer(" + std::to_string(eta) + ")");
        EDGE_LEARNING_TEST_TRY(auto o = GradientDescentOptimizer(eta); (void) o);
        auto o = GradientDescentOptimizer(eta);
        EDGE_LEARNING_TEST_TRY(o.reset());

        auto l = DenseLayer("dense_optimizer", input_size, output_size);
        for (std::size_t i = 0; i < l.param_count(); ++i)
        {
            l.param(i) = 0.0;
            l.gradient(i) = 0.0;
        }
        std::vector<NumType> old_params(l.param_count());
        SizeType t = 0;
        while(true)
        {
            for (std::size_t i = 0; i < l.param_count(); ++i)
            {
                old_params[i] = l.param(i);
                l.gradient(i) = dummy_loss_gradient<NumType>(l.param(i));
            }

            EDGE_LEARNING_TEST_TRY(o.train(l));

            bool convergence = true;
            std::cout << "optimization step " << t++ << ":";
            for (std::size_t i = 0; i < l.param_count(); ++i)
            {
                if (old_params[i] != l.param(i))
                {
                    convergence = false;
                }

                auto loss = dummy_loss<NumType>(l.param(i));
                std::cout << " { w" << i << ":" << l.param(i) << " ";
                std::cout << "l" << i << ":" << loss << " }";
            }
            std::cout << std::endl;
            if (convergence) break;
            if (t >= 20000) break;
        }

        if (t < 20000) //< convergence reached.
        {
            for (std::size_t i = 0; i < l.param_count(); ++i)
            {
                EDGE_LEARNING_TEST_EQUAL(old_params[i], l.param(i));
            }
        }
        else
        {
            for (std::size_t i = 0; i < l.param_count(); ++i)
            {
                EDGE_LEARNING_TEST_NOT_EQUAL(old_params[i], l.param(i));
            }
            t = 0;
        }

        return t;
    }

    Model m;
};

int main() {
    TestGDOptimizer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
