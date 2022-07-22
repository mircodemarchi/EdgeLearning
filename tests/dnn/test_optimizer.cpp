/***************************************************************************
 *            dnn/test_optimizer.cpp
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
#include "dnn/optimizer.hpp"

#include "dnn/model.hpp"
#include "dnn/dense.hpp"


using namespace std;
using namespace EdgeLearning;

class CustomOptimizer : public Optimizer
{
public:
    CustomOptimizer()
        : Optimizer()
    {}

private:
    void _train(Layer& layer_from, Layer& layer_to) override
    {
        SizeType param_count = layer_to.param_count();
        for (SizeType i = 0; i < param_count; ++i)
        {
            NumType& param    = layer_to.param(i);
            NumType& gradient = layer_from.gradient(i);

            param -= 0.03 * gradient;

            // Reset the gradient accumulated again in the next training epoch.
            gradient = NumType{0.0};
        }
    }
};


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


class TestOptimizer {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_optimizer());
        EDGE_LEARNING_TEST_CALL(test_train_check());
    }

private:
    void test_optimizer() {
        std::size_t input_size = 1;
        std::size_t output_size = 1;

        EDGE_LEARNING_TEST_TRY(auto o = CustomOptimizer(); (void) o);
        auto o = CustomOptimizer();
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
            if (t >= 10000) break;
        }

        if (t < 10000) //< convergence reached.
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
        }
    }

    void test_train_check()
    {
        auto o = CustomOptimizer();
        auto l1 = DenseLayer("dense_optimizer1", 10, 10);
        auto l2 = DenseLayer("dense_optimizer2", 20, 20);
        auto l3 = DenseLayer("dense_optimizer3", 10, 10);

        EDGE_LEARNING_TEST_FAIL(o.train_check(l1, l2));
        EDGE_LEARNING_TEST_THROWS(o.train_check(l1, l2), std::runtime_error);
        EDGE_LEARNING_TEST_TRY(o.train_check(l1, l3));
        EDGE_LEARNING_TEST_TRY(o.train(l2, l1));
    }

    Model m;
};

int main() {
    TestOptimizer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
