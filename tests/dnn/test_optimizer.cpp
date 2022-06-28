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

    void train(Layer& layer) override
    {
        SizeType param_count = layer.param_count();
        for (SizeType i = 0; i < param_count; ++i)
        {
            NumType& param    = layer.param(i);
            NumType& gradient = layer.gradient(i);

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
    }

private:
    void test_optimizer() {
        std::size_t input_size = 1;
        std::size_t output_size = 1;

        EDGE_LEARNING_TEST_TRY(auto o = CustomOptimizer());
        auto o = CustomOptimizer();
        EDGE_LEARNING_TEST_TRY(o.reset());

        auto l = DenseLayer(m, "dense_optimizer", input_size, output_size);
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
        }

        for (std::size_t i = 0; i < l.param_count(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(old_params[i], l.param(i));
        }
    }

    Model m;
};

int main() {
    TestOptimizer().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
