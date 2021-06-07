/***************************************************************************
 *            tests/test_replaceme.cpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne.
 *
 *  Ariadne is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "test.hpp"
#include "dnn/type.hpp"
#include "dnn/dlmath.i.hpp"
#include "dnn/layer.hpp"
#include "dnn/model.hpp"

#include <vector>

using namespace std;
using namespace Ariadne;

class TestDnn {
public:
    void test() {
        ARIADNE_TEST_CALL(test_dlmath());
    }
private:
    void test_dlmath() {
        const int SEED = 1;
        const int PRINT_TIMES = 4;

        ARIADNE_PRINT_TEST_COMMENT("Test normal_pdf");
        rne_t generator{SEED};
        auto dist = dlmath::normal_pdf<num_t>(0, 0.1);
        for (size_t i = 0; i < PRINT_TIMES; ++i)
        {
            ARIADNE_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(dist(generator)));
        }

        ARIADNE_PRINT_TEST_COMMENT("Test ReLU");
        std::vector<num_t> test_vec{-2,-1,0,1,2};
        std::vector<num_t> truth_vec{0,0,0,1,2};
        dlmath::relu<num_t>(test_vec.data(), test_vec.data(), test_vec.size());
        for (size_t i = 0; i < truth_vec.size(); ++i)
        {
            ARIADNE_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(test_vec[i]));
            ARIADNE_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000000001);
        }

        ARIADNE_PRINT_TEST_COMMENT("Test SoftMax");
        test_vec = std::vector<num_t>{-2,-1,0,1,2};
        truth_vec = std::vector<num_t>{
            0.01165623095604, 0.031684920796124, 0.086128544436269,
            0.23412165725274, 0.63640864655883};
        dlmath::softmax<num_t>(test_vec.data(), test_vec.data(), test_vec.size());
        for (size_t i = 0; i < truth_vec.size(); ++i)
        {
            ARIADNE_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(test_vec[i]));
            ARIADNE_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000000001);
        }
    }

};

int main() {
    TestDnn().test();
    return ARIADNE_TEST_FAILURES;
}



