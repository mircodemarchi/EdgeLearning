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
#include <iostream>
#include <iomanip>

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

        {
            ARIADNE_PRINT_TEST_COMMENT("Test normal_pdf");
            rne_t generator{SEED};
            auto dist = dlmath::normal_pdf<num_t>(0, 0.1);
            for (size_t i = 0; i < PRINT_TIMES; ++i)
            {
                ARIADNE_TEST_PRINT(std::to_string(i) + ": " 
                    + std::to_string(dist(generator)));
            }
        }

        {
            ARIADNE_PRINT_TEST_COMMENT("Test arr sum");
            std::vector<int> test_vec1{5,4,3,2,1};
            std::vector<int> test_vec2{1,2,3,4,5};
            std::vector<int> truth_vec{6,6,6,6,6};
            dlmath::arr_sum<int>(test_vec1.data(), test_vec1.data(), 
                test_vec2.data(), test_vec1.size());
            for (size_t i = 0; i < truth_vec.size(); ++i)
            {
                ARIADNE_TEST_EQUAL(test_vec1[i], truth_vec[i]);
            }
        }

        {
            ARIADNE_PRINT_TEST_COMMENT("Test arr mul");
            std::vector<int> test_vec1{5,4,3,2,1};
            std::vector<int> test_vec2{1,2,3,4,5};
            std::vector<int> truth_vec{5,8,9,8,5};
            dlmath::arr_mul<int>(test_vec1.data(), test_vec1.data(), 
                test_vec2.data(), test_vec1.size());
            for (size_t i = 0; i < truth_vec.size(); ++i)
            {
                ARIADNE_TEST_EQUAL(test_vec1[i], truth_vec[i]);
            }
        }

        {
            ARIADNE_PRINT_TEST_COMMENT("Test matrix array mul");
            std::vector<int> test_mat{1,2,3,4};
            std::vector<int> test_vec{1,2};
            std::vector<int> truth_vec{5,11};
            ARIADNE_TEST_FAIL(
                dlmath::matarr_mul<int>(test_vec.data(), test_mat.data(), 
                                        test_vec.data(), 2, 2)
            );
            std::vector<int> res_vec; res_vec.resize(test_vec.size());
            dlmath::matarr_mul<int>(res_vec.data(), test_mat.data(), 
                                    test_vec.data(), 2, 2);
            for (size_t i = 0; i < truth_vec.size(); ++i)
            {
                ARIADNE_TEST_EQUAL(res_vec[i], truth_vec[i]);
            }
        }

        {
            ARIADNE_PRINT_TEST_COMMENT("Test ReLU");
            std::vector<num_t> test_vec{-2,-1,0,1,2};
            std::vector<num_t> truth_vec{0,0,0,1,2};
            dlmath::relu<num_t>(test_vec.data(), test_vec.data(), 
                test_vec.size());
            for (size_t i = 0; i < truth_vec.size(); ++i)
            {
                ARIADNE_TEST_PRINT(std::to_string(i) + ": " 
                    + std::to_string(test_vec[i]));
                ARIADNE_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000000001);
            }
        }

        {        
            ARIADNE_PRINT_TEST_COMMENT("Test SoftMax");
            std::vector<num_t> test_vec{-2,-1,0,1,2};
            std::vector<num_t> truth_vec{
                0.01165623095604, 0.031684920796124, 0.086128544436269,
                0.23412165725274, 0.63640864655883};
            dlmath::softmax<num_t>(test_vec.data(), test_vec.data(), 
                test_vec.size());
            for (size_t i = 0; i < truth_vec.size(); ++i)
            {
                ARIADNE_TEST_PRINT(std::to_string(i) + ": " 
                    + std::to_string(test_vec[i]));
                ARIADNE_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000000001);
            }
        }

        {
            ARIADNE_PRINT_TEST_COMMENT("Test ReLU First Derivative");
            std::vector<num_t> test_vec{-2,-1,0,1,2};
            std::vector<num_t> truth_vec{0,0,0,1,1};
            dlmath::relu_1<num_t>(test_vec.data(), test_vec.data(), 
                test_vec.size());
            for (size_t i = 0; i < truth_vec.size(); ++i)
            {
                ARIADNE_TEST_PRINT(std::to_string(i) + ": " 
                    + std::to_string(test_vec[i]));
                ARIADNE_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000000001);
            }
        }

        {
            ARIADNE_PRINT_TEST_COMMENT("Test SoftMax First Derivative");
            std::vector<num_t> test_vec{-2,-1,0,1,2};
            ARIADNE_TEST_FAIL(dlmath::softmax_1_opt<num_t>(test_vec.data(), 
                test_vec.data(), test_vec.size()));
            ARIADNE_TEST_EXECUTE(dlmath::softmax_1<num_t>(test_vec.data(), 
                test_vec.data(), test_vec.size()));
            for (size_t i = 0; i < test_vec.size(); ++i)
            {
                std::cout << std::fixed << std::setprecision(25) 
                    << "test_vec[i]: " << test_vec[i] << std::endl;
            }
        }
    }

};

int main() {
    TestDnn().test();
    return ARIADNE_TEST_FAILURES;
}



