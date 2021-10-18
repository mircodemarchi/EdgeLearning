/***************************************************************************
 *            tests/test_dlmath.cpp
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
#include "dnn/type.hpp"
#include "dnn/dlmath.hpp"
#include "dnn/layer.hpp"
#include "dnn/model.hpp"

#include <vector>
#include <iostream>
#include <iomanip>
#include <random>

using namespace std;
using namespace EdgeLearning;

class TestDLMath {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_normal_pdf());
        EDGE_LEARNING_TEST_CALL(test_arr_sum());
        EDGE_LEARNING_TEST_CALL(test_arr_mul());
        EDGE_LEARNING_TEST_CALL(test_matarr_mul());
        EDGE_LEARNING_TEST_CALL(test_relu());
        EDGE_LEARNING_TEST_CALL(test_softmax());
        EDGE_LEARNING_TEST_CALL(test_relu_1());
        EDGE_LEARNING_TEST_CALL(test_softmax_1());
        EDGE_LEARNING_TEST_CALL(test_cross_entropy());
        EDGE_LEARNING_TEST_CALL(test_cross_entropy_1());
        EDGE_LEARNING_TEST_CALL(test_mean_squared_error());
        EDGE_LEARNING_TEST_CALL(test_mean_squared_error_1());
        EDGE_LEARNING_TEST_CALL(test_max_argmax());
        EDGE_LEARNING_TEST_CALL(test_tanh());
        EDGE_LEARNING_TEST_CALL(test_tanh_1());
    }

private:
    const RneType::result_type SEED = 1;
    const size_t PRINT_TIMES = 4;

    void test_normal_pdf() {
        RneType generator{SEED};
        auto dist = DLMath::normal_pdf<NumType>(0.0, 0.1);
        for (size_t i = 0; i < PRINT_TIMES; ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(dist(generator)));
        }

        std::random_device rd;
        generator = RneType{rd()};
        size_t gt1_count = 0, lt1_count = 0;
        for (size_t i = 0; i < 10000; ++i)
        {
            if (dist(generator) > 0.0)
            {
                ++gt1_count;
            }
            else 
            {
                ++lt1_count;
            }
        }
        EDGE_LEARNING_TEST_PRINT("Normal distribution >0 count similar to <=0 count:"
            + std::to_string(gt1_count) + ", " + std::to_string(lt1_count));
    }

    void test_arr_sum() {
        std::vector<int> test_vec1{5,4,3,2,1};
        std::vector<int> test_vec2{1,2,3,4,5};
        std::vector<int> truth_vec{6,6,6,6,6};
        DLMath::arr_sum<int>(test_vec1.data(), test_vec1.data(), 
            test_vec2.data(), test_vec1.size());
        for (size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(test_vec1[i], truth_vec[i]);
        }
    }

    void test_arr_mul() {
        std::vector<int> test_vec1{5,4,3,2,1};
        std::vector<int> test_vec2{1,2,3,4,5};
        std::vector<int> truth_vec{5,8,9,8,5};
        DLMath::arr_mul<int>(test_vec1.data(), test_vec1.data(), 
            test_vec2.data(), test_vec1.size());
        for (size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(test_vec1[i], truth_vec[i]);
        }
    }

    void test_matarr_mul() {
        std::vector<int> test_mat{1,2,3,4};
        std::vector<int> test_vec{1,2};
        std::vector<int> truth_vec{5,11};
        EDGE_LEARNING_TEST_FAIL(
            DLMath::matarr_mul<int>(test_vec.data(), test_mat.data(), 
                                    test_vec.data(), 2, 2)
        );
        std::vector<int> res_vec; res_vec.resize(test_vec.size());
        DLMath::matarr_mul<int>(res_vec.data(), test_mat.data(), 
                                test_vec.data(), 2, 2);
        for (size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(res_vec[i], truth_vec[i]);
        }
    }

    void test_relu() {
        std::vector<NumType> test_vec{-2,-1,0,1,2};
        std::vector<NumType> truth_vec{0,0,0,1,2};
        DLMath::relu<NumType>(test_vec.data(), test_vec.data(), 
            test_vec.size());
        for (size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(test_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000000001);
        }
    }

    void test_softmax() 
    {
        std::vector<NumType> test_vec{-2,-1,0,1,2};
        std::vector<NumType> truth_vec{
            0.01165623095604, 0.031684920796124, 0.086128544436269,
            0.23412165725274, 0.63640864655883};
        DLMath::softmax<NumType>(test_vec.data(), test_vec.data(), 
            test_vec.size());
        for (size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(test_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000000001);
        }
    }

    void test_relu_1() {
        std::vector<NumType> test_vec{-2,-1,0,1,2};
        std::vector<NumType> truth_vec{0,0,0,1,1};
        DLMath::relu_1<NumType>(test_vec.data(), test_vec.data(), 
            test_vec.size());
        for (size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(test_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000000001);
        }
    }

    void test_softmax_1() {
        std::vector<NumType> test_vec{-2.0,-1.0,0.0,1.0,2.0};
        EDGE_LEARNING_TEST_FAIL(DLMath::softmax_1_opt<NumType>(test_vec.data(), 
            test_vec.data(), test_vec.size()));
        EDGE_LEARNING_TEST_EXECUTE(DLMath::softmax_1<NumType>(test_vec.data(), 
            test_vec.data(), test_vec.size()));
        for (size_t i = 0; i < test_vec.size(); ++i)
        {
            std::cout << std::fixed << std::setprecision(40) 
                << "test_vec[i]: " << test_vec[i] << std::endl << std::endl;
        }
    }

    void test_cross_entropy() {
        std::vector<NumType> test_y    {0.0, 0.0, 0.00, 0.00, 1.0};
        std::vector<NumType> test_y_hat{0.1, 0.1, 0.25, 0.05, 0.5};
        NumType truth_ce = 0.6931471805599453;
        auto ret = DLMath::cross_entropy(test_y.data(), test_y_hat.data(), 
            test_y_hat.size());
        EDGE_LEARNING_TEST_WITHIN(ret, truth_ce, 0.00000000001);

        NumType test_val = 0.5;
        NumType truth_val = 0.34657359027997264;
        ret = DLMath::cross_entropy(test_val, test_val);
        EDGE_LEARNING_TEST_WITHIN(ret, truth_val, 0.00000000001);
    }

    void test_cross_entropy_1() {
        std::vector<NumType> test_y    {0.0, 0.0, 0.00, 0.00, 1.0};
        std::vector<NumType> test_y_hat{0.1, 0.1, 0.25, 0.05, 0.5};
        std::vector<NumType> truth_ce1 {0.0, 0.0, 0.00, 0.00, -2.0};
        std::vector<NumType> ret_vec; ret_vec.resize(truth_ce1.size());
        DLMath::cross_entropy_1(ret_vec.data(), test_y.data(), 
            test_y_hat.data(), 1.0, test_y_hat.size());
        for (size_t i = 0; i < truth_ce1.size(); ++i)
        {
            EDGE_LEARNING_TEST_WITHIN(ret_vec[i], truth_ce1[i], 0.00000000001);
        }
        

        NumType test_val = 0.5;
        NumType truth_val = -1.0;
        auto ret_val = DLMath::cross_entropy_1(test_val, test_val, 1.0);
        EDGE_LEARNING_TEST_WITHIN(ret_val, truth_val, 0.00000000001);
    }

    void test_mean_squared_error() {
        NumType test_val = 1.0;
        NumType truth_val = 0.0;
        auto ret = DLMath::squared_error(test_val, test_val);
        EDGE_LEARNING_TEST_WITHIN(ret, truth_val, 0.00000000001);

        std::vector<NumType> test_y    {1.0, 1.0, 1.0, 1.0, 1.0};
        std::vector<NumType> test_y_hat{1.1, 0.1, 1.2, 1.5, 0.5};
        NumType truth_mse = 0.272;
        ret = DLMath::mean_squared_error(test_y.data(), test_y_hat.data(), 
            test_y_hat.size());
        EDGE_LEARNING_TEST_WITHIN(ret, truth_mse, 0.00000000001);
    }

    void test_mean_squared_error_1() {
        NumType test_val1 = 1.0;
        NumType test_val2 = 1.5;
        NumType truth_val = 0.5;
        auto ret = DLMath::squared_error_1(test_val1, test_val2, 0.5);
        EDGE_LEARNING_TEST_WITHIN(ret, truth_val, 0.00000000001);

        std::vector<NumType> test_y    {1.0, 1.0, 1.0, 1.0, 1.0};
        std::vector<NumType> test_y_hat{1.1, 0.1, 1.2, 1.5, 0.5};
        std::vector<NumType> truth_mse1 {0.2, -1.8, 0.4, 1.0, -1.0};
        std::vector<NumType> ret_vec; ret_vec.resize(truth_mse1.size());
        DLMath::mean_squared_error_1(ret_vec.data(), test_y.data(), 
            test_y_hat.data(), 1.0, test_y_hat.size());
        for (size_t i = 0; i < truth_mse1.size(); ++i)
        {
            EDGE_LEARNING_TEST_WITHIN(ret_vec[i], truth_mse1[i], 0.00000000001);
        }
    }

    void test_max_argmax() {
        std::vector<NumType> test_vec{0,1,5,4,3};
        NumType truth_max = 5;
        NumType ret_max = DLMath::max<NumType>(test_vec.data(), test_vec.size());
        EDGE_LEARNING_TEST_EQUAL(ret_max, truth_max);

        NumType truth_argmax = 2;
        NumType ret_argmax = DLMath::argmax<NumType>(test_vec.data(), 
            test_vec.size());
        EDGE_LEARNING_TEST_EQUAL(ret_argmax, truth_argmax);

        auto ret_tuple = DLMath::max_and_argmax<NumType>(test_vec.data(), 
            test_vec.size());
        EDGE_LEARNING_TEST_EQUAL(std::get<0>(ret_tuple), truth_max);
        EDGE_LEARNING_TEST_EQUAL(std::get<1>(ret_tuple), truth_argmax);
    }

    void test_tanh() {
        std::vector<NumType> test_vec{-10.0, 0.0, 1.0, 7.0, 10000.0};
        std::vector<NumType> truth_vec{-1.0, 0.0, 0.76159416, 0.99999834, 1.0};
        DLMath::tanh<NumType>(test_vec.data(), test_vec.data(), 
            test_vec.size());
        for (size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(test_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000001);
        }
    }

    void test_tanh_1() {
        std::vector<NumType> test_vec{-10.0, 0.0, 1.0, 7.0, 10000.0};
        std::vector<NumType> truth_vec{8.24461455e-09, 1.00000000e+00, 
            4.19974342e-01, 3.32610934e-06, 0.00000000e+00};
        DLMath::tanh_1<NumType>(test_vec.data(), test_vec.data(), 
            test_vec.size());
        for (size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(test_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000001);
        }
    }
};

int main() {
    TestDLMath().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
