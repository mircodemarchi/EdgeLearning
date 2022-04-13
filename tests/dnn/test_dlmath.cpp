/***************************************************************************
 *            dnn/test_dlmath.cpp
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
#include "type.hpp"
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
        EDGE_LEARNING_TEST_CALL(test_unique());
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
        EDGE_LEARNING_TEST_CALL(test_conv2d());
        EDGE_LEARNING_TEST_CALL(test_max_pool());
        EDGE_LEARNING_TEST_CALL(test_avg_pool());
    }

private:
    const RneType::result_type SEED = 1;
    const std::size_t PRINT_TIMES = 4;

    void test_normal_pdf() {
        RneType generator{SEED};
        auto dist = DLMath::normal_pdf<NumType>(0.0, 0.1);
        for (std::size_t i = 0; i < PRINT_TIMES; ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(dist(generator)));
        }

        std::random_device rd;
        generator = RneType{rd()};
        std::size_t gt1_count = 0, lt1_count = 0;
        for (std::size_t i = 0; i < 10000; ++i)
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

    void test_unique() {
        for (std::size_t i = 0; i < 100; ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(DLMath::unique(), i);
        }
    }

    void test_arr_sum() {
        std::vector<int> test_vec1{5,4,3,2,1};
        std::vector<int> test_vec2{1,2,3,4,5};
        std::vector<int> truth_vec{6,6,6,6,6};
        DLMath::arr_sum<int>(test_vec1.data(), test_vec1.data(), 
            test_vec2.data(), test_vec1.size());
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
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
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
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
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(res_vec[i], truth_vec[i]);
        }
    }

    void test_relu() {
        std::vector<NumType> test_vec{-2,-1,0,1,2};
        std::vector<NumType> truth_vec{0,0,0,1,2};
        DLMath::relu<NumType>(test_vec.data(), test_vec.data(), 
            test_vec.size());
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
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
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
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
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
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
        for (std::size_t i = 0; i < test_vec.size(); ++i)
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
        for (std::size_t i = 0; i < truth_ce1.size(); ++i)
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
        for (std::size_t i = 0; i < truth_mse1.size(); ++i)
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
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
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
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(test_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000001);
        }
    }

    void test_conv2d() {
        SizeType input_width = 3;
        SizeType input_height = 3;
        SizeType f = 2;
        SizeType output_width = 2;
        SizeType output_height = 2;
        std::vector<NumType> test_img{
            0, 1, 2,
            3, 4, 5,
            6, 7, 8.5
        };
        std::vector<NumType> test_k{
            0, 0,
            0, 1
        };
        std::vector<NumType> truth_vec{
            4, 5,
            7, 8.5
        };
        std::vector<NumType> result(truth_vec.size());
        DLMath::conv2d<NumType>(result.data(),
                                test_img.data(), input_width, input_height,
                                test_k.data(), f);
        for (std::size_t r = 0; r < output_height; ++r)
        {
            for (std::size_t c = 0; c < output_width; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(
                    "[" + std::to_string(r) + "," + std::to_string(c) + "] "
                    + std::to_string(result[r * output_width + c]));
                EDGE_LEARNING_TEST_WITHIN(
                    result[r * output_width + c],
                    truth_vec[r * output_width + c], 0.0000000000001);
            }
        }

        output_width = 4;
        output_height = 4;
        truth_vec = std::vector<NumType>{
            0, 1, 2,   0,
            3, 4, 5,   0,
            6, 7, 8.5, 0,
            0, 0, 0,   0
        };
        result.resize(truth_vec.size());
        DLMath::conv2d<NumType>(result.data(),
                                test_img.data(), input_width, input_height,
                                test_k.data(), f, 1, 1);
        for (std::size_t r = 0; r < output_height; ++r)
        {
            for (std::size_t c = 0; c < output_width; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(
                    "[" + std::to_string(r) + "," + std::to_string(c) + "] "
                    + std::to_string(result[r * output_width + c]));
                EDGE_LEARNING_TEST_WITHIN(
                    result[r * output_width + c],
                    truth_vec[r * output_width + c], 0.0000000000001);
            }
        }

        output_width = 2;
        output_height = 2;
        truth_vec = std::vector<NumType>{
            0, 2,
            6, 8.5
        };
        result.resize(truth_vec.size());
        DLMath::conv2d<NumType>(result.data(),
                                test_img.data(), input_width, input_height,
                                test_k.data(), f, 2, 1);
        for (std::size_t r = 0; r < output_height; ++r)
        {
            for (std::size_t c = 0; c < output_width; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(
                    "[" + std::to_string(r) + "," + std::to_string(c) + "] "
                    + std::to_string(result[r * output_width + c]));
                EDGE_LEARNING_TEST_WITHIN(
                    result[r * output_width + c],
                    truth_vec[r * output_width + c], 0.0000000000001);
            }
        }

        input_width = 5;
        input_height = 4;
        f = 3;
        output_width = 3;
        output_height = 2;
        test_img = std::vector<NumType>{
            0,  1,  2,  4,  5,
            3,  4,  5,  6,  7,
            6,  7,  8,  9,  10,
            9,  10, 11, 12, 13
        };
        test_k = std::vector<NumType>{
            0, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        truth_vec = std::vector<NumType>{
            12, 14, 16,
            18, 20, 22
        };
        result.resize(truth_vec.size());
        DLMath::conv2d<NumType>(result.data(),
                                test_img.data(), input_width, input_height,
                                test_k.data(), f);
        for (std::size_t r = 0; r < output_height; ++r)
        {
            for (std::size_t c = 0; c < output_width; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(
                    "[" + std::to_string(r) + "," + std::to_string(c) + "] "
                    + std::to_string(result[r * output_width + c]));
                EDGE_LEARNING_TEST_WITHIN(
                    result[r * output_width + c],
                    truth_vec[r * output_width + c], 0.0000000000001);
            }
        }

        output_width = 5;
        output_height = 4;
        truth_vec = std::vector<NumType>{
            4,  6,  8,  11,  5,
            10, 12, 14, 16,  7,
            16, 18, 20, 22, 10,
            9,  10, 11, 12, 13
        };
        result.resize(truth_vec.size());
        DLMath::conv2d<NumType>(result.data(),
                                test_img.data(), input_width, input_height,
                                test_k.data(), f, 1, 1);
        for (std::size_t r = 0; r < output_height; ++r)
        {
            for (std::size_t c = 0; c < output_width; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(
                    "[" + std::to_string(r) + "," + std::to_string(c) + "] "
                    + std::to_string(result[r * output_width + c]));
                EDGE_LEARNING_TEST_WITHIN(
                    result[r * output_width + c],
                    truth_vec[r * output_width + c], 0.0000000000001);
            }
        }

        output_width = 3;
        output_height = 2;
        truth_vec = std::vector<NumType>{
            4,  8,   5,
            16, 20, 10,
        };
        result.resize(truth_vec.size());
        DLMath::conv2d<NumType>(result.data(),
                                test_img.data(), input_width, input_height,
                                test_k.data(), f, 2, 1);
        for (std::size_t r = 0; r < output_height; ++r)
        {
            for (std::size_t c = 0; c < output_width; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(
                    "[" + std::to_string(r) + "," + std::to_string(c) + "] "
                    + std::to_string(result[r * output_width + c]));
                EDGE_LEARNING_TEST_WITHIN(
                    result[r * output_width + c],
                    truth_vec[r * output_width + c], 0.0000000000001);
            }
        }
    }

    void test_max_pool() {
        SizeType input_width = 3;
        SizeType input_height = 3;
        SizeType f = 2;
        SizeType output_width = 2;
        SizeType output_height = 2;
        std::vector<NumType> test_img{
            10, 1, 2,
            3,  4, 5,
            6,  7, 8.5
        };
        std::vector<NumType> truth_vec{
            10, 5,
            7,  8.5
        };
        std::vector<NumType> result(truth_vec.size());
        DLMath::max_pool<NumType>(
            result.data(), test_img.data(), input_width, input_height, f);
        for (std::size_t r = 0; r < output_height; ++r) {
            for (std::size_t c = 0; c < output_width; ++c) {
                EDGE_LEARNING_TEST_PRINT(
                    "[" + std::to_string(r) + "," + std::to_string(c) + "] "
                    + std::to_string(result[r * output_width + c]));
                EDGE_LEARNING_TEST_WITHIN(
                    result[r * output_width + c],
                    truth_vec[r * output_width + c], 0.0000000000001);
            }
        }

        input_width = 5;
        input_height = 4;
        f = 3;
        output_width = 3;
        output_height = 2;
        test_img = std::vector<NumType>{
            10,  1,  2,  4,  5,
            3,   4,  5,  6,  7,
            6,   7,  8,  9,  10,
            9,   10, 11, 12, 13
        };
        truth_vec = std::vector<NumType>{
            10, 9,  10,
            11, 12, 13
        };
        result.resize(truth_vec.size());
        DLMath::max_pool<NumType>(
            result.data(), test_img.data(), input_width, input_height, f);
        for (std::size_t r = 0; r < output_height; ++r) {
            for (std::size_t c = 0; c < output_width; ++c) {
                EDGE_LEARNING_TEST_PRINT(
                    "[" + std::to_string(r) + "," + std::to_string(c) + "] "
                    + std::to_string(result[r * output_width + c]));
                EDGE_LEARNING_TEST_WITHIN(
                    result[r * output_width + c],
                    truth_vec[r * output_width + c], 0.0000000000001);
            }
        }

        output_width = 2;
        output_height = 1;
        truth_vec = std::vector<NumType>{
            10, 10
        };
        result.resize(truth_vec.size());
        DLMath::max_pool<NumType>(
            result.data(), test_img.data(), input_width, input_height, f, 2);
        for (std::size_t r = 0; r < output_height; ++r) {
            for (std::size_t c = 0; c < output_width; ++c) {
                EDGE_LEARNING_TEST_PRINT(
                    "[" + std::to_string(r) + "," + std::to_string(c) + "] "
                    + std::to_string(result[r * output_width + c]));
                EDGE_LEARNING_TEST_WITHIN(
                    result[r * output_width + c],
                    truth_vec[r * output_width + c], 0.0000000000001);
            }
        }
    }

    void test_avg_pool() {
        SizeType input_width = 3;
        SizeType input_height = 3;
        SizeType f = 2;
        SizeType output_width = 2;
        SizeType output_height = 2;
        std::vector<NumType> test_img{
            10, 1, 2,
            3,  4, 5,
            6,  7, 8.5
        };
        std::vector<NumType> truth_vec{
            4.5, 3,
            5,   6.125
        };
        std::vector<NumType> result(truth_vec.size());
        DLMath::avg_pool<NumType>(
            result.data(), test_img.data(), input_width, input_height, f);
        for (std::size_t r = 0; r < output_height; ++r) {
            for (std::size_t c = 0; c < output_width; ++c) {
                EDGE_LEARNING_TEST_PRINT(
                    "[" + std::to_string(r) + "," + std::to_string(c) + "] "
                    + std::to_string(result[r * output_width + c]));
                EDGE_LEARNING_TEST_WITHIN(
                    result[r * output_width + c],
                    truth_vec[r * output_width + c], 0.0000000000001);
            }
        }

        input_width = 5;
        input_height = 4;
        f = 3;
        output_width = 3;
        output_height = 2;
        test_img = std::vector<NumType>{
            10,  1,  2,  4,  5,
            3,   4,  5,  6,  7,
            6,   7,  8,  9,  10,
            9,   10, 11, 12, 13
        };
        truth_vec = std::vector<NumType>{
            46.0/9, 46.0/9, 56.0/9,
            63.0/9, 72.0/9, 81.0/9
        };
        result.resize(truth_vec.size());
        DLMath::avg_pool<NumType>(
            result.data(), test_img.data(), input_width, input_height, f);
        for (std::size_t r = 0; r < output_height; ++r) {
            for (std::size_t c = 0; c < output_width; ++c) {
                EDGE_LEARNING_TEST_PRINT(
                    "[" + std::to_string(r) + "," + std::to_string(c) + "] "
                    + std::to_string(result[r * output_width + c]));
                EDGE_LEARNING_TEST_WITHIN(
                    result[r * output_width + c],
                    truth_vec[r * output_width + c], 0.0000000000001);
            }
        }

        output_width = 2;
        output_height = 1;
        truth_vec = std::vector<NumType>{
            46.0/9, 56.0/9
        };
        result.resize(truth_vec.size());
        DLMath::avg_pool<NumType>(
            result.data(), test_img.data(), input_width, input_height, f, 2);
        for (std::size_t r = 0; r < output_height; ++r) {
            for (std::size_t c = 0; c < output_width; ++c) {
                EDGE_LEARNING_TEST_PRINT(
                    "[" + std::to_string(r) + "," + std::to_string(c) + "] "
                    + std::to_string(result[r * output_width + c]));
                EDGE_LEARNING_TEST_WITHIN(
                    result[r * output_width + c],
                    truth_vec[r * output_width + c], 0.0000000000001);
            }
        }
    }
};

int main() {
    TestDLMath().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
