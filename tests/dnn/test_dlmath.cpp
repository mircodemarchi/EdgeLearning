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
    using TestNumType = double;

    void test() {
        EDGE_LEARNING_TEST_CALL(test_shape());
        EDGE_LEARNING_TEST_CALL(test_index_of());
        EDGE_LEARNING_TEST_CALL(test_normal_pdf());
        EDGE_LEARNING_TEST_CALL(test_uniform_pdf());
        EDGE_LEARNING_TEST_CALL(test_pdf());
        EDGE_LEARNING_TEST_CALL(test_kaiming_initialization());
        EDGE_LEARNING_TEST_CALL(test_xavier_initialization());
        EDGE_LEARNING_TEST_CALL(test_initialization());
        EDGE_LEARNING_TEST_CALL(test_initialization_pdf());
        EDGE_LEARNING_TEST_CALL(test_unique());
        EDGE_LEARNING_TEST_CALL(test_arr_sum());
        EDGE_LEARNING_TEST_CALL(test_arr_mul());
        EDGE_LEARNING_TEST_CALL(test_matarr_mul());
        EDGE_LEARNING_TEST_CALL(test_relu());
        EDGE_LEARNING_TEST_CALL(test_relu_1());
        EDGE_LEARNING_TEST_CALL(test_elu());
        EDGE_LEARNING_TEST_CALL(test_elu_1());
        EDGE_LEARNING_TEST_CALL(test_tanh());
        EDGE_LEARNING_TEST_CALL(test_tanh_1());
        EDGE_LEARNING_TEST_CALL(test_sigmoid());
        EDGE_LEARNING_TEST_CALL(test_sigmoid_1());
        EDGE_LEARNING_TEST_CALL(test_softmax());
        EDGE_LEARNING_TEST_CALL(test_softmax_1());
        EDGE_LEARNING_TEST_CALL(test_cross_entropy());
        EDGE_LEARNING_TEST_CALL(test_cross_entropy_1());
        EDGE_LEARNING_TEST_CALL(test_mean_squared_error());
        EDGE_LEARNING_TEST_CALL(test_mean_squared_error_1());
        EDGE_LEARNING_TEST_CALL(test_max_argmax());
        EDGE_LEARNING_TEST_CALL(test_cross_correlation_without_channels());
        EDGE_LEARNING_TEST_CALL(test_cross_correlation_with_channels());
        EDGE_LEARNING_TEST_CALL(
            test_cross_correlation_with_channels_with_filters());
        EDGE_LEARNING_TEST_CALL(test_max_pool());
        EDGE_LEARNING_TEST_CALL(test_avg_pool());
        EDGE_LEARNING_TEST_CALL(test_append());
        EDGE_LEARNING_TEST_CALL(test_extract());
        EDGE_LEARNING_TEST_CALL(test_concatenate());
        EDGE_LEARNING_TEST_CALL(test_separate());
    }

private:
    const RneType::result_type SEED = 1;
    const std::size_t PRINT_TIMES = 10;

    void test_index_of() {
        std::vector<TestNumType> test_vec = {1,2,3,4,5};
        TestNumType test_e = 3;
        std::int64_t truth_val = 2;
        EDGE_LEARNING_TEST_EQUAL(DLMath::index_of(test_vec, test_e), truth_val);

        test_e = 6;
        truth_val = -1;
        EDGE_LEARNING_TEST_EQUAL(DLMath::index_of(test_vec, test_e), truth_val);
    }

    void test_shape() {
        SizeType h = 10;
        SizeType w = 11;
        DLMath::Shape2d shape_2d(h, w);
        EDGE_LEARNING_TEST_EQUAL(shape_2d.height(), h);
        EDGE_LEARNING_TEST_EQUAL(shape_2d.width(), w);
        EDGE_LEARNING_TEST_EQUAL(
                static_cast<std::vector<SizeType>>(shape_2d).size(), 2);
        EDGE_LEARNING_TEST_EQUAL(
                static_cast<std::vector<SizeType>>(shape_2d)[0], h);
        EDGE_LEARNING_TEST_EQUAL(
                static_cast<std::vector<SizeType>>(shape_2d)[1], w);
        EDGE_LEARNING_TEST_EQUAL(shape_2d[0], h);
        EDGE_LEARNING_TEST_EQUAL(shape_2d[1], w);
        EDGE_LEARNING_TEST_FAIL((void) shape_2d.at(2));
        EDGE_LEARNING_TEST_THROWS((void) shape_2d.at(2), std::runtime_error);

        h = 12;
        w = 13;
        SizeType c = 3;
        DLMath::Shape3d shape_3d(h, w, c);
        EDGE_LEARNING_TEST_EQUAL(shape_3d.height(), h);
        EDGE_LEARNING_TEST_EQUAL(shape_3d.width(), w);
        EDGE_LEARNING_TEST_EQUAL(shape_3d.channels(), c);
        EDGE_LEARNING_TEST_EQUAL(
                static_cast<std::vector<SizeType>>(shape_3d).size(), 3);
        EDGE_LEARNING_TEST_EQUAL(
                static_cast<std::vector<SizeType>>(shape_3d)[0], h);
        EDGE_LEARNING_TEST_EQUAL(
                static_cast<std::vector<SizeType>>(shape_3d)[1], w);
        EDGE_LEARNING_TEST_EQUAL(
                static_cast<std::vector<SizeType>>(shape_3d)[2], c);
        EDGE_LEARNING_TEST_EQUAL(shape_3d[0], h);
        EDGE_LEARNING_TEST_EQUAL(shape_3d[1], w);
        EDGE_LEARNING_TEST_EQUAL(shape_3d[2], c);
        EDGE_LEARNING_TEST_FAIL((void) shape_3d.at(3));
        EDGE_LEARNING_TEST_THROWS((void) shape_3d.at(3), std::runtime_error);
    }

    void test_normal_pdf() {
        std::random_device rd;
        RneType generator{rd()};
        auto dist = DLMath::normal_pdf<TestNumType>(0.0, 0.1);
        for (std::size_t i = 0; i < PRINT_TIMES; ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(dist(generator)));
        }

        generator = RneType{SEED};
        std::int64_t gt1_count = 0, lt1_count = 0;
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
        EDGE_LEARNING_TEST_PRINT(
            "Normal distribution >0 count similar to <=0 count:"
            + std::to_string(gt1_count) + ", " + std::to_string(lt1_count));
        EDGE_LEARNING_TEST_WITHIN(gt1_count, lt1_count, 200);
    }

    void test_uniform_pdf() {
        auto const range = 0.2;
        std::random_device rd;
        RneType generator{rd()};
        auto dist = DLMath::uniform_pdf<TestNumType>(0.0, range);
        for (std::size_t i = 0; i < PRINT_TIMES; ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(dist(generator)));
            EDGE_LEARNING_TEST_ASSERT(dist(generator) <= (range / 2));
            EDGE_LEARNING_TEST_ASSERT(dist(generator) >= -(range / 2));
        }

        generator = RneType{SEED};
        std::int64_t gt1_count = 0, lt1_count = 0;
        for (std::size_t i = 0; i < 10000; ++i)
        {
            if (dist(generator) > 0.0) ++gt1_count;
            else ++lt1_count;
        }
        EDGE_LEARNING_TEST_PRINT(
            "Normal distribution >0 count similar to <=0 count:"
            + std::to_string(gt1_count) + ", " + std::to_string(lt1_count));
        EDGE_LEARNING_TEST_WITHIN(gt1_count, lt1_count, 200);
    }

    void test_pdf() {
        EDGE_LEARNING_TEST_FAIL(
            DLMath::pdf<TestNumType>(
                0.0, 0.1, static_cast<DLMath::ProbabilityDensityFunction>(-1)));
        EDGE_LEARNING_TEST_THROWS(
            DLMath::pdf<TestNumType>(
                0.0, 0.1, static_cast<DLMath::ProbabilityDensityFunction>(-1)),
                std::runtime_error);

        std::random_device rd;
        RneType generator{rd()};
        auto dist = DLMath::pdf<TestNumType>(
            0.0, 0.1, DLMath::ProbabilityDensityFunction::NORMAL);
        for (std::size_t i = 0; i < PRINT_TIMES; ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(dist(generator)));
        }

        dist = DLMath::pdf<TestNumType>(
            0.0, 0.1, DLMath::ProbabilityDensityFunction::UNIFORM);
        for (std::size_t i = 0; i < PRINT_TIMES; ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(dist(generator)));
        }
    }

    void test_kaiming_initialization()
    {
        int truth_mean = 0;
        EDGE_LEARNING_TEST_EQUAL(
            DLMath::kaiming_initialization_mean<int>(), truth_mean);
        EDGE_LEARNING_TEST_EQUAL(
            std::get<0>(DLMath::kaiming_initialization<int>(100)), truth_mean);

        SizeType n = 10;
        TestNumType truth_variance = 0.4472135954999579;
        EDGE_LEARNING_TEST_WITHIN(
            DLMath::kaiming_initialization_variance<TestNumType>(n),
            truth_variance, 0.000000000000001);
        EDGE_LEARNING_TEST_WITHIN(
            std::get<1>(DLMath::kaiming_initialization<TestNumType>(n)),
            truth_variance, 0.000000000000001);

        n = 1;
        truth_variance = 1.4142135623730951;
        EDGE_LEARNING_TEST_WITHIN(
            DLMath::kaiming_initialization_variance<TestNumType>(n),
            truth_variance, 0.000000000000001);
        EDGE_LEARNING_TEST_WITHIN(
            std::get<1>(DLMath::kaiming_initialization<TestNumType>(n)),
            truth_variance, 0.000000000000001);

        n = 2;
        truth_variance = 1.0;
        EDGE_LEARNING_TEST_WITHIN(
            DLMath::kaiming_initialization_variance<TestNumType>(n),
            truth_variance, 0.000000000000001);
        EDGE_LEARNING_TEST_WITHIN(
            std::get<1>(DLMath::kaiming_initialization<TestNumType>(n)),
            truth_variance, 0.000000000000001);
    }

    void test_xavier_initialization()
    {
        int truth_mean = 0;
        EDGE_LEARNING_TEST_EQUAL(
            DLMath::xavier_initialization_mean<int>(), truth_mean);
        EDGE_LEARNING_TEST_EQUAL(
            std::get<0>(DLMath::xavier_initialization<int>(100)), truth_mean);

        SizeType n = 10;
        TestNumType truth_variance = 0.31622776601683794;
        EDGE_LEARNING_TEST_WITHIN(
            DLMath::xavier_initialization_variance<TestNumType>(n),
            truth_variance, 0.000000000000001);
        EDGE_LEARNING_TEST_WITHIN(
            std::get<1>(DLMath::xavier_initialization<TestNumType>(n)),
            truth_variance, 0.000000000000001);

        n = 1;
        truth_variance = 1.0;
        EDGE_LEARNING_TEST_WITHIN(
            DLMath::xavier_initialization_variance<TestNumType>(n),
            truth_variance, 0.000000000000001);
        EDGE_LEARNING_TEST_WITHIN(
            std::get<1>(DLMath::xavier_initialization<TestNumType>(n)),
            truth_variance, 0.000000000000001);

        n = 2;
        truth_variance = 0.7071067811865476;
        EDGE_LEARNING_TEST_WITHIN(
            DLMath::xavier_initialization_variance<TestNumType>(n),
            truth_variance, 0.000000000000001);
        EDGE_LEARNING_TEST_WITHIN(
            std::get<1>(DLMath::xavier_initialization<TestNumType>(n)),
            truth_variance, 0.000000000000001);
    }

    void test_initialization()
    {
        int truth_mean = 0;
        auto kaiming_mean = std::get<0>(DLMath::initialization<int>(
            DLMath::InitializationFunction::KAIMING, 100));
        auto xavier_mean = std::get<0>(DLMath::initialization<int>(
            DLMath::InitializationFunction::XAVIER, 100));
        EDGE_LEARNING_TEST_EQUAL(kaiming_mean, truth_mean);
        EDGE_LEARNING_TEST_EQUAL(xavier_mean, truth_mean);

        SizeType n = 10;
        auto kaiming_variance = std::get<1>(
            DLMath::initialization<TestNumType>(
                DLMath::InitializationFunction::KAIMING, n));
        TestNumType kaiming_truth_variance = 0.4472135954999579;
        auto xavier_variance = std::get<1>(
            DLMath::initialization<TestNumType>(
                DLMath::InitializationFunction::XAVIER, n));
        TestNumType xavier_truth_variance = 0.31622776601683794;
        EDGE_LEARNING_TEST_WITHIN(kaiming_variance, kaiming_truth_variance,
                                  0.000000000000001);
        EDGE_LEARNING_TEST_WITHIN(xavier_variance, xavier_truth_variance,
                                  0.000000000000001);

        EDGE_LEARNING_TEST_FAIL(
            DLMath::initialization<TestNumType>(
                static_cast<DLMath::InitializationFunction>(-1), n));
        EDGE_LEARNING_TEST_THROWS(
            DLMath::initialization<TestNumType>(
                static_cast<DLMath::InitializationFunction>(-1), n),
                std::runtime_error);
    }

    void test_initialization_pdf()
    {
        SizeType n = 10;

        EDGE_LEARNING_TEST_FAIL(
            DLMath::initialization_pdf<TestNumType>(
                DLMath::InitializationFunction::KAIMING,
                static_cast<DLMath::ProbabilityDensityFunction>(-1), n));
        EDGE_LEARNING_TEST_THROWS(
            DLMath::initialization_pdf<TestNumType>(
                DLMath::InitializationFunction::KAIMING,
                static_cast<DLMath::ProbabilityDensityFunction>(-1), n),
            std::runtime_error);

        EDGE_LEARNING_TEST_FAIL(
            DLMath::initialization_pdf<TestNumType>(
                static_cast<DLMath::InitializationFunction>(-1),
                DLMath::ProbabilityDensityFunction::NORMAL, n));
        EDGE_LEARNING_TEST_THROWS(
            DLMath::initialization_pdf<TestNumType>(
                static_cast<DLMath::InitializationFunction>(-1),
                DLMath::ProbabilityDensityFunction::NORMAL, n),
            std::runtime_error);

        std::random_device rd;
        RneType generator{rd()};
        EDGE_LEARNING_TEST_TRY(
            DLMath::initialization_pdf<TestNumType>(
                DLMath::InitializationFunction::KAIMING,
                DLMath::ProbabilityDensityFunction::NORMAL, n));
        auto dist = DLMath::initialization_pdf<TestNumType>(
            DLMath::InitializationFunction::KAIMING,
            DLMath::ProbabilityDensityFunction::NORMAL, n);
        for (std::size_t i = 0; i < PRINT_TIMES; ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(dist(generator)));
        }

        EDGE_LEARNING_TEST_TRY(
            DLMath::initialization_pdf<TestNumType>(
                DLMath::InitializationFunction::XAVIER,
                DLMath::ProbabilityDensityFunction::NORMAL, n));
        dist = DLMath::initialization_pdf<TestNumType>(
            DLMath::InitializationFunction::XAVIER,
            DLMath::ProbabilityDensityFunction::NORMAL,n);
        for (std::size_t i = 0; i < PRINT_TIMES; ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(dist(generator)));
        }

        EDGE_LEARNING_TEST_TRY(
            DLMath::initialization_pdf<TestNumType>(
                DLMath::InitializationFunction::KAIMING,
                DLMath::ProbabilityDensityFunction::UNIFORM, n));
        dist = DLMath::initialization_pdf<TestNumType>(
            DLMath::InitializationFunction::KAIMING,
            DLMath::ProbabilityDensityFunction::UNIFORM, n);
        for (std::size_t i = 0; i < PRINT_TIMES; ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(dist(generator)));
        }

        EDGE_LEARNING_TEST_TRY(
            DLMath::initialization_pdf<TestNumType>(
                DLMath::InitializationFunction::XAVIER,
                DLMath::ProbabilityDensityFunction::UNIFORM, n));
        dist = DLMath::initialization_pdf<TestNumType>(
            DLMath::InitializationFunction::XAVIER,
            DLMath::ProbabilityDensityFunction::UNIFORM, n);
        for (std::size_t i = 0; i < PRINT_TIMES; ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(dist(generator)));
        }
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

        int val = 10;
        truth_vec = {16,16,16,16,16};
        DLMath::arr_sum<int>(test_vec1.data(), test_vec1.data(),
                             val, test_vec1.size());
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

        int val = 10;
        truth_vec = {50,80,90,80,50};
        DLMath::arr_mul<int>(test_vec1.data(), test_vec1.data(),
                             val, test_vec1.size());
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(test_vec1[i], truth_vec[i]);
        }
    }

    void test_matarr_mul() {
        std::vector<int> test_mat{1,2,3,4};
        std::vector<int> test_vec{1,2};
        std::vector<int> truth_vec{5,11};

        std::vector<int> res_vec; res_vec.resize(test_vec.size());
        DLMath::matarr_mul_no_check<int>(res_vec.data(), test_mat.data(),
                                         test_vec.data(), 2, 2);
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(res_vec[i], truth_vec[i]);
        }

        EDGE_LEARNING_TEST_FAIL(
            DLMath::matarr_mul<int>(test_vec.data(), test_mat.data(), 
                                    test_vec.data(), 2, 2)
        );
        res_vec.clear();
        DLMath::matarr_mul<int>(res_vec.data(), test_mat.data(), 
                                test_vec.data(), 2, 2);
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(res_vec[i], truth_vec[i]);
        }
    }

    void test_relu() {
        std::vector<TestNumType> test_vec{-2,-1,0,1,2};
        std::vector<TestNumType> truth_vec{0,0,0,1,2};
        DLMath::relu<TestNumType>(test_vec.data(), test_vec.data(), 
            test_vec.size());
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(test_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000000001);
        }
    }

    void test_relu_1() {
        std::vector<TestNumType> test_vec{-2,-1,0,1,2};
        std::vector<TestNumType> truth_vec{0,0,0,1,1};
        DLMath::relu_1<TestNumType>(test_vec.data(), test_vec.data(),
                                    test_vec.size());
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(test_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000000001);
        }
    }

    void test_elu() {
        std::vector<TestNumType> test_vec{-2,-1,0,1,2};
        std::vector<TestNumType> truth_vec{-0.8646647167633873,
                                           -0.6321205588285577, 0.0, 1.0, 2.0};
        DLMath::elu<TestNumType>(test_vec.data(), test_vec.data(),
                                 test_vec.size(), 1.0);
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(test_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000000001);
        }
    }

    void test_elu_1() {
        std::vector<TestNumType> test_vec{-2,-1,0,1,2};
        std::vector<TestNumType> truth_vec{0.1353352832366127,
                                           0.36787944117144233, 1.0, 1.0, 1.0};

        std::vector<TestNumType> test_non_opt_vec{test_vec};
        DLMath::elu_1<TestNumType>(test_non_opt_vec.data(),
                                   test_non_opt_vec.data(),
                                   test_non_opt_vec.size(), 1.0);
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(test_non_opt_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_non_opt_vec[i], truth_vec[i],
                                      0.00000000001);
        }

        std::vector<TestNumType> test_opt_vec(test_vec.size());
        DLMath::elu<TestNumType>(test_opt_vec.data(), test_vec.data(),
                                 test_vec.size(), 1.0);
        DLMath::elu_1_opt<TestNumType>(test_opt_vec.data(), test_opt_vec.data(),
                                       test_opt_vec.size(), 1.0);
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(test_opt_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_opt_vec[i], truth_vec[i],
                                      0.00000000001);
        }
    }

    void test_tanh() {
        std::vector<TestNumType> test_vec{-10.0, 0.0, 1.0, 7.0, 10000.0};
        std::vector<TestNumType> truth_vec{-1.0, 0.0, 0.76159416,
                                           0.99999834, 1.0};
        DLMath::tanh<TestNumType>(test_vec.data(), test_vec.data(),
                                  test_vec.size());
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(test_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000001);
        }
    }

    void test_tanh_1() {
        std::vector<TestNumType> test_vec{-10.0, 0.0, 1.0, 7.0, 10000.0};
        std::vector<TestNumType> truth_vec{8.24461455e-09, 1.00000000e+00,
                                           4.19974342e-01, 3.32610934e-06,
                                           0.00000000e+00};

        std::vector<TestNumType> test_non_opt_vec{test_vec};
        DLMath::tanh_1<TestNumType>(test_non_opt_vec.data(),
                                    test_non_opt_vec.data(),
                                    test_non_opt_vec.size());
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(test_non_opt_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_non_opt_vec[i], truth_vec[i],
                                      0.00000001);
        }

        std::vector<TestNumType> test_opt_vec(test_vec.size());
        DLMath::tanh<TestNumType>(test_opt_vec.data(),
                                  test_vec.data(),
                                  test_vec.size());
        DLMath::tanh_1_opt<TestNumType>(test_opt_vec.data(),
                                        test_opt_vec.data(),
                                        test_opt_vec.size());
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(test_opt_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_opt_vec[i], truth_vec[i],
                                      0.00000001);
        }
    }

    void test_sigmoid() {
        std::vector<TestNumType> test_vec{-10.0, 0.0, 1.0, 7.0, 10000.0};
        std::vector<TestNumType> truth_vec{4.5397868702434395e-05, 0.5,
                                           0.7310585786300049,
                                           0.9990889488055994, 1.0};
        DLMath::sigmoid<TestNumType>(test_vec.data(), test_vec.data(),
                                     test_vec.size());
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(test_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000000001);
        }
    }

    void test_sigmoid_1() {
        std::vector<TestNumType> test_vec{-10.0, 0.0, 1.0, 7.0, 10000.0};
        std::vector<TestNumType> truth_vec{4.5395807735951673e-05, 0.25,
                                           0.19661193324148185,
                                           0.000910221180121784, 0.0};

        std::vector<TestNumType> test_non_opt_vec{test_vec};
        DLMath::sigmoid_1<TestNumType>(test_non_opt_vec.data(),
                                       test_non_opt_vec.data(),
                                       test_non_opt_vec.size());
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(test_non_opt_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_non_opt_vec[i], truth_vec[i],
                                      0.00000000001);
        }

        std::vector<TestNumType> test_opt_vec(test_vec.size());
        DLMath::sigmoid<TestNumType>(test_opt_vec.data(), test_vec.data(),
                                     test_vec.size());
        DLMath::sigmoid_1_opt<TestNumType>(test_opt_vec.data(),
                                           test_opt_vec.data(),
                                           test_opt_vec.size());
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": "
                                     + std::to_string(test_opt_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_opt_vec[i], truth_vec[i],
                                      0.00000000001);
        }
    }

    void test_softmax() 
    {
        std::vector<TestNumType> test_vec{-2,-1,0,1,2};
        std::vector<TestNumType> truth_vec{
            0.01165623095604, 0.031684920796124, 0.086128544436269,
            0.23412165725274, 0.63640864655883};
        DLMath::softmax<TestNumType>(test_vec.data(), test_vec.data(), 
            test_vec.size());
        for (std::size_t i = 0; i < truth_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_PRINT(std::to_string(i) + ": " 
                + std::to_string(test_vec[i]));
            EDGE_LEARNING_TEST_WITHIN(test_vec[i], truth_vec[i], 0.00000000001);
        }
    }

    void test_softmax_1() {
        std::vector<TestNumType> test_vec;
        std::vector<TestNumType> test_gradients{1.0,1.0,1.0,1.0,1.0};
        EDGE_LEARNING_TEST_FAIL(DLMath::softmax_1_opt<TestNumType>(
            test_vec.data(), test_vec.data(), test_gradients.data(),
            test_vec.size()));

        test_vec = {-2.0,-1.0,0.0,1.0,2.0};
        std::vector<TestNumType> softmax; softmax.resize(test_vec.size());
        DLMath::softmax<TestNumType>(test_vec.data(), test_vec.data(),
                                     test_vec.size());
        EDGE_LEARNING_TEST_EXECUTE(DLMath::softmax_1_opt<TestNumType>(
            test_vec.data(), softmax.data(), test_gradients.data(),
            test_vec.size()));
        for (std::size_t i = 0; i < test_vec.size(); ++i)
        {
            std::cout << std::fixed << std::setprecision(40)
                      << "test_vec[i]: " << test_vec[i] << std::endl << std::endl;
        }

        test_vec = {-2.0,-1.0,0.0,1.0,2.0};
        EDGE_LEARNING_TEST_EXECUTE(DLMath::softmax_1<TestNumType>(test_vec.data(), 
            test_vec.data(), test_gradients.data(), test_vec.size()));
        for (std::size_t i = 0; i < test_vec.size(); ++i)
        {
            std::cout << std::fixed << std::setprecision(40) 
                << "test_vec[i]: " << test_vec[i] << std::endl << std::endl;
        }
    }

    void test_cross_entropy() {
        std::vector<TestNumType> test_y    {0.0, 0.0, 0.00, 0.00, 1.0};
        std::vector<TestNumType> test_y_hat{0.1, 0.1, 0.25, 0.05, 0.5};
        TestNumType truth_ce = 0.6931471805599453;
        auto ret = DLMath::cross_entropy(test_y.data(), test_y_hat.data(), 
            test_y_hat.size());
        EDGE_LEARNING_TEST_WITHIN(ret, truth_ce, 0.00000000001);

        TestNumType test_val = 0.5;
        TestNumType truth_val = 0.34657359027997264;
        ret = DLMath::cross_entropy(test_val, test_val);
        EDGE_LEARNING_TEST_WITHIN(ret, truth_val, 0.00000000001);
    }

    void test_cross_entropy_1() {
        std::vector<TestNumType> test_y    {0.0, 0.0, 0.00, 0.00, 1.0};
        std::vector<TestNumType> test_y_hat{0.1, 0.1, 0.25, 0.05, 0.5};
        std::vector<TestNumType> truth_ce1 {0.0, 0.0, 0.00, 0.00, -2.0};
        std::vector<TestNumType> ret_vec; ret_vec.resize(truth_ce1.size());
        DLMath::cross_entropy_1<TestNumType>(ret_vec.data(), test_y.data(),
            test_y_hat.data(), 1.0, test_y_hat.size());
        for (std::size_t i = 0; i < truth_ce1.size(); ++i)
        {
            EDGE_LEARNING_TEST_WITHIN(ret_vec[i], truth_ce1[i], 0.00000000001);
        }
        

        TestNumType test_val = 0.5;
        TestNumType truth_val = -1.0;
        auto ret_val = DLMath::cross_entropy_1<TestNumType>(
            test_val, test_val);
        EDGE_LEARNING_TEST_WITHIN(ret_val, truth_val, 0.00000000001);
    }

    void test_mean_squared_error() {
        TestNumType test_val = 1.0;
        TestNumType truth_val = 0.0;
        auto ret = DLMath::squared_error(test_val, test_val);
        EDGE_LEARNING_TEST_WITHIN(ret, truth_val, 0.00000000001);

        std::vector<TestNumType> test_y    {1.0, 1.0, 1.0, 1.0, 1.0};
        std::vector<TestNumType> test_y_hat{1.1, 0.1, 1.2, 1.5, 0.5};
        TestNumType truth_mse = 0.272;
        ret = DLMath::mean_squared_error(test_y.data(), test_y_hat.data(), 
            test_y_hat.size());
        EDGE_LEARNING_TEST_WITHIN(ret, truth_mse, 0.00000000001);
    }

    void test_mean_squared_error_1() {
        TestNumType test_val1 = 1.0;
        TestNumType test_val2 = 1.5;
        TestNumType truth_val = 0.5;
        auto ret = DLMath::squared_error_1<TestNumType>(test_val1, test_val2, 0.5);
        EDGE_LEARNING_TEST_WITHIN(ret, truth_val, 0.00000000001);

        std::vector<TestNumType> test_y    {1.0, 1.0, 1.0, 1.0, 1.0};
        std::vector<TestNumType> test_y_hat{1.1, 0.1, 1.2, 1.5, 0.5};
        std::vector<TestNumType> truth_mse1 {0.2, -1.8, 0.4, 1.0, -1.0};
        std::vector<TestNumType> ret_vec; ret_vec.resize(truth_mse1.size());
        DLMath::mean_squared_error_1<TestNumType>(ret_vec.data(), test_y.data(),
            test_y_hat.data(), 1.0, test_y_hat.size());
        for (std::size_t i = 0; i < truth_mse1.size(); ++i)
        {
            EDGE_LEARNING_TEST_WITHIN(ret_vec[i], truth_mse1[i], 0.00000000001);
        }
    }

    void test_max_argmax() {
        std::vector<TestNumType> test_vec{0,1,5,4,3};
        TestNumType truth_max = 5;
        TestNumType ret_max = DLMath::max<TestNumType>(test_vec.data(), test_vec.size());
        EDGE_LEARNING_TEST_EQUAL(ret_max, truth_max);

        TestNumType truth_argmax = 2;
        TestNumType ret_argmax = DLMath::argmax<TestNumType>(test_vec.data(), 
            test_vec.size());
        EDGE_LEARNING_TEST_EQUAL(ret_argmax, truth_argmax);

        auto ret_tuple = DLMath::max_and_argmax<TestNumType>(test_vec.data(), 
            test_vec.size());
        EDGE_LEARNING_TEST_EQUAL(std::get<0>(ret_tuple), truth_max);
        EDGE_LEARNING_TEST_EQUAL(std::get<1>(ret_tuple), truth_argmax);
    }

    void test_cross_correlation_without_channels() {
        SizeType input_width = 3;
        SizeType input_height = 3;
        SizeType f = 2;
        SizeType output_width = 2;
        SizeType output_height = 2;
        std::vector<TestNumType> test_img{
            0, 1, 2,
            3, 4, 5,
            6, 7, 8.5
        };
        std::vector<TestNumType> test_k{
            0, 0,
            0, 1
        };
        std::vector<TestNumType> truth_vec{
            4, 5,
            7, 8.5
        };
        std::vector<TestNumType> result(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            DLMath::Shape2d{input_height, input_width},
            test_k.data(), {f, f});
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
        truth_vec = std::vector<TestNumType>{
            0, 1, 2,   0,
            3, 4, 5,   0,
            6, 7, 8.5, 0,
            0, 0, 0,   0
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            DLMath::Shape2d{input_height, input_width},
            test_k.data(), {f, f}, {1, 1}, {1, 1});
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
        truth_vec = std::vector<TestNumType>{
            0, 2,
            6, 8.5
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            DLMath::Shape2d{input_height, input_width},
            test_k.data(), {f, f}, {2, 2}, {1, 1});
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
        test_img = std::vector<TestNumType>{
            0,  1,  2,  4,  5,
            3,  4,  5,  6,  7,
            6,  7,  8,  9,  10,
            9,  10, 11, 12, 13
        };
        test_k = std::vector<TestNumType>{
            0, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        truth_vec = std::vector<TestNumType>{
            12, 14, 16,
            18, 20, 22
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            DLMath::Shape2d{input_height, input_width},
            test_k.data(), {f, f});
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
        truth_vec = std::vector<TestNumType>{
            4,  6,  8,  11, 5,
            10, 12, 14, 16, 7,
            16, 18, 20, 22, 10,
            9,  10, 11, 12, 13
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            DLMath::Shape2d{input_height, input_width},
            test_k.data(), {f, f}, {1, 1}, {1, 1});
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
        truth_vec = std::vector<TestNumType>{
            4,  8,  5,
            16, 20, 10,
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            DLMath::Shape2d{input_height, input_width},
            test_k.data(), {f, f}, {2, 2}, {1, 1});
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

    void test_cross_correlation_with_channels() {
        SizeType input_width = 3;
        SizeType input_height = 3;
        SizeType input_channels = 2;
        SizeType f = 2;
        SizeType output_width = 2;
        SizeType output_height = 2;
        std::vector<TestNumType> test_img{
            0,0, 1,1, 2,2,
            3,3, 4,4, 5,5,
            6,6, 7,7, 8.5,8.5
        };
        std::vector<TestNumType> test_k{
            0,0, 0,0,
            0,0, 1,1
        };
        std::vector<TestNumType> truth_vec{
            4+4, 5+5,
            7+7, 8.5+8.5
        };
        std::vector<TestNumType> result(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, input_channels},
            test_k.data(), {f, f});
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
        truth_vec = std::vector<TestNumType>{
            0+0, 1+1, 2+2,     0+0,
            3+3, 4+4, 5+5,     0+0,
            6+6, 7+7, 8.5+8.5, 0+0,
            0+0, 0+0, 0+0,     0+0
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, input_channels},
            test_k.data(), {f, f}, {1, 1}, {1, 1});
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
        truth_vec = std::vector<TestNumType>{
            0+0, 2+2,
            6+6, 8.5+8.5
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, input_channels},
            test_k.data(), {f, f}, {2, 2}, {1, 1});
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
        input_channels = 3;
        f = 3;
        output_width = 3;
        output_height = 2;
        test_img = std::vector<TestNumType>{
            0,1,2,   4,5,0,   1,2,4,    5,0,1,   2,4,5,
            3,4,5,   6,7,3,   4,5,6,    7,3,4,   5,6,7,
            6,7,8,   9,10,6,  7,8,9,    10,6,7,  8,9,10,
            9,10,11, 12,13,9, 10,11,12, 13,9,10, 11,12,13
        };
        test_k = std::vector<TestNumType>{
            0,0,0, 0,0,0, 0,0,0,
            0,0,0, 1,1,1, 0,0,0,
            0,0,0, 0,0,0, 1,1,1
        };
        truth_vec = std::vector<TestNumType>{
            40, 38, 41,
            58, 56, 59
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, input_channels},
            test_k.data(), {f, f});
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
        truth_vec = std::vector<TestNumType>{
            19, 24, 21, 24, 11,
            37, 40, 38, 41, 18,
            55, 58, 56, 59, 27,
            30, 34, 33, 32, 36
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, input_channels},
            test_k.data(), {f, f}, {1, 1}, {1, 1});
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
        truth_vec = std::vector<TestNumType>{
            19, 21, 11,
            55, 56, 27
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, input_channels},
            test_k.data(), {f, f}, {2, 2}, {1, 1});
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

    void test_cross_correlation_with_channels_with_filters() {
        SizeType input_width = 3;
        SizeType input_height = 3;
        SizeType input_channels = 2;
        SizeType f = 2;
        SizeType n_filters = 2;
        SizeType output_width = 2 * n_filters;
        SizeType output_height = 2;
        std::vector<TestNumType> test_img{
            0,0, 1,1, 2,2,
            3,3, 4,4, 5,5,
            6,6, 7,7, 8.5,8.5
        };
        std::vector<TestNumType> test_k{
        /*  ----col0-----   ----col1-----  */
        /*  -ch0-   -ch1-   -ch0-   -ch1-  */
        /*  f0,f1   f0,f1   f0,f1   f0,f1  */
             0,0,    0,0,    0,0,    0,0,
             0,0,    0,0,    1,1,    1,0,
        };
        std::vector<TestNumType> truth_vec{
            4+4,4, 5+5,5,
            7+7,7, 8.5+8.5,8.5
        };
        std::vector<TestNumType> result(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, input_channels},
            test_k.data(), {f, f}, n_filters);
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

        output_width = 4 * n_filters;
        output_height = 4;
        truth_vec = std::vector<TestNumType>{
            0+0,0, 1+1,1, 2+2,2,       0+0,0,
            3+3,3, 4+4,4, 5+5,5,       0+0,0,
            6+6,6, 7+7,7, 8.5+8.5,8.5, 0+0,0,
            0+0,0, 0+0,0, 0+0,0,       0+0,0
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, input_channels},
            test_k.data(), {f, f}, n_filters, {1, 1}, {1, 1});
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

        output_width = 2 * n_filters;
        output_height = 2;
        truth_vec = std::vector<TestNumType>{
            0+0,0, 2+2,2,
            6+6,6, 8.5+8.5,8.5
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, input_channels},
            test_k.data(), {f, f}, n_filters, {2, 2}, {1, 1});
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
        input_channels = 3;
        f = 3;
        output_width = 3 * n_filters;
        output_height = 2;
        test_img = std::vector<TestNumType>{
            0,1,2,   4,5,0,   1,2,4,    5,0,1,   2,4,5,
            3,4,5,   6,7,3,   4,5,6,    7,3,4,   5,6,7,
            6,7,8,   9,10,6,  7,8,9,    10,6,7,  8,9,10,
            9,10,11, 12,13,9, 10,11,12, 13,9,10, 11,12,13
        };
        test_k = std::vector<TestNumType>{
        /*  ------col0-------  ------col1-------  ------col2-------  */
        /*  -ch0- -ch1- -ch2-  -ch0- -ch1- -ch2-  -ch0- -ch1- -ch2-  */
        /*  f0,f1 f0,f1 f0,f1  f0,f1 f0,f1 f0,f1  f0,f1 f0,f1 f0,f1  */
             0,0,  0,0,  0,0,   0,0,  0,0,  0,0,   0,0,  0,0,  0,0,
             0,0,  0,0,  0,0,   1,1,  1,0,  1,0,   0,0,  0,0,  0,0,
             0,0,  0,0,  0,0,   0,0,  0,0,  0,0,   1,1,  1,0,  1,0,
        };
        truth_vec = std::vector<TestNumType>{
            40,13, 38,14, 41,15,
            58,19, 56,20, 59,21
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, input_channels},
            test_k.data(), {f, f}, n_filters);
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

        output_width = 5 * n_filters;
        output_height = 4;
        truth_vec = std::vector<TestNumType>{
            19,6,  24,8,  21,8,  24,10, 11,2,
            37,12, 40,13, 38,14, 41,15, 18,5,
            55,18, 58,19, 56,20, 59,21, 27,8,
            30,9,  34,12, 33,10, 32,13, 36,11
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, input_channels},
            test_k.data(), {f, f}, n_filters, {1, 1}, {1, 1});
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

        output_width = 3 * n_filters;
        output_height = 2;
        truth_vec = std::vector<TestNumType>{
            19,6,  21,8,  11,2,
            55,18, 56,20, 27,8,
        };
        result.resize(truth_vec.size());
        DLMath::cross_correlation<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, input_channels},
            test_k.data(), {f, f}, n_filters, {2, 2}, {1, 1});
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
        std::vector<TestNumType> test_img{
            10, 1, 2,
            3,  4, 5,
            6,  7, 8.5
        };
        std::vector<TestNumType> truth_vec{
            10, 5,
            7,  8.5
        };
        std::vector<TestNumType> result(truth_vec.size());
        DLMath::max_pool<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width}, {f, f});
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
        test_img = std::vector<TestNumType>{
            10,  1,  2,  4,  5,
            3,   4,  5,  6,  7,
            6,   7,  8,  9,  10,
            9,   10, 11, 12, 13
        };
        truth_vec = std::vector<TestNumType>{
            10, 9,  10,
            11, 12, 13
        };
        result.resize(truth_vec.size());
        DLMath::max_pool<TestNumType>(
            result.data(), test_img.data(), {input_height, input_width}, {f, f});
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
        truth_vec = std::vector<TestNumType>{
            10, 10
        };
        result.resize(truth_vec.size());
        DLMath::max_pool<TestNumType>(
            result.data(), test_img.data(), {input_height, input_width},
            {f, f}, {2, 2});
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

        SizeType channels = 3;
        output_width = 3;
        output_height = 2;
        auto step = output_width * channels;
        test_img = std::vector<TestNumType>{
            10,1,2,  4,5,10,  1,2,4,    5,10,1,  2,4,5,
            3,4,5,   6,7,3,   4,5,6,    7,3,4,   5,6,7,
            6,7,8,   9,10,6,  7,8,9,    10,6,7,  8,9,10,
            9,10,11, 12,13,9, 10,11,12, 13,9,10, 11,12,13
        };
        truth_vec = std::vector<TestNumType>{
            10,10,10, 10,10,10, 10,10,10,
            12,13,12, 13,13,12, 13,12,13
        };
        result.resize(truth_vec.size());
        DLMath::max_pool<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, channels}, {f, f});
        for (std::size_t r = 0; r < output_height; ++r) {
            for (std::size_t c = 0; c < output_width; ++c) {
                for (std::size_t ch = 0; ch < channels; ++ch) {
                    EDGE_LEARNING_TEST_PRINT(
                        "[" + std::to_string(r) + "," + std::to_string(c)
                        + "," + std::to_string(ch) + "] "
                        + std::to_string(result[r * step + c * channels + ch]));
                    EDGE_LEARNING_TEST_WITHIN(
                        result[r * step + c * channels + ch],
                        truth_vec[r * step + c * channels + ch],
                        0.0000000000001);
                }
            }
        }

        output_width = 2;
        output_height = 1;
        test_img = std::vector<TestNumType>{
            10,1,2,  4,5,10,  1,2,4,    5,10,1,  2,4,5,
            3,4,5,   6,7,3,   4,5,6,    7,3,4,   5,6,7,
            6,7,8,   9,10,6,  7,8,9,    10,6,7,  8,9,10,
            9,10,11, 12,13,9, 10,11,12, 13,9,10, 11,12,13
        };
        truth_vec = std::vector<TestNumType>{
            10,10,10, 10,10,10,
        };
        result.resize(truth_vec.size());
        DLMath::max_pool<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, channels}, {f, f}, {2, 2});
        for (std::size_t r = 0; r < output_height; ++r) {
            for (std::size_t c = 0; c < output_width; ++c) {
                for (std::size_t ch = 0; ch < channels; ++ch) {
                    EDGE_LEARNING_TEST_PRINT(
                        "[" + std::to_string(r) + "," + std::to_string(c)
                        + "," + std::to_string(ch) + "] "
                        + std::to_string(result[r * step + c * channels + ch]));
                    EDGE_LEARNING_TEST_WITHIN(
                        result[r * step + c * channels + ch],
                        truth_vec[r * step + c * channels + ch],
                        0.0000000000001);
                }
            }
        }
    }

    void test_avg_pool() {
        SizeType input_width = 3;
        SizeType input_height = 3;
        SizeType f = 2;
        SizeType output_width = 2;
        SizeType output_height = 2;
        std::vector<TestNumType> test_img{
            10, 1, 2,
            3,  4, 5,
            6,  7, 8.5
        };
        std::vector<TestNumType> truth_vec{
            4.5, 3,
            5,   6.125
        };
        std::vector<TestNumType> result(truth_vec.size());
        DLMath::avg_pool<TestNumType>(
            result.data(), test_img.data(), {input_height, input_width},
            {f, f});
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
        test_img = std::vector<TestNumType>{
            10,  1,  2,  4,  5,
            3,   4,  5,  6,  7,
            6,   7,  8,  9,  10,
            9,   10, 11, 12, 13
        };
        truth_vec = std::vector<TestNumType>{
            46.0/9, 46.0/9, 56.0/9,
            63.0/9, 72.0/9, 81.0/9
        };
        result.resize(truth_vec.size());
        DLMath::avg_pool<TestNumType>(
            result.data(), test_img.data(), {input_height, input_width},
            {f, f});
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
        truth_vec = std::vector<TestNumType>{
            46.0/9, 56.0/9
        };
        result.resize(truth_vec.size());
        DLMath::avg_pool<TestNumType>(
            result.data(), test_img.data(), {input_height, input_width}, 
            {f, f}, {2, 2});
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

        SizeType channels = 3;
        output_width = 3;
        output_height = 2;
        auto step = output_width * channels;
        test_img = std::vector<TestNumType>{
            10,1,2,  4,5,10,  1,2,4,    5,10,1,  2,4,5,
            3,4,5,   6,7,3,   4,5,6,    7,3,4,   5,6,7,
            6,7,8,   9,10,6,  7,8,9,    10,6,7,  8,9,10,
            9,10,11, 12,13,9, 10,11,12, 13,9,10, 11,12,13
        };
        truth_vec = std::vector<TestNumType>{
            50.0/9,49.0/9,53.0/9, 53.0/9,56.0/9,50.0/9, 49.0/9,53.0/9,53.0/9,
            66.0/9,75.0/9,69.0/9, 78.0/9,72.0/9,66.0/9, 75.0/9,69.0/9,78.0/9
        };
        result.resize(truth_vec.size());
        DLMath::avg_pool<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, channels}, {f, f});
        for (std::size_t r = 0; r < output_height; ++r) {
            for (std::size_t c = 0; c < output_width; ++c) {
                for (std::size_t ch = 0; ch < channels; ++ch) {
                    EDGE_LEARNING_TEST_PRINT(
                        "[" + std::to_string(r) + "," + std::to_string(c)
                        + "," + std::to_string(ch) + "] "
                        + std::to_string(result[r * step + c * channels + ch]));
                    EDGE_LEARNING_TEST_WITHIN(
                        result[r * step + c * channels + ch],
                        truth_vec[r * step + c * channels + ch],
                        0.0000000000001);
                }
            }
        }

        output_width = 2;
        output_height = 1;
        test_img = std::vector<TestNumType>{
            10,1,2,  4,5,10,  1,2,4,    5,10,1,  2,4,5,
            3,4,5,   6,7,3,   4,5,6,    7,3,4,   5,6,7,
            6,7,8,   9,10,6,  7,8,9,    10,6,7,  8,9,10,
            9,10,11, 12,13,9, 10,11,12, 13,9,10, 11,12,13
        };
        truth_vec = std::vector<TestNumType>{
            50.0/9,49.0/9,53.0/9, 49.0/9,53.0/9,53.0/9
        };
        result.resize(truth_vec.size());
        DLMath::avg_pool<TestNumType>(
            result.data(), test_img.data(),
            {input_height, input_width, channels}, {f, f}, {2, 2});
        for (std::size_t r = 0; r < output_height; ++r) {
            for (std::size_t c = 0; c < output_width; ++c) {
                for (std::size_t ch = 0; ch < channels; ++ch) {
                    EDGE_LEARNING_TEST_PRINT(
                        "[" + std::to_string(r) + "," + std::to_string(c)
                        + "," + std::to_string(ch) + "] "
                        + std::to_string(result[r * step + c * channels + ch]));
                    EDGE_LEARNING_TEST_WITHIN(
                        result[r * step + c * channels + ch],
                        truth_vec[r * step + c * channels + ch],
                        0.0000000000001);
                }
            }
        }
    }

    void test_append() {
        SizeType cols = 3;
        SizeType rows = 5;
        std::vector<TestNumType> result = {
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                0, 0, 0,
                0, 0, 0,
        };
        std::vector<TestNumType> test_vec = {
                10, 11, 12,
                13, 14, 15,
        };
        std::vector<TestNumType> truth_vec = {
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                10, 11, 12,
                13, 14, 15,
        };
        DLMath::append_check(result.data(), {rows, cols},
                             test_vec.data(), 2, 0, 3);
        for (SizeType r = 0; r < rows; ++r) {
            for (SizeType c = 0; c < cols; ++c) {
                EDGE_LEARNING_TEST_PRINT(result[r * cols + c]);
                EDGE_LEARNING_TEST_EQUAL(result[r * cols + c],
                                         truth_vec[r * cols + c]);
            }
        }

        cols = 5;
        rows = 3;
        result = {
                1,  2,  3,  0, 0,
                6,  7,  8,  0, 0,
                11, 12, 13, 0, 0,
        };
        test_vec = {
                4,  5,
                9,  10,
                14, 15,
        };
        result.resize(result.size() + test_vec.size());
        truth_vec = {
                1,  2,  3,  4,  5,
                6,  7,  8,  9,  10,
                11, 12, 13, 14, 15,
        };
        DLMath::append_check(result.data(), {rows, cols},
                             test_vec.data(), 2, 1, 3);
        for (SizeType r = 0; r < rows; ++r)
        {
            for (SizeType c = 0; c < cols; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(result[r * cols + c]);
                EDGE_LEARNING_TEST_EQUAL(result[r * cols + c], truth_vec[r * cols + c]);
            }
        }

        cols = 3;
        rows = 3;
        SizeType channels = 5;
        result = {
                1,1,1,0,0, 2,2,2,0,0, 3,3,3,0,0,
                1,2,3,0,0, 1,2,3,0,0, 1,2,3,0,0,
                0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0,
        };
        test_vec = {
                1,1, 2,2, 3,3,
                4,5, 4,5, 4,5,
                1,1, 1,1, 1,1,
        };
        result.resize(result.size() + test_vec.size());
        truth_vec = {
                1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3,
                1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5,
                0,0,0,1,1, 0,0,0,1,1, 0,0,0,1,1,
        };
        DLMath::append_check(result.data(), {rows, cols, channels},
                             test_vec.data(), 2, 2, 3);
        for (SizeType r = 0; r < rows; ++r)
        {
            for (SizeType c = 0; c < cols; ++c)
            {
                for (SizeType ch = 0; ch < channels; ++ch)
                {
                    EDGE_LEARNING_TEST_PRINT(
                            result[r * (cols + channels) + c * channels + ch]);
                    EDGE_LEARNING_TEST_EQUAL(
                            result[r * (cols + channels) + c * channels + ch],
                            truth_vec[r * (cols + channels) + c * channels + ch]);
                }
            }
        }

        EDGE_LEARNING_TEST_FAIL(
                DLMath::append_check(result.data(), {rows, cols, channels},
                                     test_vec.data(), 2, 3, 3));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::append_check(result.data(), {rows, cols, channels},
                                     test_vec.data(), 2, 3, 3),
                std::runtime_error);
    }

    void test_extract() {
        SizeType cols = 3;
        SizeType rows = 2;
        std::vector<TestNumType> result;
        result.resize(cols * rows);
        std::vector<TestNumType> test_vec = {
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                10, 11, 12,
                13, 14, 15,
        };
        std::vector<TestNumType> truth_vec = {
                10, 11, 12,
                13, 14, 15,
        };
        DLMath::extract_check(result.data(), {rows, cols},
                              test_vec.data(), 5, 0, 3);
        for (SizeType r = 0; r < rows; ++r) {
            for (SizeType c = 0; c < cols; ++c) {
                EDGE_LEARNING_TEST_PRINT(result[r * cols + c]);
                EDGE_LEARNING_TEST_EQUAL(result[r * cols + c],
                                         truth_vec[r * cols + c]);
            }
        }

        cols = 2;
        rows = 3;
        result.clear();
        result.resize(cols * rows);
        test_vec = {
                1,  2,  3,  4,  5,
                6,  7,  8,  9,  10,
                11, 12, 13, 14, 15,
        };
        truth_vec = {
                4,  5,
                9,  10,
                14, 15,
        };
        DLMath::extract_check(result.data(), {rows, cols},
                              test_vec.data(), 5, 1, 3);
        for (SizeType r = 0; r < rows; ++r)
        {
            for (SizeType c = 0; c < cols; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(result[r * cols + c]);
                EDGE_LEARNING_TEST_EQUAL(result[r * cols + c], truth_vec[r * cols + c]);
            }
        }

        cols = 3;
        rows = 3;
        SizeType channels = 2;
        result.clear();
        result.resize(cols * rows * channels);
        test_vec = {
                1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3,
                1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5,
                0,0,0,1,1, 0,0,0,1,1, 0,0,0,1,1,
        };
        truth_vec = {
                1,1, 2,2, 3,3,
                4,5, 4,5, 4,5,
                1,1, 1,1, 1,1,
        };
        DLMath::extract_check(result.data(), {rows, cols, channels},
                              test_vec.data(), 5, 2, 3);
        for (SizeType r = 0; r < rows; ++r)
        {
            for (SizeType c = 0; c < cols; ++c)
            {
                for (SizeType ch = 0; ch < channels; ++ch)
                {
                    EDGE_LEARNING_TEST_PRINT(
                            result[r * (cols + channels) + c * channels + ch]);
                    EDGE_LEARNING_TEST_EQUAL(
                            result[r * (cols + channels) + c * channels + ch],
                            truth_vec[r * (cols + channels) + c * channels + ch]);
                }
            }
        }

        EDGE_LEARNING_TEST_FAIL(
                DLMath::extract_check(result.data(), {rows, cols, channels},
                                      test_vec.data(), 5, 3, 3));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::extract_check(result.data(), {rows, cols, channels},
                                      test_vec.data(), 5, 3, 3),
                std::runtime_error);
    }

    void test_concatenate()
    {
        SizeType cols = 3;
        SizeType rows = 5;
        std::vector<TestNumType> test_vec1 = {
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
        };
        std::vector<TestNumType> test_vec2 = {
                10, 11, 12,
                13, 14, 15,
        };
        std::vector<TestNumType> truth_vec = {
                1,  2,  3,
                4,  5,  6,
                7,  8,  9,
                10, 11, 12,
                13, 14, 15,
        };
        std::vector<TestNumType> result(truth_vec.size());
        DLMath::concatenate(result.data(),
                            test_vec1.data(), {3,3},
                            test_vec2.data(), {2,3}, 0);
        EDGE_LEARNING_TEST_FAIL(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3},
                                    test_vec2.data(), {2,3}, 1));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3},
                                    test_vec2.data(), {2,3}, 1),
                std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3},
                                    test_vec2.data(), {2,3}, 2));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3},
                                    test_vec2.data(), {2,3}, 2),
                std::runtime_error);
        for (SizeType r = 0; r < rows; ++r)
        {
            for (SizeType c = 0; c < cols; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(result[r * cols + c]);
                EDGE_LEARNING_TEST_EQUAL(result[r * cols + c], truth_vec[r * cols + c]);
            }
        }

        cols = 5;
        rows = 3;
        test_vec1 = {
                1,  2,  3,
                6,  7,  8,
                11, 12, 13,
        };
        test_vec2 = {
                4,  5,
                9,  10,
                14, 15,
        };
        truth_vec = {
                1,  2,  3,  4,  5,
                6,  7,  8,  9,  10,
                11, 12, 13, 14, 15,
        };
        result.clear();
        result.resize(truth_vec.size());
        DLMath::concatenate(result.data(),
                            test_vec1.data(), {3,3},
                            test_vec2.data(), {3,2}, 1);
        EDGE_LEARNING_TEST_FAIL(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3},
                                    test_vec2.data(), {3,2}, 0));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3},
                                    test_vec2.data(), {3,2}, 0),
                std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3},
                                    test_vec2.data(), {3,2}, 2));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3},
                                    test_vec2.data(), {3,2}, 2),
                std::runtime_error);
        for (SizeType r = 0; r < rows; ++r)
        {
            for (SizeType c = 0; c < cols; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(result[r * cols + c]);
                EDGE_LEARNING_TEST_EQUAL(result[r * cols + c], truth_vec[r * cols + c]);
            }
        }

        cols = 3;
        rows = 3;
        SizeType channels = 5;
        test_vec1 = {
                1,1,1, 2,2,2, 3,3,3,
                1,2,3, 1,2,3, 1,2,3,
                0,0,0, 0,0,0, 0,0,0,
        };
        test_vec2 = {
                1,1, 2,2, 3,3,
                4,5, 4,5, 4,5,
                1,1, 1,1, 1,1,
        };
        truth_vec = {
                1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3,
                1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5,
                0,0,0,1,1, 0,0,0,1,1, 0,0,0,1,1,
        };
        result.clear();
        result.resize(truth_vec.size());
        DLMath::concatenate(result.data(),
                            test_vec1.data(), {3,3,3},
                            test_vec2.data(), {3,3,2}, 2);
        EDGE_LEARNING_TEST_FAIL(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3,3},
                                    test_vec2.data(), {3,3,2}, 0));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3,3},
                                    test_vec2.data(), {3,3,2}, 0),
                std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3,3},
                                    test_vec2.data(), {3,3,2}, 1));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3,3},
                                    test_vec2.data(), {3,3,2}, 1),
                std::runtime_error);
        for (SizeType r = 0; r < rows; ++r)
        {
            for (SizeType c = 0; c < cols; ++c)
            {
                for (SizeType ch = 0; ch < channels; ++ch)
                {
                    EDGE_LEARNING_TEST_PRINT(
                        result[r * (cols + channels) + c * channels + ch]);
                    EDGE_LEARNING_TEST_EQUAL(
                        result[r * (cols + channels) + c * channels + ch],
                        truth_vec[r * (cols + channels) + c * channels + ch]);
                }
            }
        }

        EDGE_LEARNING_TEST_FAIL(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3,3},
                                    test_vec2.data(), {3,3,2}, 3));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::concatenate(result.data(),
                                    test_vec1.data(), {3,3,3},
                                    test_vec2.data(), {3,3,2}, 3),
                std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(
            DLMath::concatenate(result.data(),
                                test_vec1.data(), {3,3,3},
                                test_vec2.data(), {3,0,2}, 2));
        EDGE_LEARNING_TEST_THROWS(
            DLMath::concatenate(result.data(),
                                test_vec1.data(), {3,3,3},
                                test_vec2.data(), {3,0,2}, 2),
            std::runtime_error);

        std::vector<TestNumType> test_vec = {
                1, 2, 3,
                1, 1, 1,
                0, 0, 0,

                1,1, 2,2, 3,3,
                2,3, 2,3, 2,3,
                0,0, 0,0, 0,0,

                1,1, 2,2, 3,3,
                4,5, 4,5, 4,5,
                1,1, 1,1, 1,1,
        };
        truth_vec = {
                1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3,
                1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5,
                0,0,0,1,1, 0,0,0,1,1, 0,0,0,1,1,
        };
        result.clear();
        result.resize(truth_vec.size());
        std::vector<DLMath::Shape3d> test_shape = {{3,3,1}, {3,3,2}, {3,3,2}};
        DLMath::concatenate(result.data(),
                            test_vec.data(), test_shape, 2);
        for (SizeType r = 0; r < rows; ++r)
        {
            for (SizeType c = 0; c < cols; ++c)
            {
                for (SizeType ch = 0; ch < channels; ++ch)
                {
                    EDGE_LEARNING_TEST_PRINT(
                            result[r * (cols + channels) + c * channels + ch]);
                    EDGE_LEARNING_TEST_EQUAL(
                            result[r * (cols + channels) + c * channels + ch],
                            truth_vec[r * (cols + channels) + c * channels + ch]);
                }
            }
        }

        EDGE_LEARNING_TEST_FAIL(
            DLMath::concatenate(result.data(),
                                test_vec.data(), test_shape, 3));
        EDGE_LEARNING_TEST_THROWS(
            DLMath::concatenate(result.data(),
                                test_vec.data(), test_shape, 3),
            std::runtime_error);

        EDGE_LEARNING_TEST_FAIL(
            DLMath::concatenate(result.data(),
                                test_vec.data(), {{3,3,1}, {3,3,2}, {3,0,2}}, 2));
        EDGE_LEARNING_TEST_THROWS(
            DLMath::concatenate(result.data(),
                                test_vec.data(), {{3,3,1}, {3,3,2}, {3,0,2}}, 2),
            std::runtime_error);
    }

    void test_separate()
    {
        SizeType cols1 = 3;
        SizeType rows1 = 3;
        SizeType cols2 = 2;
        SizeType rows2 = 3;
        std::vector<TestNumType> test_vec = {
                1,  2,  3,
                4,  5,  6,
                7,  8,  9,
                10, 11, 12,
                13, 14, 15,
        };
        std::vector<TestNumType> truth_vec1 = {
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
        };
        std::vector<TestNumType> truth_vec2 = {
                10, 11, 12,
                13, 14, 15,
        };
        std::vector<TestNumType> result1(truth_vec1.size());
        std::vector<TestNumType> result2(truth_vec2.size());
        DLMath::separate(result1.data(), {3,3},
                         result2.data(), {2,3},
                         test_vec.data(), 0);
        EDGE_LEARNING_TEST_FAIL(
                DLMath::separate(result1.data(), {3,3},
                                 result2.data(), {2,3},
                                 test_vec.data(), 2));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::separate(result1.data(), {3,3},
                                 result2.data(), {2,3},
                                 test_vec.data(), 1),
                std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(
                DLMath::separate(result1.data(), {3,3},
                                 result2.data(), {2,3},
                                 test_vec.data(), 2));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::separate(result1.data(), {3,3},
                                 result2.data(), {2,3},
                                 test_vec.data(), 2),
                std::runtime_error);
        for (SizeType r = 0; r < rows1; ++r)
        {
            for (SizeType c = 0; c < cols1; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(result1[r * cols1 + c]);
                EDGE_LEARNING_TEST_EQUAL(result1[r * cols1 + c],
                                         truth_vec1[r * cols1 + c]);
            }
        }
        for (SizeType r = 0; r < rows2; ++r)
        {
            for (SizeType c = 0; c < cols2; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(result1[r * cols2 + c]);
                EDGE_LEARNING_TEST_EQUAL(result1[r * cols2 + c],
                                         truth_vec1[r * cols2 + c]);
            }
        }

        cols1 = 3;
        rows1 = 3;
        cols2 = 3;
        rows2 = 2;
        test_vec = {
                1,  2,  3,  4,  5,
                6,  7,  8,  9,  10,
                11, 12, 13, 14, 15,
        };
        truth_vec1 = {
                1,  2,  3,
                6,  7,  8,
                11, 12, 13,
        };
        truth_vec2 = {
                4,  5,
                9,  10,
                14, 15,
        };
        result1.clear();
        result1.resize(truth_vec1.size());
        result2.clear();
        result2.resize(truth_vec2.size());
        DLMath::separate(result1.data(), {3,3},
                         result2.data(), {3,2},
                         test_vec.data(), 1);
        EDGE_LEARNING_TEST_FAIL(
                DLMath::separate(result1.data(), {3,3},
                                 result2.data(), {3,2},
                                 test_vec.data(), 0));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::separate(result1.data(), {3,3},
                                 result2.data(), {3,2},
                                 test_vec.data(), 0),
                std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(
                DLMath::separate(result1.data(), {3,3},
                                 result2.data(), {3,2},
                                 test_vec.data(), 2));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::separate(result1.data(), {3,3},
                                 result2.data(), {3,2},
                                 test_vec.data(), 2),
                std::runtime_error);
        for (SizeType r = 0; r < rows1; ++r)
        {
            for (SizeType c = 0; c < cols1; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(result1[r * cols1 + c]);
                EDGE_LEARNING_TEST_EQUAL(result1[r * cols1 + c],
                                         truth_vec1[r * cols1 + c]);
            }
        }
        for (SizeType r = 0; r < rows2; ++r)
        {
            for (SizeType c = 0; c < cols2; ++c)
            {
                EDGE_LEARNING_TEST_PRINT(result1[r * cols2 + c]);
                EDGE_LEARNING_TEST_EQUAL(result1[r * cols2 + c],
                                         truth_vec1[r * cols2 + c]);
            }
        }

        cols1 = 3;
        rows1 = 3;
        SizeType channels1 = 3;
        cols2 = 3;
        rows2 = 3;
        SizeType channels2 = 2;
        test_vec = {
                1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3,
                1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5,
                0,0,0,1,1, 0,0,0,1,1, 0,0,0,1,1,
        };
        truth_vec1 = {
                1,1,1, 2,2,2, 3,3,3,
                1,2,3, 1,2,3, 1,2,3,
                0,0,0, 0,0,0, 0,0,0,
        };
        truth_vec2 = {
                1,1, 2,2, 3,3,
                4,5, 4,5, 4,5,
                1,1, 1,1, 1,1,
        };
        result1.clear();
        result1.resize(truth_vec1.size());
        result2.clear();
        result2.resize(truth_vec2.size());
        DLMath::separate(result1.data(), {3,3,3},
                         result2.data(), {3,3,2},
                         test_vec.data(), 2);
        EDGE_LEARNING_TEST_FAIL(
                DLMath::separate(result1.data(), {3,3,3},
                                 result2.data(), {3,3,2},
                                 test_vec.data(), 0));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::separate(result1.data(), {3,3,3},
                                 result2.data(), {3,3,2},
                                 test_vec.data(), 0),
                std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(
                DLMath::separate(result1.data(), {3,3,3},
                                 result2.data(), {3,3,2},
                                 test_vec.data(), 1));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::separate(result1.data(), {3,3,3},
                                 result2.data(), {3,3,2},
                                 test_vec.data(), 1),
                std::runtime_error);
        for (SizeType r = 0; r < rows1; ++r)
        {
            for (SizeType c = 0; c < cols1; ++c)
            {
                for (SizeType ch = 0; ch < channels1; ++ch)
                {
                    EDGE_LEARNING_TEST_PRINT(
                            result1[r * (cols1 + channels1) + c * channels1 + ch]);
                    EDGE_LEARNING_TEST_EQUAL(
                            result1[r * (cols1 + channels1) + c * channels1 + ch],
                            truth_vec1[r * (cols1 + channels1) + c * channels1 + ch]);
                }
            }
        }
        for (SizeType r = 0; r < rows2; ++r)
        {
            for (SizeType c = 0; c < cols2; ++c)
            {
                for (SizeType ch = 0; ch < channels2; ++ch)
                {
                    EDGE_LEARNING_TEST_PRINT(
                            result2[r * (cols2 + channels2) + c * channels2 + ch]);
                    EDGE_LEARNING_TEST_EQUAL(
                            result2[r * (cols2 + channels2) + c * channels2 + ch],
                            truth_vec2[r * (cols2 + channels2) + c * channels2 + ch]);
                }
            }
        }

        EDGE_LEARNING_TEST_FAIL(
                DLMath::separate(result1.data(), {3,3,3},
                                 result1.data(), {3,3,2},
                                 test_vec.data(), 3));
        EDGE_LEARNING_TEST_THROWS(
                DLMath::separate(result1.data(), {3,3,3},
                                 result2.data(), {3,3,2},
                                 test_vec.data(), 3),
                std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(
            DLMath::separate(result1.data(), {3,3,3},
                             result1.data(), {3,0,2},
                             test_vec.data(), 2));
        EDGE_LEARNING_TEST_THROWS(
            DLMath::separate(result1.data(), {3,3,3},
                             result2.data(), {3,0,2},
                             test_vec.data(), 2),
            std::runtime_error);

        test_vec = {
                1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3,
                1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5,
                0,0,0,1,1, 0,0,0,1,1, 0,0,0,1,1,
        };
        std::vector<TestNumType> truth_vec = {
                1, 2, 3,
                1, 1, 1,
                0, 0, 0,

                1,1, 2,2, 3,3,
                2,3, 2,3, 2,3,
                0,0, 0,0, 0,0,

                1,1, 2,2, 3,3,
                4,5, 4,5, 4,5,
                1,1, 1,1, 1,1,
        };
        std::vector<TestNumType> result(truth_vec.size());
        std::vector<DLMath::Shape3d> result_shape = {{3,3,1}, {3,3,2}, {3,3,2}};
        DLMath::separate(result.data(), result_shape,
                         test_vec.data(), 2);
        SizeType offset = 0;
        for (const auto& shape: result_shape)
        {
            for (SizeType r = 0; r < shape.height(); ++r)
            {
                for (SizeType c = 0; c < shape.width(); ++c)
                {
                    for (SizeType ch = 0; ch < shape.channels(); ++ch)
                    {
                        EDGE_LEARNING_TEST_PRINT(
                                result[offset + r * (shape.width() + shape.channels()) + c * shape.channels() + ch]);
                        EDGE_LEARNING_TEST_EQUAL(
                                result[offset + r * (shape.width() + shape.channels()) + c * shape.channels() + ch],
                                truth_vec[offset + r * (shape.width() + shape.channels()) + c * shape.channels() + ch]);
                    }
                }
            }
            offset += shape.size();
        }

        EDGE_LEARNING_TEST_FAIL(
            DLMath::separate(result.data(), result_shape,
                             test_vec.data(), 3));
        EDGE_LEARNING_TEST_THROWS(
            DLMath::separate(result.data(), result_shape,
                             test_vec.data(), 3),
            std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(
            DLMath::separate(result.data(), {{3,3,1}, {3,3,2}, {3,0,2}},
                             test_vec.data(), 2));
        EDGE_LEARNING_TEST_THROWS(
            DLMath::separate(result.data(), {{3,3,1}, {3,3,2}, {3,0,2}},
                             test_vec.data(), 2),
            std::runtime_error);
    }

};

int main() {
    TestDLMath().test();
    return EDGE_LEARNING_TEST_FAILURES;
}
