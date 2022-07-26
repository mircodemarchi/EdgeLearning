/***************************************************************************
 *            dnn/dlmath.hpp
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

/*! \file  dnn/dlmath.hpp
 *  \brief Deep Learning Math functionalities.
 */

#include <cmath>
#include <functional>
#include <cassert>
#include <stdexcept>
#include <tuple>
#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>
#include <numeric>
#include <string>

#include <iostream>

#ifndef EDGE_LEARNING_DNN_DLMATH_HPP
#define EDGE_LEARNING_DNN_DLMATH_HPP

namespace EdgeLearning {

class DLMath 
{
public:
    static constexpr RneType::result_type max_rand = 
        std::numeric_limits<RneType::result_type>::max();

    struct Coord2d {
        SizeType row;
        SizeType col;
    };

    struct Coord3d {
        Coord3d(Coord2d c2d)
            : row{c2d.row}
            , col{c2d.col}
            , channel{0}
        {}

        SizeType row;
        SizeType col;
        SizeType channel;
    };

    struct Shape {
    public:
        Shape(std::vector<SizeType> values)
            : _shape{values}
        {}

        [[nodiscard]] SizeType size() const
        {
            return std::accumulate(_shape.begin(), _shape.end(),
                                   SizeType(1),
                                   std::multiplies<SizeType>());
        }

        operator std::vector<SizeType>() const { return _shape; }

        SizeType& operator[](SizeType idx) { return _shape[idx]; }
        [[nodiscard]] const SizeType& at(SizeType idx) const
        { return _shape[idx]; }

    protected:
        std::vector<SizeType> _shape;
    };

    struct Shape2d : public Shape {
    public:
        static inline SizeType SIZE = 2;
        static inline SizeType HEIGHT_IDX = 0;
        static inline SizeType WIDTH_IDX = 1;

        Shape2d(SizeType h, SizeType w)
            : Shape{{h,w}}
        {}

        Shape2d(SizeType s)
            : Shape{{s,s}}
        {}

        [[nodiscard]] const SizeType& height() const
        { return _shape[HEIGHT_IDX]; }
        [[nodiscard]]       SizeType& height()
        { return _shape[HEIGHT_IDX]; }

        [[nodiscard]] const SizeType& width() const
        { return _shape[WIDTH_IDX]; }
        [[nodiscard]]       SizeType& width()
        { return _shape[WIDTH_IDX]; }
    };

    struct Shape3d : public Shape {
    public:
        static inline SizeType SIZE = 3;
        static inline SizeType HEIGHT_IDX = 0;
        static inline SizeType WIDTH_IDX = 1;
        static inline SizeType CHANNEL_IDX = 2;

        Shape3d(Shape2d s2d)
            : Shape({s2d.height(),s2d.width(),1})
        {}

        Shape3d(SizeType h, SizeType w=1, SizeType c=1)
            : Shape({h,w,c})
        {}

        [[nodiscard]] const SizeType& height() const
        { return _shape[HEIGHT_IDX]; }
        [[nodiscard]]       SizeType& height()
        { return _shape[HEIGHT_IDX]; }

        [[nodiscard]] const SizeType& width() const
        { return _shape[WIDTH_IDX]; }
        [[nodiscard]]       SizeType& width()
        { return _shape[WIDTH_IDX]; }

        [[nodiscard]] const SizeType& channels() const
        { return _shape[CHANNEL_IDX]; }
        [[nodiscard]]       SizeType& channels()
        { return _shape[CHANNEL_IDX]; }
    };

    /**
     * \brief Enumeration of the PDFs (Probability Density Functions).
     */
    enum class ProbabilityDensityFunction
    {
        NORMAL, ///< \brief Normal distribution.
        UNIFORM ///< \brief Uniform distribution.
    };

    /**
     * \brief Enumeration of the initialization functions.
     */
    enum class InitializationFunction
    {
        XAVIER, ///< \brief sqrt( 2 / n_in )
        KAIMING ///< \brief sqrt( 1 / n_in )
    };

    /**
     * \brief Calculate the index of the element in the vector.
     * \tparam T The type of the elements.
     * \param vec   const std::vector<T>& The vector of elements.
     * \param n     const T& The element to find.
     * \return std::int64_t Index of the element or -1 if not found.
     */
    template <typename T>
    static std::int64_t index_of(const std::vector<T>& vec, const T& e)
    {
        auto itr = std::find(vec.begin(), vec.end(), e);
        return itr != vec.cend() ? std::distance(vec.begin(), itr) : -1;
    }


    /**
     * \brief Gaussian Probability Density Function.
     * \tparam T      Output type.
     * \param mean    Mean of the probability distribution required.
     * \param std_dev Standard Deviation of the probability distribution
     * required.
     * \return std::function<T(RneType)> The distribution function.
     */
    template <typename T>
    static std::function<T(RneType&)> normal_pdf(NumType mean, NumType std_dev)
    {
        return std::normal_distribution<NumType>{mean, std_dev};
    }

    /**
     * \brief Uniform Probability Density Function.
     * \tparam T      Return function output type.
     * \param center  Center of the probability distribution required.
     * \param delta   Range in which the density function will expand.
     * \return std::function<T(RneType)> The distribution function.
     */
    template <typename T>
    static std::function<T(RneType&)> uniform_pdf(
        NumType center, NumType delta)
    {
        delta /= 2.0;
        std::function<T(RneType&)> ret =
            [delta, center](RneType& x)
            {
                T rand = ((static_cast<T>(x()) / static_cast<T>(max_rand))
                          * T{2.0}) - T{1.0};
                rand = (rand * delta) + center;
                return rand;
            };
        return ret;
    }

    /**
     * \brief Uniform Probability Density Function.
     * \tparam T      Return function output type.
     * \param center  Center of the probability distribution required.
     * \param delta   Range in which the density function will expand.
     * \return std::function<T(RneType)> The distribution function.
     */
    template <typename T>
    static std::function<T(RneType&)> pdf(
        NumType center, NumType delta, ProbabilityDensityFunction type)
    {
        switch (type) {
            case ProbabilityDensityFunction::UNIFORM:
                return uniform_pdf<T>(center, delta);
            case ProbabilityDensityFunction::NORMAL:
                return normal_pdf<T>(center, delta);
            default:
                throw std::runtime_error(
                    "Probability density function not recognized");
        }
    }

    /**
     * \brief Kaiming He, et. al. initialization
     * https://arxiv.org/pdf/1502.01852.pdf
     * Normal distribution with variance := sqrt( 2 / n_in )
     * \tparam T    Output type.
     * \param n     The input size.
     * \return T The variance defined by Kaiming for neural network
     * initialization.
     */
    template <typename T>
    static T kaiming_initialization_variance(SizeType n)
    {
        return std::sqrt(T{2} / static_cast<T>(n));
    }

    /**
     * \brief Kaiming He, et. al. initialization
     * https://arxiv.org/pdf/1502.01852.pdf
     * Normal distribution with mean := 0.0
     * \tparam T Output type.
     * \return The mean defined by Kaiming for neural network
     * initialization.
     */
    template <typename T>
    static T kaiming_initialization_mean()
    {
        return T{0};
    }

    /**
     * \brief Kaiming He, et. al. initialization
     * https://arxiv.org/pdf/1502.01852.pdf
     * Normal distribution with mean := 0.0 and variance := sqrt( 2 / n_in )
     * \tparam T    Tuple output type.
     * \param n     The input size.
     * \return A tuple of Kaiming initialization mean and variance.
     */
    template <typename T>
    static std::tuple<T, T> kaiming_initialization(SizeType n)
    {
        return {
            kaiming_initialization_mean<T>(),
            kaiming_initialization_variance<T>(n)
        };
    }

    /**
     * \brief Xavier initialization
     * https://arxiv.org/pdf/1706.02515.pdf
     * Normal distribution with variance := sqrt( 1 / n_in )
     * \tparam T    Output type.
     * \param n     The input size.
     * \return T The variance defined by Kaiming for neural network
     * initialization.
     */
    template <typename T>
    static T xavier_initialization_variance(SizeType n)
    {
        return std::sqrt(T{1} / static_cast<T>(n));
    }

    /**
     * \brief Xavier initialization
     * https://arxiv.org/pdf/1706.02515.pdf
     * Normal distribution with mean := 0.0
     * \tparam T Output type.
     * \return The mean defined by Xavier for neural network
     * initialization.
     */
    template <typename T>
    static T xavier_initialization_mean()
    {
        return T{0};
    }

    /**
     * \brief Xavier initialization
     * https://arxiv.org/pdf/1706.02515.pdf
     * Normal distribution with mean := 0.0 and variance := sqrt( 1 / n_in )
     * \tparam T    Tuple output type.
     * \param n     The input size.
     * \return A tuple of Xavier initialization mean and variance.
     */
    template <typename T>
    static std::tuple<T, T> xavier_initialization(SizeType n)
    {
        return {
            xavier_initialization_mean<T>(),
            xavier_initialization_variance<T>(n)
        };
    }

    /**
     * \brief Initialization mean and variance parameters selector.
     * \tparam T    Tuple output type.
     * \param type  Type of initialization requested.
     * \param n     The input size.
     * \return A tuple of initialization mean and variance.
     */
    template <typename T>
    static std::tuple<T, T> initialization(
        InitializationFunction type, SizeType n)
    {
        switch (type) {
            case InitializationFunction::XAVIER:
                return xavier_initialization<T>(n);
            case InitializationFunction::KAIMING:
                return kaiming_initialization<T>(n);
            default:
                throw std::runtime_error(
                    "Initialization function not recognized");
        }
    }

    /**
     * \brief Initialization probability density function.
     * \tparam T            Return function output type.
     * \param init_type     Type of initialization requested.
     * \param pdf_type      Type of Probability density function.
     * \param n             The input size.
     * \return std::function<T(RneType)> The distribution function.
     */
    template <typename T>
    static std::function<T(RneType&)> initialization_pdf(
        InitializationFunction init_type,
        ProbabilityDensityFunction pdf_type,
        SizeType n)
    {
        auto pdf_params = initialization<T>(init_type, n);
        return pdf<T>(std::get<0>(pdf_params), std::get<1>(pdf_params),
                      pdf_type);
    }


    /**
     * \brief Element wise multiplication between two arrays.
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src1   First operand array.
     * \param src2   Second operand array.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* arr_mul(T* dst, const T* src1, const T* src2, SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = src1[i] * src2[i];
        }
        return dst;
    }

    /**
     * \brief Element wise multiplication between two arrays.
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    First operand array.
     * \param val    Value to multiply.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* arr_mul(T* dst, const T* src, T val, SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = src[i] * val;
        }
        return dst;
    }

    static SizeType unique()
    {
        static SizeType id = 0;
        return id++;
    }

    /**
     * \brief Element wise summation between two arrays.
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src1   First operand array.
     * \param src2   Second operand array.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* arr_sum(T* dst, const T* src1, const T* src2, SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = src1[i] + src2[i];
        }
        return dst;
    }

    /**
     * \brief Element wise summation between two arrays.
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    First operand array.
     * \param val    Value to sum.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* arr_sum(T* dst, const T* src, T val, SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = src[i] + val;
        }
        return dst;
    }

    /**
     * \brief Multiplication between a matrix and an array.
     * Used for y = Wx
     * \tparam T      Type of each source and destination elements.
     * \param arr_dst Array destination to write the result.
     * \param mat_src Matrix source, left operand.
     * \param arr_src Array source, right operand.
     * \param rows    Amount of rows.
     * \param cols    Amount of columns.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* matarr_mul_no_check(
        T* arr_dst, const T* mat_src, const T* arr_src,
        SizeType rows, SizeType cols)
    {
        for (SizeType i = 0; i < rows; ++i)
        {
            arr_dst[i] = T{0};
            for (SizeType j = 0; j < cols; ++j)
            {
                arr_dst[i] += mat_src[(i * cols) + j] * arr_src[j];
            }
        }
        return arr_dst;
    }

    template <typename T>
    static T* matarr_mul(T* arr_dst, const T* mat_src, const T* arr_src, 
        SizeType rows, SizeType cols)
    {
        if (arr_src == arr_dst) 
        {
            throw std::runtime_error("arr_src, arr_dst have to be different "
                                     "in order to perform matarr_mul");
        }
        return matarr_mul_no_check<T>(arr_dst, mat_src, arr_src, rows, cols);
    }

    /**
     * \brief ReLU Function.
     * relu(x) = max(0, x)
     * \tparam T Type of the input and return type.
     * \param x  Input value.
     * \return T The ReLU result.
     */
    template <typename T>
    static T relu(T x)
    {
        return std::max(x, T{0});
    }

    /**
     * \brief ReLU Function applied to a vector.
     * relu(z)_i = max(0, z_i)
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    Array of read elements.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* relu(T* dst, const T* src, SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = relu(src[i]);
        }
        return dst;
    }

    /**
     * \brief Derivative of ReLU Function.
     * relu'[z] = 1 if z > 0 else 0
     * \tparam T Type of the input and return type.
     * \param x  Input value.
     * \return T The ReLU derivative result.
     */
    template <typename T>
    static T relu_1(T x)
    {
        return x > T{0} ? T{1} : T{0};
    }

    /**
     * \brief Derivative of ReLU Function.
     * relu'[z]_i = 1 if z_i > 0 else 0
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    Array of read elements.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* relu_1(T* dst, const T* src, SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = relu_1(src[i]);
        }
        return dst;
    }

    /**
     * \brief ELU Function.
     * elu(x) = x if x > 0 else alpha * (e^x - 1)
     * \tparam T    Type of the input and return type.
     * \param x     Input value.
     * \param alpha Saturation value (in general 1.0).
     * \return T The ELU value of x.
     */
    template <typename T>
    static T elu(T x, T alpha)
    {
        return x > 0 ? x : alpha * (std::exp(x) - 1);
    }

    /**
     * \brief ELU Function applied to a vector.
     * elu(x)_i = x_i if x_i > 0 else alpha * (e^x_i - 1)
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    Array of read elements.
     * \param length Length of the arrays.
     * \param alpha  Saturation value (in general 1.0).
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* elu(T* dst, const T* src, SizeType length, T alpha)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = elu(src[i], alpha);
        }
        return dst;
    }

    /**
     * \brief Derivative of ELU Function optimized version that consider to
     * have the ELU result function in the input value.
     * \tparam T    Type of the input and return type.
     * \param x     Input value.
     * \param alpha Saturation value (in general 1.0).
     * \return T The ELU derivative value of x.
     */
    template <typename T>
    static T elu_1_opt(T x, T alpha)
    {
        return x > T{0} ? T{1} : x + alpha;
    }

    /**
     * \brief Derivative of ELU Function optimized version that consider to
     * have the ELU result function in the src array.
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    Array of read elements.
     * \param length Length of the arrays.
     * \param alpha  Saturation value (in general 1.0).
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* elu_1_opt(T* dst, const T* src, SizeType length, T alpha)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = elu_1_opt(src[i], alpha);
        }
        return dst;
    }

    /**
     * \brief Derivative of ELU Function.
     * elu(x) = x if x > 0 else alpha * (e^x - 1)
     * \tparam T    Type of the input and return type.
     * \param x     Input value.
     * \param alpha Saturation value (in general 1.0).
     * \return T The ELU derivative value of x.
     */
    template <typename T>
    static T elu_1(T x, T alpha)
    {
        return x > T{0} ? T{1} : alpha * std::exp(x);
    }

    /**
     * \brief Derivative of ELU Function.
     * elu'[z]_i = 1 if z_i > 0 else 0
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    Array of read elements.
     * \param length Length of the arrays.
     * \param alpha  Saturation value (in general 1.0).
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* elu_1(T* dst, const T* src, SizeType length, T alpha)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = elu_1(src[i], alpha);
        }
        return dst;
    }

    /**
     * \brief Hyperbolic Tangent Function.
     * \tparam T Type of the input and return type.
     * \param x  Input value.
     * \return T The tanh value of x.
     */
    template <typename T>
    static T tanh(T x)
    {
        return std::tanh(x);
    }

    /**
     * \brief Hyperbolic Tangent Function applied to a vector.
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    Array of read elements.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* tanh(T* dst, const T* src, SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = tanh(src[i]);
        }
        return dst;
    }

    /**
     * \brief Derivative of Hyperbolic Tangent Function optimized version that
     * consider to have the TanH result function in the input value.
     * \tparam T    Type of the input and return type.
     * \param x     Input value.
     * \return T The TanH derivative value of x.
     */
    template <typename T>
    static T tanh_1_opt(T x)
    {
        return T{1} - x * x;
    }

    /**
     * \brief Derivative of Hyperbolic Tangent Function optimized version that
     * consider to have the TanH result function in the input value.
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    Array of read elements.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* tanh_1_opt(T* dst, const T* src, SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = tanh_1_opt(src[i]);
        }
        return dst;
    }

    /**
     * \brief Hyperbolic Tangent Function first derivative.
     * \tparam T Type of the input and return type.
     * \param x  Input value.
     * \return T The tanh derivative value of x.
     */
    template <typename T>
    static T tanh_1(T x)
    {
        T t = std::tanh(x);
        return tanh_1_opt(t);
    }

    /**
     * \brief Hyperbolic Tangent Function first derivative applied to a vector.
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    Array of read elements.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* tanh_1(T* dst, const T* src, SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = tanh_1(src[i]);
        }
        return dst;
    }

    /**
     * \brief Sigmoid Function.
     * \tparam T Type of the input and return type.
     * \param x  Input value.
     * \return T The sigmoid value of x.
     */
    template <typename T>
    static T sigmoid(T x)
    {
        return 1 / (1 + std::exp(-x));
    }

    /**
     * \brief Sigmoid Function applied to a vector.
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    Array of read elements.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* sigmoid(T* dst, const T* src, SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = sigmoid(src[i]);
        }
        return dst;
    }

    /**
     * \brief Derivative of Sigmoid Function optimized version that
     * consider to have the sigmoid result function in the input value.
     * \tparam T    Type of the input and return type.
     * \param x     Input value.
     * \return T The TanH derivative value of x.
     */
    template <typename T>
    static T sigmoid_1_opt(T x)
    {
        return x * (1 - x);
    }

    /**
     * \brief Derivative of Sigmoid Function optimized version that
     * consider to have the sigmoid result function in the input value.
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    Array of read elements.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* sigmoid_1_opt(T* dst, const T* src, SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = sigmoid_1_opt(src[i]);
        }
        return dst;
    }

    /**
     * \brief Sigmoid Function first derivative.
     * \tparam T Type of the input and return type.
     * \param x  Input value.
     * \return T The sigmoid derivative value of x.
     */
    template <typename T>
    static T sigmoid_1(T x)
    {
        T s = sigmoid(x);
        return sigmoid_1_opt(s);
    }

    /**
     * \brief Sigmoid Function first derivative applied to a vector.
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    Array of read elements.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* sigmoid_1(T* dst, const T* src, SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = sigmoid_1(src[i]);
        }
        return dst;
    }

    /**
     * \brief Softmax Function.
     * softmax(z)_i = exp(z_i) / \sum_j(exp(z_j))
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    Array of read elements.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* softmax(T* dst, const T* src, SizeType length)
    {
        // Compute the exponential of each value and compute the sum. 
        T sum_exp_z{0};
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = std::exp(src[i]);
            sum_exp_z += dst[i];
        }

        // Compute the inverse of the sum.
        T inv_sum_exp_z = T{1} / sum_exp_z;

        // Multiply the inverse of the sum for each value.
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] *= inv_sum_exp_z;
        }
        return dst;
    }

    /**
     * \brief Derivative Optimized of Softmax Function with the value of the 
     * argmax already saved in the src array. Source and Destination has to be 
     * different. If Source and Destination pointers are equal, a runtime_error
     * will be thrown.
     * softmax'(z)_i = \sum_j(
     *  softmax(z_i)(1 - softmax(z_i)) if i == j else -softmax(z_i)softmax(z_j))
     * \tparam T        Type of each source and destination elements.
     * \param dst       Array to write the result. It has to be different by src.
     * \param src       Array of input elements that has to already contains the
     *                  Softmax results of the requested input.
     *                  It has to be different by dst.
     * \param gradients Softmax derivation has to be calculated with the
     *                  reference of the backward gradients in input, in order
     *                  to obtain the right result.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* softmax_1_opt(T* dst, const T* src, const T* gradients,
                            SizeType length)
    {
        if (src == dst) 
        {
            throw std::runtime_error("src, dst have to be different "
                                     "in order to perform softmax_1_opt");
        }
        return softmax_1_opt_no_check(dst, src, gradients, length);
    }

    /**
     * \brief Derivative Optimized of Softmax Function with the value of the
     * argmax already saved in the src array. Source and Destination has to be
     * different but if Source and Destination pointers are equal, no exception
     * will be thrown.
     * softmax'(z)_i = \sum_j(
     *  softmax(z_i)(1 - softmax(z_i)) if i == j else -softmax(z_i)softmax(z_j))
     * \tparam T        Type of each source and destination elements.
     * \param dst       Array to write the result. It has to be different by src.
     * \param src       Array of input elements that has to already contains the
     *                  Softmax results of the requested input.
     *                  It has to be different by dst.
     * \param gradients Softmax derivation has to be calculated with the
     *                  reference of the backward gradients in input, in order
     *                  to obtain the right result.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* softmax_1_opt_no_check(T* dst, const T* src, const T* gradients,
                                     SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = T{0.0};
            for(SizeType j = 0; j < length; ++j)
            {
                dst[i] += (i == j)
                          ? src[i] * (T{1.0} - src[i]) * gradients[j]
                          : -src[i] * src[j] * gradients[j];
            }
        }
        return dst;
    }

    /**
     * \brief Derivative of Softmax Function.
     * softmax'(z)_i = \sum_j(
     *  softmax(z_i) * (1 - softmax(z_i)) * gradients[j] if i == j
     *      else -softmax(z_i) * softmax(z_j) * gradients[j])
     * \tparam T        Type of each source and destination elements.
     * \param dst       Array to write the result.
     * \param src       Array of input elements that will be used for Softmax
     *                  calculus.
     * \param gradients Softmax derivation has to be calculated with the
     *                  reference of the backward gradients in input, in order
     *                  to obtain the right result.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* softmax_1(T* dst, const T* src, const T* gradients,
                        SizeType length)
    {
        T* tmp = new T[length];
        assert(tmp);
        softmax(tmp, src, length);
        softmax_1_opt_no_check(dst, tmp, gradients, length);
        delete[] tmp;
        return dst;
    }

    /**
     * \brief Cross-Entropy Function.
     * cross_entropy(y, y_hat) = - y * log(max(y_hat, epsilon))
     * \tparam T Type of the inputs and return type.
     * \param y     Target value.
     * \param y_hat Estimated value.
     * \return T The resulting Cross-Entropy.
     */
    template <typename T>
    static T cross_entropy(T y, T y_hat)
    {
        return - y * std::log(std::max(y_hat, 
            std::numeric_limits<T>::epsilon()));
    }

    /**
     * \brief Cross-Entropy Function.
     * cross_entropy(y, y_hat) = - \sum_j(y_j * log( max(y_hat_j, epsilon) ))
     * \tparam T Type of the inputs and return type.
     * \param y      Target array values.
     * \param y_hat  Estimated array values.
     * \param length Length of the arrays.
     * \return T The resulting Cross-Entropy.
     */
    template <typename T>
    static T cross_entropy(const T* y, const T* y_hat, SizeType length)
    {
        T ret{0.0};
        for (SizeType i = 0; i < length; ++i)
        {
            ret += cross_entropy(y[i], y_hat[i]);
        }
        return ret;
    }

    /**
     * \brief Cross-Entropy Function first derivative.
     * cross_entropy'(y, y_hat) = - y / max(y_hat, epsilon)
     * \tparam T Type of the inputs and return type.
     * \param y     Target value.
     * \param y_hat Estimated value.
     * \param norm  Normalizer term to multiply.
     * \return T The resulting Cross-Entropy first derivative.
     */
    template <typename T>
    static T cross_entropy_1(T y, T y_hat, T norm = T(1))
    {
        return norm * (-y / (std::max(y_hat, std::numeric_limits<T>::min())));
    }

    /**
     * \brief Cross-Entropy Function first derivative.
     * cross_entropy'(y, y_hat)_i = - y_i / max(y_hat_i, epsilon)
     * \tparam T Type of the inputs and return type.
     * \param dst    Destination array.
     * \param y      Target array values.
     * \param y_hat  Estimated array values.
     * \param norm   Normalizer term.
     * \param length Length of the arrays. 
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* cross_entropy_1(T* dst, const T* y, const T* y_hat, T norm,
        SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = cross_entropy_1(y[i], y_hat[i], norm);
        }
        return dst;
    }

    /**
     * \brief Squared Error Function.
     * squared_error(y, y_hat) = (y - y_hat)^2
     * \tparam T Type of the inputs and return type.
     * \param y      Target array values.
     * \param y_hat  Estimated array values.
     * \param length Length of the arrays. 
     * \return T The squared error value.
     */
    template <typename T>
    static T squared_error(T y, T y_hat)
    {
        return (y - y_hat) * (y - y_hat);
    }

    /**
     * \brief Mean Squared Error Function.
     * mean_squared_error(y, y_hat) = (1 / n) * \sum_i( (y_i - y_hat_i)^2 )
     * \tparam T Type of the inputs and return type.
     * \param y      Target array values.
     * \param y_hat  Estimated array values.
     * \param length Length of the arrays. 
     * \return T The mean squared error value.
     */
    template <typename T>
    static T mean_squared_error(const T* y, const T* y_hat, SizeType length)
    {
        T ret{0.0};
        for (SizeType i = 0; i < length; ++i)
        {
            ret += squared_error(y[i], y_hat[i]);
        }
        return ret / length;
    }

    /**
     * \brief Squared Error Function first derivative.
     * squared_error(y, y_hat) = -2 * (y - y_hat)
     * \tparam T Type of the inputs and return type.
     * \param y      Target array values.
     * \param y_hat  Estimated array values.
     * \param norm   Normalizer term.
     * \return T The first derivative squared error value.
     */
    template <typename T>
    static T squared_error_1(T y, T y_hat, T norm)
    {
        return -T{2.0} * norm * (y - y_hat);
    }

    /**
     * \brief Mean Squared Error Function first derivative. 
     * mean_squared_error(y, y_hat)_i = -2 * ( y_i - y_hat_i )
     * \tparam T Type of the inputs and return type.
     * \param dst    Destination array.
     * \param y      Target array values.
     * \param y_hat  Estimated array values.
     * \param norm   Normalizer term.
     * \param length Length of the arrays. 
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* mean_squared_error_1(T* dst, const T* y, const T* y_hat, T norm, 
        SizeType length)
    {
        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = squared_error_1(y[i], y_hat[i], norm);
        }
        return dst;
    }

    /**
     * \brief Find the max value of a vector.
     * \tparam T Type of the input and return type.
     * \param src    Source array.
     * \param length Length of the array.
     * \return T The max value.
     */
    template <typename T>
    static T max(const T* src, SizeType length) 
    {
        return *std::max_element(src, src + length);
    }

    /**
     * \brief Find the argument that point to the maximum value.
     * \tparam T Type of the input.
     * \param src    Source array.
     * \param length Length of the array.
     * \return SizeType The argmax index.
     */
    template <typename T>
    static SizeType argmax(const T* src, SizeType length) 
    {
        return static_cast<SizeType>(std::distance(src, 
            std::max_element(src, src + length)));
    }

    /**
     * \brief Find the max and the argmax values.
     * \tparam T Type of the input.
     * \param src    Source array.
     * \param length Length of the array.
     * \return std::tuple<T, SizeType> Tuple of max and argmax.
     */
    template <typename T>
    static std::tuple<T, SizeType> max_and_argmax(T* src, SizeType length) 
    {
        auto max_iter = std::max_element(src, src + length);
        auto dist = static_cast<SizeType>(std::distance(src, max_iter));
        return {*max_iter, dist};
    }

    /**
     * \brief Cross Correlation 2D of a source 2D matrix and a squared kernel.
     * \tparam T        Type of each source and destination elements.
     * \param dst       The destination matrix in which put the resulting
     *                  matrix.
     * \param src       The source matrix on which calculate the convolution.
     * \param src_shape The shape of the source matrix: height, width.
     * \param k         The kernel matrix to use for convolution.
     * \param k_shape   The shape of the kernel: height, width.
     * \param s         The stride amount: the number of cells that the kernel
     *                  will move. It is defined in 2d: the width is the amount
     *                  stride when moving from left to right, the height
     *                  from up to down.
     * \param p         The padding of the source matrix to include defined in
     *                  2d: the width is the amount padding introduced in right
     *                  and left side, the height in up and down side.
     * \return The pointer to the destination matrix.
     *
     * The destination matrix will be of shape:
     *  width_dst  = ((width_src  - width_k  + (2 * p)) / s) + 1
     *  height_dst = ((height_src - height_k + (2 * p)) / s) + 1
     */
    template <typename T>
    static T* cross_correlation(
        T* dst, const T* src, Shape2d src_shape, const T* k, Shape2d k_shape,
        Shape2d s = {1, 1}, Shape2d p = {0, 0})
    {
        return cross_correlation<T>(
            dst, src, Shape3d(src_shape), k, k_shape, s, p);
    }

    /**
     * \brief Convolution 2D of a 3D source matrix and a cubic kernel.
     * \tparam T        Type of each source and destination elements.
     * \param dst       The destination matrix in which put the resulting
     *                  matrix.
     * \param src       The source matrix on which calculate the convolution.
     * \param src_shape The shape of the source matrix: height, width, channels.
     * \param k         The kernel matrix to use for convolution.
     * \param k_shape   The shape of the kernel: height, width.
     * The third dimension is the same of the src matrix.
     * \param s         The stride amount: the number of cells that the kernel
     *                  will move. It is defined in 2d: the width is the amount
     *                  stride when moving from left to right, the height
     *                  from up to down.
     * \param p         The padding of the source matrix to include defined in
     *                  2d: the width is the amount padding introduced in right
     *                  and left side, the height in up and down side.
     * \return The pointer to the destination matrix.
     *
     * The destination matrix will be of shape:
     *  width_dst  = ((width_src  - width_k  + (2 * p)) / s) + 1
     *  height_dst = ((height_src - height_k + (2 * p)) / s) + 1
     */
    template <typename T>
    static T* cross_correlation(
        T* dst, const T* src, Shape3d src_shape, const T* k, Shape2d k_shape,
        Shape2d s = {1, 1}, Shape2d p = {0, 0})
    {
        return cross_correlation<T>(dst, src, src_shape, k, k_shape, 1, s, p);
    }

    /**
     * \brief Multi Cross Correlation 2D of a 3D source matrix iterated on
     * n_filters of cubic kernel.
     * \tparam T        Type of each source and destination elements.
     * \param dst       The destination matrix in which put the resulting
     *                  matrix.
     * \param src       The source matrix on which calculate the convolution.
     * \param src_shape The shape of the source matrix: height, width, channels.
     * \param k         The kernel matrix to use for convolution.
     * \param k_shape   The shape of the kernel: height, width.
     * The third dimension is the same of the src matrix.
     * \param n_filters The number of filters contained in k to apply to the
     *                  matrix.
     * \param s         The stride amount: the number of cells that the kernel
     *                  will move. It is defined in 2d: the width is the amount
     *                  stride when moving from left to right, the height
     *                  from up to down.
     * \param p         The padding of the source matrix to include defined in
     *                  2d: the width is the amount padding introduced in right
     *                  and left side, the height in up and down side.
     * \return The pointer to the destination matrix.
     *
     * The destination matrix will be of shape:
     *  width_dst  = ((width_src  - width_k  + (2 * p)) / s) + 1
     *  height_dst = ((height_src - height_k + (2 * p)) / s) + 1
     */
    template <typename T>
    static T* cross_correlation(
        T* dst, const T* src, Shape3d src_shape, const T* k, Shape2d k_shape,
        SizeType n_filters, Shape2d s = {1, 1}, Shape2d p = {0, 0})
    {
        return kernel_slide<T>(
            _conv4d_op<T>, dst, src, src_shape, k, k_shape, n_filters, s, p);
    }

    /**
     * \brief Max pooling of a source matrix.
     * \tparam T        Type of each source and destination elements.
     * \param dst       The destination matrix in which put the resulting
     *                  matrix.
     * \param src       The source matrix on which calculate the max pooling.
     * \param src_shape The shape of the source matrix: height, width, channels.
     * \param k_shape   The shape of the kernel: height, width.
     * The third dimension is the same of the src matrix.
     * \param s         The stride amount: the number of cells that the kernel
     *                  will move. It is defined in 2d: the width is the amount
     *                  stride when moving from left to right, the height
     *                  from up to down.
     * \return The pointer to the destination matrix.
     *
     * The destination matrix will be of shape:
     *  width_dst  = ((width_src  - width_k)  / s) + 1
     *  height_dst = ((height_src - height_k) / s) + 1
     */
    template <typename T>
    static T* max_pool(T* dst, const T* src, Shape3d src_shape,
                       Shape2d k_shape, Shape2d s = {1, 1})
    {
        return kernel_slide<T>(
            _max_pool_op<T>, dst, src, src_shape, nullptr, k_shape, 1, s);
    }

    /**
     * \brief Average pooling of a source matrix.
     * \tparam T        Type of each source and destination elements.
     * \param dst       The destination matrix in which put the resulting
     *                  matrix.
     * \param src       The source matrix on which calculate the average
     *                  pooling.
     * \param src_shape The shape of the source matrix: height, width, channels.
     * \param k_shape   The shape of the kernel: height, width.
     * The third dimension is the same of the src matrix.
     * \param s         The stride amount: the number of cells that the kernel
     *                  will move. It is defined in 2d: the width is the amount
     *                  stride when moving from left to right, the height
     *                  from up to down.
     * \return The pointer to the destination matrix.
     *
     * The destination matrix will be of shape:
     *  width_dst  = ((width_src  - width_k)  / s) + 1
     *  height_dst = ((height_src - height_k) / s) + 1
     */
    template <typename T>
    static T* avg_pool(T* dst, const T* src, Shape3d src_shape,
                       Shape2d k_shape, Shape2d s = {1, 1})
    {
        return kernel_slide<T>(
            _avg_pool_op<T>, dst, src, src_shape, nullptr, k_shape, 1, s);
    }

    /**
     * \brief Kernel slicing on the source matrix.
     * \tparam T        Type of each source and destination elements.
     * \param k_to_src_operation The operation to perform at each overlapping
     * step between the source matrix and the kernel.
     * \param dst       The destination matrix in which put the resulting
     *                  matrix.
     * \param src       The source matrix on which calculate the operation
     *                  defined in k_to_src_operation.
     * \param src_shape The shape of the source matrix: height, width, channels.
     * \param k         The kernel matrix to use for convolution. Can be nullptr
     *                  if the kernel is not used in k_to_src_operation (see
     *                  average pooling and max pooling).
     * \param k_shape   The shape of the kernel: height, width.
     *                  The third dimension is the same of the src matrix.
     * \param n_filters The number of filters contained in k to apply to the
     *                  matrix.
     * \param s         The stride amount: the number of cells that the kernel
     *                  will move. It is defined in 2d: the width is the amount
     *                  stride when moving from left to right, the height
     *                  from up to down.
     * \param p         The padding of the source matrix to include defined in
     *                  2d: the width is the amount padding introduced in right
     *                  and left side, the height in up and down side.
     * \return The pointer to the destination matrix.
     *
     * The destination matrix will be of shape:
     *  width_dst  = ((width_src  - f + (2 * p)) / s) + 1
     *  height_dst = ((height_src - f + (2 * p)) / s) + 1
     */
    template <typename T>
    static T* kernel_slide(
        std::function<void(T*, Shape2d, Coord2d,
                           const T*, Shape3d,
                           const T*, Shape2d, SizeType,
                           int64_t, int64_t)> k_to_src_operation,
        T* dst, const T* src, Shape3d src_shape,
        const T* k, Shape2d k_shape, SizeType n_filters = 1,
        Shape2d s = {1, 1}, Shape2d p = {0, 0})
    {
        s.width() = std::max(s.width(), SizeType(1));
        s.height() = std::max(s.height(), SizeType(1));
        auto width_dst = src_shape.width() == 0 ? 0 :
            ((src_shape.width() - k_shape.width() + 2 * p.width()) / s.width()) + 1;
        auto height_dst = src_shape.height() == 0 ? 0 :
            ((src_shape.height() - k_shape.height() + 2 * p.height()) / s.height()) + 1;
        for (SizeType row_dst = 0; row_dst < height_dst; ++row_dst)
        {
            for (SizeType col_dst = 0; col_dst < width_dst; ++col_dst)
            {
                auto col = (static_cast<int64_t>(col_dst * s.width())
                    - static_cast<int64_t>(p.width()))
                    * static_cast<int64_t>(src_shape.channels());
                auto row = static_cast<int64_t>(row_dst * s.height())
                    - static_cast<int64_t>(p.height());
                k_to_src_operation(
                    dst, {height_dst, width_dst}, {row_dst, col_dst},
                    src, src_shape, k, k_shape, n_filters, row, col);
            }
        }
        return dst;
    }

    /**
     * \brief Append a n-dimensional matrix to a destination address on an axis.
     * \tparam T Type of source and destination matrix elements.
     * \param dst               The destination n-dimensional matrix.
     * \param dst_shape         The shape of the destination matrix.
     * \param src               The source n-dimensional matrix.
     * \param src_shape_axis    The shape of the axis in which perform the
     *                          append of the source matrix.
     * \param axis              The axis in which perform the append.
     * \param dst_axis_offset   The offset of the axis in which perform the
     *                          append of the destination matrix.
     * \return T* The destination address.
     */
    template <typename T>
    static T* append(T* dst, std::vector<SizeType> dst_shape,
                     const T* src, SizeType src_shape_axis,
                     SizeType axis, SizeType dst_axis_offset)
    {
        // Calculate number of iterations and src offset.
        SizeType iteration_amount = 1;
        SizeType tmp_offset = 1;
        for (SizeType i = 0; i < dst_shape.size(); ++i)
        {
            if (i < axis)
            {
                iteration_amount *= dst_shape[i];
            }
            else if (i > axis)
            {
                tmp_offset *= dst_shape[i];
            }
        }

        SizeType dst_offset = tmp_offset * dst_shape[axis];
        SizeType src_offset = tmp_offset * src_shape_axis;
        SizeType offset     = tmp_offset * dst_axis_offset;
        for (SizeType i = 0; i < iteration_amount; ++i)
        {
            for (SizeType src_idx = 0; src_idx < src_offset; ++src_idx)
            {
                dst[i * dst_offset + offset + src_idx] = src[
                        i * src_offset + src_idx];
            }
        }

        return dst;
    }

    /**
     * \brief Extract a n-dimensional sub-matrix from a source address on an
     * axis.
     * \tparam T Type of source and destination matrix elements.
     * \param dst               The destination n-dimensional matrix.
     * \param dst_shape         The shape of the destination matrix.
     * \param src               The source n-dimensional matrix.
     * \param src_shape_axis    The shape of the axis in which perform the
     *                          extract of the source matrix.
     * \param axis              The axis in which perform the extract.
     * \param src_axis_offset   The offset of the axis in which perform the
     *                          extract of the source matrix.
     * \return T* The destination address.
     */
    template <typename T>
    static T* extract(T* dst, const std::vector<SizeType>& dst_shape,
                      const T* src, SizeType src_shape_axis,
                      SizeType axis, SizeType src_axis_offset)
    {
        // Calculate number of iterations and src offset.
        SizeType iteration_amount = 1;
        SizeType tmp_offset = 1;
        for (SizeType i = 0; i < dst_shape.size(); ++i)
        {
            if (i < axis)
            {
                iteration_amount *= dst_shape[i];
            }
            else if (i > axis)
            {
                tmp_offset *= dst_shape[i];
            }
        }

        SizeType dst_offset = tmp_offset * dst_shape[axis];
        SizeType src_offset = tmp_offset * src_shape_axis;
        SizeType offset     = tmp_offset * src_axis_offset;
        for (SizeType i = 0; i < iteration_amount; ++i)
        {
            for (SizeType dst_idx = 0; dst_idx < dst_offset; ++dst_idx)
            {
                dst[i * dst_offset + dst_idx] = src[
                        i * src_offset + offset + dst_idx];
            }
        }

        return dst;
    }

    /**
     * \brief Append a n-dimensional matrix to a destination address on an axis.
     * It also check if the axis in input is a valid value in relation with the
     * destination shape.
     * \tparam T Type of source and destination matrix elements.
     * \param dst               The destination n-dimensional matrix.
     * \param dst_shape         The shape of the destination matrix.
     * \param src               The source n-dimensional matrix.
     * \param src_shape_axis    The shape of the axis in which perform the
     *                          append of the source matrix.
     * \param axis              The axis in which perform the append.
     * \param dst_axis_offset   The offset of the axis in which perform the
     *                          append of the destination matrix.
     * \return T* The destination address.
     */
    template <typename T>
    static T* append_check(T* dst, std::vector<SizeType> dst_shape,
                           const T* src, SizeType src_shape_axis,
                           SizeType axis, SizeType dst_axis_offset)
    {
        if (axis >= dst_shape.size())
        {
            throw std::runtime_error("concatenate error: axis param overload.");
        }
        return append(dst, dst_shape, src, src_shape_axis,
                      axis, dst_axis_offset);
    }

    /**
     * \brief Extract a n-dimensional sub-matrix from a source address on an
     * axis. It also check if the axis in input is a valid value in relation
     * with the destination shape.
     * \tparam T Type of source and destination matrix elements.
     * \param dst               The destination n-dimensional matrix.
     * \param dst_shape         The shape of the destination matrix.
     * \param src               The source n-dimensional matrix.
     * \param src_shape_axis    The shape of the axis in which perform the
     *                          extract of the source matrix.
     * \param axis              The axis in which perform the extract.
     * \param src_axis_offset   The offset of the axis in which perform the
     *                          extract of the source matrix.
     * \return T* The destination address.
     */
    template <typename T>
    static T* extract_check(T* dst, const std::vector<SizeType>& dst_shape,
                            const T* src, SizeType src_shape_axis,
                            SizeType axis, SizeType src_axis_offset)
    {
        if (axis >= dst_shape.size())
        {
            throw std::runtime_error("extract error: axis param overload.");
        }
        return extract(dst, dst_shape, src, src_shape_axis,
                       axis, src_axis_offset);
    }

    /**
     * \brief Concatenate two cubes on the specified axis.
     * \tparam T Type of source and destination matrix elements.
     * \param dst        The destination cube pointer.
     * \param src1       The first source cube to concatenate.
     * \param src1_shape The shape of the first cube.
     * \param src2       The second source cube to concatenate.
     * \param src2_shape The shape of the second cube.
     * \param axis       The axis in which perform the concatenation of the
     *                   two cubes.
     * \return T* The destination address.
     */
    template <typename T>
    static T* concatenate(T* dst,
                          const T* src1, Shape3d src1_shape,
                          const T* src2, Shape3d src2_shape, SizeType axis)
    {
        // Check valid params.
        if (axis >= Shape3d::SIZE)
        {
            throw std::runtime_error("concatenate error: axis param overload.");
        }
        for (SizeType i = 0; i < Shape3d::SIZE; ++i)
        {
            if (i != axis && src1_shape[i] != src2_shape[i])
            {
                throw std::runtime_error("concatenate error: shape invalid.");
            }
        }
        Shape3d dst_shape(src1_shape);
        dst_shape[axis] += src2_shape[axis];
        append(dst, dst_shape, src1, src1_shape[axis], axis,0);
        return append(dst, dst_shape, src2, src2_shape[axis],
                      axis, src1_shape[axis]);
    }

    /**
     * \brief Separate two cubes on the specified axis.
     * \tparam T Type of source and destination matrix elements.
     * \param dst1       The first destination cube pointer.
     * \param dst1_shape The shape of the first destination cube.
     * \param dst2       The second destination cube pointer.
     * \param dst2_shape The shape of the second destination cube.
     * \param src        The source cube pointer to separate.
     * \param axis       The axis in which perform the separation.
     * \return T* The destination address.
     */
    template <typename T>
    static T* separate(T* dst1, Shape3d dst1_shape,
                       T* dst2, Shape3d dst2_shape,
                       const T* src, SizeType axis)
    {
        // Check valid params.
        if (axis >= Shape3d::SIZE)
        {
            throw std::runtime_error("separate error: axis param overload.");
        }
        for (SizeType i = 0; i < Shape3d::SIZE; ++i)
        {
            if (i != axis && dst1_shape[i] != dst2_shape[i])
            {
                throw std::runtime_error("separate error: shape invalid.");
            }
        }

        auto src_shape_axis = dst1_shape[axis] + dst2_shape[axis];
        extract(dst1, dst1_shape, src, src_shape_axis, axis, 0);
        return extract(dst2, dst2_shape, src, src_shape_axis,
                      axis, dst1_shape[axis]);
    }

    /**
     * \brief Concatenation of N cubes contained in the src address.
     * \tparam T Type of source and destination matrix elements.
     * \param dst        The destination cube pointer.
     * \param src        The source pointer of the cubes to concatenate.
     * \param src_shapes The vector of shapes of the cubes contained in src.
     * \param axis       The axis in which perform the concatenation.
     * \return T* The destination address.
     */
    template <typename T>
    static T* concatenate(T* dst, const T* src,
                          std::vector<Shape3d> src_shapes, SizeType axis)
    {
        // Check valid params.
        if (axis >= Shape3d::SIZE)
        {
            throw std::runtime_error("separate error: axis param overload.");
        }
        for (SizeType shape_idx = 1; shape_idx < src_shapes.size(); ++shape_idx)
        {
            for (SizeType i = 0; i < DLMath::Shape3d::SIZE; ++i)
            {
                if (i != axis
                    && src_shapes[shape_idx - 1][i] != src_shapes[shape_idx][i])
                {
                    throw std::runtime_error("separate layer error: "
                                             "shapes invalid.");
                }
            }
        }

        // Calculate dst_shape.
        Shape3d dst_shape(src_shapes[0]);
        for (SizeType i = 1; i < src_shapes.size(); ++i)
        {
            dst_shape[axis] += src_shapes[i][axis];
        }

        SizeType dst_axis_offset = 0;
        SizeType src_offset = 0;
        for (auto& src_shape: src_shapes)
        {
            append(dst, dst_shape, src + src_offset, src_shape[axis],
                   axis, dst_axis_offset);
            dst_axis_offset += src_shape[axis];
            src_offset += src_shape.size();
        }

        return dst;
    }

    /**
     * \brief Separation in N cubes from a cube contained in the src address.
     * \tparam T Type of source and destination matrix elements.
     * \param dst        The destination address in which place the divided
     *                   N cubes.
     * \param dst_shapes The vector of shapes of the cubes divides in dst.
     * \param src        The source cube to divide in N cubes.
     * \param axis       The axis in which perform the concatenation.
     * \return T* The destination address.
     */
    template <typename T>
    static T* separate(T* dst, std::vector<Shape3d> dst_shapes,
                       const T* src, SizeType axis)
    {
        // Check valid params.
        if (axis >= Shape3d::SIZE)
        {
            throw std::runtime_error("separate error: axis param overload.");
        }
        for (SizeType shape_idx = 1; shape_idx < dst_shapes.size(); ++shape_idx)
        {
            for (SizeType i = 0; i < DLMath::Shape3d::SIZE; ++i)
            {
                if (i != axis
                    && dst_shapes[shape_idx - 1][i] != dst_shapes[shape_idx][i])
                {
                    throw std::runtime_error("separate layer error: "
                                             "shapes invalid.");
                }
            }
        }

        // Calculate src_shape[axis].
        SizeType src_shape_axis = 0;
        for (SizeType i = 0; i < dst_shapes.size(); ++i)
        {
            src_shape_axis += dst_shapes[i][axis];
        }

        SizeType src_axis_offset = 0;
        SizeType dst_offset = 0;
        for (auto& dst_shape: dst_shapes)
        {
            extract(dst + dst_offset, dst_shape, src, src_shape_axis,
                    axis, src_axis_offset);
            src_axis_offset += dst_shape[axis];
            dst_offset += dst_shape.size();
        }

        return dst;
    }

private:
    /**
     * \brief Sum of multiplication between the kernel and the source matrix
     * for Convolution 3D.
     * \tparam T        Type of each source and destination elements.
     * \param src       The source matrix on which calculate the convolution.
     * \param src_shape The shape of the source matrix: height, width, channels.
     * \param k         The kernel matrix to use for convolution.
     * \param k_shape   The shape of the kernel: height, width.
     *                  The third dimension is the same of the src matrix.
     * \param n_filters The number of filters contained in k to apply to the
     *                  matrix.
     * \param col       The column in which the kernel is moved over the source
     *                  matrix.
     * \param row       The row in which the kernel is moved over the source
     *                  matrix.
     * \return The value of the convolution 2D between the kernel and the
     * source matrix in the current position.
     */
    template <typename T>
    static void _conv4d_op(T* dst, Shape2d dst_shape, Coord2d dst_coord,
                           const T* src, Shape3d src_shape,
                           const T* k, Shape2d k_shape, SizeType n_filters,
                           int64_t row, int64_t col)
    {
        auto k_size = k_shape.size() * src_shape.channels();
        auto k_step = k_shape.width() * src_shape.channels();
        auto src_step = src_shape.width() * src_shape.channels();
        for (SizeType f = 0; f < n_filters; ++f)
        {
            T sum = 0;
            for (SizeType k_i = 0; k_i < k_size; ++k_i)
            {
                auto row_k = k_i / k_step;
                auto col_k = k_i % k_step;
                auto row_src = row + static_cast<int64_t>(row_k);
                auto col_src = col + static_cast<int64_t>(col_k);
                if (col_src < 0 || row_src < 0 ||
                    col_src >= static_cast<int64_t>(src_step) ||
                    row_src >= static_cast<int64_t>(src_shape.height()))
                {
                    continue; //< zero-padding.
                }
                sum += src[row_src * static_cast<int64_t>(src_step)
                           + col_src] * k[k_i * n_filters + f];
            }
            dst[dst_coord.row * dst_shape.width() * n_filters
                + dst_coord.col * n_filters
                + f] = sum;
        }
    }

    /**
     * \brief Maximum value of the kernel portion in the source matrix.
     * \tparam T        Type of each source and destination elements.
     * \param src       The source matrix on which calculate the max pooling.
     * \param src_shape The shape of the source matrix: height, width, channels.
     * \param k         Parameter not used (nullptr).
     * \param k_shape   The shape of the kernel: height, width.
     *                  The third dimension is the same of the src matrix.
     * \param n_filters The number of filters contained in k to apply to the
     *                  matrix.
     * \param col       The column in which the kernel is moved over the source
     *                  matrix.
     * \param row       The row in which the kernel is moved over the source
     *                  matrix.
     * \return The value of the max pooling of the kernel portion defined by
     * the current position in the source matrix.
     */
    template <typename T>
    static void _max_pool_op(T* dst, Shape2d dst_shape, Coord2d dst_coord,
                          const T* src, Shape3d src_shape,
                          const T* k, Shape2d k_shape, SizeType n_filters,
                          int64_t row, int64_t col)
    {
        (void) k;
        (void) n_filters;
        auto src_step = src_shape.width() * src_shape.channels();
        auto dst_step = dst_shape.width() * src_shape.channels();
        for (SizeType c = 0; c < src_shape.channels(); ++c)
        {
            T max = src[
                row * static_cast<int64_t>(src_step)
                + col + static_cast<int64_t>(c)];
            for (SizeType k_i = 1; k_i < k_shape.height() * k_shape.width(); ++k_i)
            {
                auto row_k = k_i / k_shape.width();
                auto col_k = k_i % k_shape.width();
                auto row_src = (row + static_cast<int64_t>(row_k))
                    * static_cast<int64_t>(src_step);
                auto col_src = col
                    + static_cast<int64_t>(col_k * src_shape.channels())
                    + static_cast<int64_t>(c);
                auto curr_val = src[row_src + col_src];
                if (curr_val > max) max = curr_val;
            }
            dst[dst_coord.row * dst_step
                + dst_coord.col * src_shape.channels() + c] = max;
        }

    }

    /**
     * \brief Average value of the kernel portion in the source matrix.
     * \tparam T        Type of each source and destination elements.
     * \param src       The source matrix on which calculate the average
     *                  pooling.
     * \param src_shape The shape of the source matrix: height, width, channels.
     * \param k         Parameter not used (nullptr).
     * \param k_shape   The shape of the kernel: height, width.
     *                  The third dimension is the same of the src matrix.
     * \param n_filters The number of filters contained in k to apply to the
     *                  matrix.
     * \param col       The column in which the kernel is moved over the source
     *                  matrix.
     * \param row       The row in which the kernel is moved over the source
     *                  matrix.
     * \return The value of the max pooling of the kernel portion defined by
     * the current position in the source matrix.
     */
    template <typename T>
    static void _avg_pool_op(T* dst, Shape2d dst_shape, Coord2d dst_coord,
                          const T* src, Shape3d src_shape,
                          const T* k, Shape2d k_shape, SizeType n_filters,
                          int64_t row, int64_t col)
    {
        (void) k;
        (void) n_filters;
        auto src_step = src_shape.width() * src_shape.channels();
        auto dst_step = dst_shape.width() * src_shape.channels();
        for (SizeType c = 0; c < src_shape.channels(); ++c)
        {
            T sum = 0;
            for (SizeType k_i = 0; k_i < k_shape.height() * k_shape.width(); ++k_i)
            {
                auto row_k = k_i / k_shape.width();
                auto col_k = k_i % k_shape.width();
                auto row_src = (row + static_cast<int64_t>(row_k))
                    * static_cast<int64_t>(src_step);
                auto col_src = col
                    + static_cast<int64_t>(col_k * src_shape.channels())
                    + static_cast<int64_t>(c);
                sum += src[row_src + col_src];
            }
            dst[dst_coord.row * dst_step
                + dst_coord.col * src_shape.channels() + c]
                = sum / (k_shape.height() * k_shape.width());
        }
    }

};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_DLMATH_HPP
