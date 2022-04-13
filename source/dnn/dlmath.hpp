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
 *  \brief Simply replace me.
 */

#include <cmath>
#include <functional>
#include <cassert>
#include <stdexcept>
#include <tuple>
#include <algorithm>
#include <iterator>
#include <limits>

#include <iostream>

#ifndef EDGE_LEARNING_DNN_DLMATH_HPP
#define EDGE_LEARNING_DNN_DLMATH_HPP

namespace EdgeLearning {

class DLMath 
{
public:
    static constexpr RneType::result_type max_rand = 
        std::numeric_limits<RneType::result_type>::max();

    /**
     * \brief Gaussian Probability Density Function.
     * \tparam T      Input and output type.
     * \param x       Input value to compute.
     * \param mean    Mean of the probability distribution required.
     * \param std_dev Standard Deviation of the probability distribution required.
     * \return std::function<T(RneType)> The distribution function.
     */
    template <typename T>
    static std::function<T(RneType&)> normal_pdf(double mean, double std_dev)
    {
        T std_dev_coverage = T{3.0 * std_dev};
        
        std::function<T(RneType&)> ret = 
            [std_dev_coverage, mean](RneType& x) 
        {
            T rand = ((static_cast<T>(x()) / static_cast<T>(max_rand)) * T{2.0}) - T{1.0};
            rand = (rand * std_dev_coverage) + mean;
            return rand;
        };
        return ret;
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
    static T* matarr_mul(T* arr_dst, const T* mat_src, const T* arr_src, 
        SizeType rows, SizeType cols)
    {
        if (arr_src == arr_dst) 
        {
            throw std::runtime_error("arr_src, arr_dst have to be different "
                                     "in order to perform matarr_mul");
        }

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

    /**
     * \brief ReLU Function.
     * relu(x) = max(0, x)
     * \tparam T Type of the input and return type.
     * \param x  Input value.
     * \return T
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
            dst[i] = (src[i] > T{0}) ? T{1} : T{0};
        }
        return dst;
    }

    /**
     * \brief Derivative Optimized of Softmax Function with the value of the 
     * argmax already saved in the src array. Source and Destination has to be 
     * differents.
     * softmax'(z)_i = \sum_j(
     *  softmax(z_i)(1 - softmax(z_i)) if i == j else -softmax(z_i)softmax(z_j))
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result. It has to be different by src.
     * \param src    Array of read elements. It has to be different by dst.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* softmax_1_opt(T* dst, const T* src, SizeType length)
    {
        if (src == dst) 
        {
            throw std::runtime_error("src, dst have to be different "
                                    "in order to perform softmax_1_opt");
        }

        for (SizeType i = 0; i < length; ++i)
        {
            dst[i] = T{0.0};
            for(SizeType j = 0; j < length; ++j)
            {
                dst[i] += (i == j) 
                    ? src[i] * (T{1.0} - src[i]) 
                    : -src[i] * src[j];
            }
        }
        return dst;
    }

    /**
     * \brief Derivative of Softmax Function.
     * softmax'(z)_i = \sum_j(
     *  softmax(z_i)(1 - softmax(z_i)) if i == j else -softmax(z_i)softmax(z_j))
     * \tparam T     Type of each source and destination elements.
     * \param dst    Array to write the result.
     * \param src    Array of read elements.
     * \param length Length of the arrays.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* softmax_1(T* dst, const T* src, SizeType length)
    {
        T* tmp = new T[length];
        assert(tmp);
        softmax(tmp, src, length);
        softmax_1_opt(dst, tmp, length);
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
    static T cross_entropy_1(T y, T y_hat, T norm)
    {
        return norm * (-y / (std::max(y_hat, 
            std::numeric_limits<T>::epsilon())));
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
     * \brief Hyperbolic Tangent Function.
     * \tparam T Type of the input and return type.
     * \param x  Input value.
     * \return T
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
     * \brief Hyperbolic Tangent Function first derivative.
     * \tparam T Type of the input and return type.
     * \param x  Input value.
     * \return T
     */
    template <typename T>
    static T tanh_1(T x)
    {
        T t = std::tanh(x);
        return T{1} - t * t;
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

    template <typename T>
    static T* conv2d(T* dst,
                     const T* src, SizeType width, SizeType height,
                     const T* k, SizeType f,
                     SizeType s = 1, SizeType p = 0)
    {
        SizeType row = 0, col = 0;
        s = std::max(s, SizeType(1));
        auto width_dst = ((width - f + 2 * p) / s) + 1;
        auto height_dst = ((height - f + 2 * p) / s) + 1;
        for (SizeType row_dst = 0; row_dst < height_dst; ++row_dst)
        {
            for (SizeType col_dst = 0; col_dst < width_dst; ++col_dst)
            {
                SizeType sum = 0;
                for (SizeType i = 0; i < f * f; ++i)
                {
                    auto row_k = i / f;
                    auto col_k = i % f;
                    auto row_src = static_cast<int64_t>(row + row_k)
                        - static_cast<int64_t>(p);
                    auto col_src = static_cast<int64_t>(col + col_k)
                        - static_cast<int64_t>(p);
                    if (col_src < 0 || row_src < 0 ||
                        col_src >= static_cast<int64_t>(width) ||
                        row_src >= static_cast<int64_t>(height))
                        continue; //< zero-padding.
                    sum += src[row_src * static_cast<int64_t>(width) + col_src]
                        * k[i];
                }
                dst[row_dst * width_dst + col_dst] = sum;
                col += s;
            }
            row += s;
            col = 0;
        }
        return dst;
    }
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_DLMATH_HPP
