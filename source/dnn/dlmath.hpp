/***************************************************************************
 *            dlmath.hpp
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

/*! \file dlmath.hpp
 *  \brief Simply replace me.
 */

#include <cmath>
#include <functional>
#include <cassert>
#include <stdexcept>
#include <tuple>
#include <algorithm>
#include <iterator>


#ifndef ARIADNE_DL_DLMATH_HPP
#define ARIADNE_DL_DLMATH_HPP

namespace Ariadne {

class DLMath 
{
public:
    /**
     * \brief 1 / sqrt(2 * pi)
     */
    static constexpr double inv_sqrt_2pi = 0.3989422804014327;

    /**
     * \brief Gaussian Probability Density Function.
     * \tparam T      Input and output type.
     * \param x       Input value to compute.
     * \param mean    Mean of the probability distribution required.
     * \param std_dev Standard Deviation of the probability distribution required.
     * \return std::function<T(rne_t)> The distribution function.
     */
    template <typename T>
    static std::function<T(rne_t)> normal_pdf(float mean, float std_dev)
    {
        double inv_sqrt_2pi_std_dev = (inv_sqrt_2pi / std_dev);
        
        std::function<T(rne_t)> ret = 
            [inv_sqrt_2pi_std_dev, mean, std_dev](rne_t x) 
        {
            double a = (double(x()) - mean) / std_dev;
            return T(inv_sqrt_2pi_std_dev * std::exp(-0.5f * a * a));
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
    static T* arr_mul(T* dst, const T* src1, const T* src2, size_t length)
    {
        for (size_t i = 0; i < length; ++i)
        {
            dst[i] = src1[i] * src2[i];
        }
        return dst;
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
    static T* arr_sum(T* dst, const T* src1, const T* src2, size_t length)
    {
        for (size_t i = 0; i < length; ++i)
        {
            dst[i] = src1[i] + src2[i];
        }
        return dst;
    }

    /**
     * \brief Multiplication between a matrix and an array.
     * Used for y = Wx
     * @tparam T      Type of each source and destination elements.
     * \param arr_dst Array destination to write the result.
     * \param mat_src Matrix source, left operand.
     * \param arr_src Array source, right operand.
     * \param rows    Amount of rows.
     * \param cols    Amount of columns.
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* matarr_mul(T* arr_dst, const T* mat_src, const T* arr_src, 
        size_t rows, size_t cols)
    {
        if (arr_src == arr_dst) 
        {
            throw std::runtime_error("arr_src, arr_dst have to be different "
                                     "in order to perform matarr_mul");
        }

        for (size_t i = 0; i < rows; ++i)
        {
            arr_dst[i] = T{0};
            for (size_t j = 0; j < cols; ++j)
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
    static T* relu(T* dst, const T* src, size_t length)
    {
        for (size_t i = 0; i < length; ++i)
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
    static T* softmax(T* dst, const T* src, size_t length)
    {
        // Compute the exponential of each value and compute the sum. 
        T sum_exp_z{0};
        for (size_t i = 0; i < length; ++i)
        {
            dst[i] = std::exp(src[i]);
            sum_exp_z += dst[i];
        }

        // Compute the inverse of the sum.
        T inv_sum_exp_z = T{1} / sum_exp_z;

        // Multiply the inverse of the sum for each value.
        for (size_t i = 0; i < length; ++i)
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
    static T* relu_1(T* dst, const T* src, size_t length)
    {
        for (size_t i = 0; i < length; ++i)
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
    static T* softmax_1_opt(T* dst, const T* src, size_t length)
    {
        if (src == dst) 
        {
            throw std::runtime_error("src, dst have to be different "
                                    "in order to perform softmax_1_opt");
        }

        for (size_t i = 0; i < length; ++i)
        {
            dst[i] = T{0};
            for(size_t j = 0; j < length; ++j)
            {
                dst[i] += (i == j) 
                    ? src[i] * (T{1} - src[i]) 
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
    static T* softmax_1(T* dst, const T* src, size_t length)
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
     * relu(x) = - y * log(max(y_hat, epsilon))
     * \tparam T Type of the input and return type.
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
     * cross_entropy(x) = - \sum_j(y_j * log( max(y_hat_j, epsilon) ))
     * \tparam T Type of the input and return type.
     * \param y      Target array values.
     * \param y_hat  Estimated array values.
     * \param length Length of the arrays.
     * \return T The resulting Cross-Entropy.
     */
    template <typename T>
    static T cross_entropy(const T* y, const T* y_hat, size_t length)
    {
        T ret{0.0};
        for (size_t i = 0; i < length; ++i)
        {
            ret += cross_entropy(y[i], y_hat[i]);
        }
        return ret;
    }

    /**
     * \brief Cross-Entropy Function first derivative.
     * cross_entropy'(x) = - y / max(y_hat, epsilon)
     * \tparam T Type of the input and return type.
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
     * cross_entropy'(x)_i = - y_i / max(y_hat_i, epsilon)
     * \tparam T Type of the input and return type.
     * \param y      Target array values.
     * \param y_hat  Estimated array values.
     * \param norm   Normalizer term.
     * \param length Length of the arrays. 
     * \return T* The destination array pointer.
     */
    template <typename T>
    static T* cross_entropy_1(T* dst, const T* y, const T* y_hat, T norm, 
        size_t length)
    {
        for (size_t i = 0; i < length; ++i)
        {
            dst[i] = cross_entropy_1(y[i], y_hat[i], norm);
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
    static T max(const T* src, size_t length) 
    {
        return *std::max_element(src, src + length);
    }

    /**
     * \brief Find the argument that point to the maximum value.
     * \tparam T Type of the input.
     * \param src    Source array.
     * \param length Length of the array.
     * \return size_t The argmax index.
     */
    template <typename T>
    static size_t argmax(const T* src, size_t length) 
    {
        return static_cast<size_t>(std::distance(src, 
            std::max_element(src, src + length)));
    }

    /**
     * \brief Find the max and the argmax values.
     * \tparam T Type of the input.
     * \param src    Source array.
     * \param length Length of the array.
     * \return std::tuple<T, size_t> Tuple of max and argmax.
     */
    template <typename T>
    static std::tuple<T, size_t> max_and_argmax(T* src, size_t length) 
    {
        auto max_iter = std::max_element(src, src + length);
        auto dist = static_cast<size_t>(std::distance(src, max_iter));
        return {*max_iter, dist};
    }
};

} // namespace Ariadne

#endif // ARIADNE_DL_DLMATH_HPP
