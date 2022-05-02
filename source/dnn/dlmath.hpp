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

    struct Shape2d {
        Shape2d(SizeType h, SizeType w)
            : height{h}
            , width{w}
        {}

        Shape2d(SizeType s)
            : height{s}
            , width{s}
        {}

        [[nodiscard]] SizeType size() const { return height * width; }

        SizeType height;
        SizeType width;
    };

    struct Shape3d {
        Shape3d(Shape2d s2d)
            : height{s2d.height}
            , width{s2d.width}
            , channels{1}
        {}

        Shape3d(SizeType h, SizeType w=1, SizeType c=1)
            : height{h}
            , width{w}
            , channels{c}
        {}

        SizeType size() { return height * width * channels; }

        SizeType height;
        SizeType width;
        SizeType channels;
    };

    enum class ProbabilityDensityFunction
    {
        NORMAL,
        UNIFORM
    };

    /**
     * \brief Gaussian Probability Density Function.
     * \tparam T      Input and output type.
     * \param mean    Mean of the probability distribution required.
     * \param std_dev Standard Deviation of the probability distribution required.
     * \return std::function<T(RneType)> The distribution function.
     */
    template <typename T>
    static std::function<T(RneType&)> normal_pdf(NumType mean, NumType std_dev)
    {
        return std::normal_distribution<NumType>{mean, std_dev};
    }

    /**
     * \brief Uniform Probability Density Function.
     * \tparam T      Input and output type.
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
     * \tparam T      Input and output type.
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

    /**
     * \brief Convolution 2D of a source 2D matrix and a squared kernel.
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
    static T* conv2d(T* dst, const T* src, Shape2d src_shape,
                     const T* k, Shape2d k_shape,
                     Shape2d s = {1, 1}, Shape2d p = {0, 0})
    {
        return conv3d<T>(dst, src, Shape3d(src_shape), k, k_shape, s, p);
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
    static T* conv3d(T* dst, const T* src, Shape3d src_shape,
                     const T* k, Shape2d k_shape,
                     Shape2d s = {1, 1}, Shape2d p = {0, 0})
    {
        return conv4d<T>(dst, src, Shape3d(src_shape), k, k_shape, 1, s, p);
    }

    /**
     * \brief Multi Convolution 2D of a 3D source matrix iterated on n_filters
     * of cubic kernel.
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
    static T* conv4d(T* dst, const T* src, Shape3d src_shape,
                     const T* k, Shape2d k_shape, SizeType n_filters = 1,
                     Shape2d s = {1, 1}, Shape2d p = {0, 0})
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
        s.width = std::max(s.width, SizeType(1));
        s.height = std::max(s.height, SizeType(1));
        auto width_dst = ((src_shape.width - k_shape.width + 2 * p.width)
            / s.width) + 1;
        auto height_dst = ((src_shape.height - k_shape.height + 2 * p.height)
            / s.height) + 1;
        for (SizeType row_dst = 0; row_dst < height_dst; ++row_dst)
        {
            for (SizeType col_dst = 0; col_dst < width_dst; ++col_dst)
            {
                auto col = (static_cast<int64_t>(col_dst * s.width)
                    - static_cast<int64_t>(p.width))
                    * static_cast<int64_t>(src_shape.channels);
                auto row = static_cast<int64_t>(row_dst * s.height)
                    - static_cast<int64_t>(p.height);
                k_to_src_operation(
                    dst, {height_dst, width_dst}, {row_dst, col_dst},
                    src, src_shape, k, k_shape, n_filters, row, col);
            }
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
        auto k_size = k_shape.size() * src_shape.channels;
        auto k_step = k_shape.width * src_shape.channels;
        auto src_step = src_shape.width * src_shape.channels;
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
                    row_src >= static_cast<int64_t>(src_shape.height))
                {
                    continue; //< zero-padding.
                }
                sum += src[row_src * static_cast<int64_t>(src_step)
                           + col_src] * k[k_i * n_filters + f];
            }
            dst[dst_coord.row * dst_shape.width * n_filters
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
        auto src_step = src_shape.width * src_shape.channels;
        auto dst_step = dst_shape.width * src_shape.channels;
        for (SizeType c = 0; c < src_shape.channels; ++c)
        {
            T max = src[
                row * static_cast<int64_t>(src_step)
                + col + static_cast<int64_t>(c)];
            for (SizeType k_i = 1; k_i < k_shape.height * k_shape.width; ++k_i)
            {
                auto row_k = k_i / k_shape.width;
                auto col_k = k_i % k_shape.width;
                auto row_src = (row + static_cast<int64_t>(row_k))
                    * static_cast<int64_t>(src_step);
                auto col_src = col
                    + static_cast<int64_t>(col_k * src_shape.channels)
                    + static_cast<int64_t>(c);
                auto curr_val = src[row_src + col_src];
                if (curr_val > max) max = curr_val;
            }
            dst[dst_coord.row * dst_step
                + dst_coord.col * src_shape.channels + c] = max;
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
        auto src_step = src_shape.width * src_shape.channels;
        auto dst_step = dst_shape.width * src_shape.channels;
        for (SizeType c = 0; c < src_shape.channels; ++c)
        {
            T sum = 0;
            for (SizeType k_i = 0; k_i < k_shape.height * k_shape.width; ++k_i)
            {
                auto row_k = k_i / k_shape.width;
                auto col_k = k_i % k_shape.width;
                auto row_src = (row + static_cast<int64_t>(row_k))
                    * static_cast<int64_t>(src_step);
                auto col_src = col
                    + static_cast<int64_t>(col_k * src_shape.channels)
                    + static_cast<int64_t>(c);
                sum += src[row_src + col_src];
            }
            dst[dst_coord.row * dst_step
                + dst_coord.col * src_shape.channels + c]
                = sum / (k_shape.height * k_shape.width);
        }
    }

};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_DLMATH_HPP
