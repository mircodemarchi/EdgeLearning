/***************************************************************************
 *            dnn/convolutional.hpp
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

/*! \file  dnn/convolutional.hpp
 *  \brief Convolutional layer.
 */

#ifndef EDGE_LEARNING_DNN_CONVOLUTIONAL_HPP
#define EDGE_LEARNING_DNN_CONVOLUTIONAL_HPP

#include "feedforward.hpp"

#include "dlmath.hpp"

#include <string>
#include <vector>


namespace EdgeLearning {

class ConvolutionalLayer : public FeedforwardLayer
{
public:
    ConvolutionalLayer(Model& model,
           std::string name = std::string(),
           Activation activation = Activation::ReLU,
           DLMath::Shape3d input_shape = {0, 0, 1},
           DLMath::Shape2d kernel_shape = {0}, SizeType n_filters = 0,
           DLMath::Shape2d stride = {1}, DLMath::Shape2d padding = {0});

    void init(RneType& rne) override;

    /**
     * \brief The input data should have size _input_size, that is the shape of
     * the input matrix height x width x channels.
     * \param inputs The input matrix of input size = height * width * channels.
     */
    void forward(const NumType *inputs) override;

    /**
     * \brief The gradient data should have size _output_size.
     * Compute dJ/dz = dJ/dg(z) * dg(z)/dz
     * where dJ/dg(z) is the input gradients, dg(z)/dz is the activation_grad 
     * computed in the function and dJ/dz will be the result saved in 
     * _activation_gradients.
     * \param gradients The gradients of the subsequent layer. The size of
     * the gradients is _output_size that is: h_out x w_out x n_filters, where
     *  h_out  = ((h_in - h_kernel + (2 * h_padding)) / h_stride) + 1
     *  w_out  = ((w_in - w_kernel + (2 * w_padding)) / w_stride) + 1
     */
    void reverse(const NumType *gradients) override;

    /**
     * \brief The kernels entries + bias entries.
     * \return SizeType The amount of learnable parameters, given by the
     * n_filters of kernel size and the bias on the result of the filters.
     */
    SizeType param_count() const noexcept override
    {
        return (_kernel_shape.size() * _input_shape.channels * _n_filters)
            + _n_filters;
    }

    NumType* param(SizeType index) override;
    NumType* gradient(SizeType index) override;

    void print() const override;

    [[nodiscard]] SizeType input_size() const override
    {
        return Layer::input_size();
    }

    /**
     * \brief Input shape setter. In this layer, all the 3 fields contained in
     * DLMath::Shape3d are used to calculate the layer input size.
     * \param input_shape 3D object with input matrix shape.
     */
    void input_size(DLMath::Shape3d input_shape) override;

    /**
     * \brief Input shape getter of Convolution layer.
     * \return DLMath::Shape3d the input shape 3D.
     */
    [[nodiscard]] const DLMath::Shape3d& input_shape() const
    { return _input_shape; }

    /**
     * \brief Output shape getter of Convolution layer.
     * \return DLMath::Shape3d the output shape 3D.
     */
    [[nodiscard]] const DLMath::Shape3d& output_shape() const
    { return _output_shape; }

    /**
     * \brief Kernel shape getter of Convolution layer.
     * \return DLMath::Shape2d the kernel shape 2D.
     */
    [[nodiscard]] const DLMath::Shape2d& kernel_shape() const
    { return _kernel_shape; }

    /**
     * \brief Filters amount applied by Convolution layer.
     * \return The filters amount.
     */
    [[nodiscard]] SizeType n_filters() const
    { return _n_filters; }

private:
    /// \brief Input shape. Size: height * width * channels (_input_size).
    DLMath::Shape3d _input_shape;
    /**
     * \brief Output shape.
     * Size: height_out * width_out * n_filters (_output_size).
     *  height_out  = ((h_in - h_kernel + (2 * h_padding)) / h_stride) + 1
     *  width_out   = ((w_in - w_kernel + (2 * w_padding)) / w_stride) + 1
     */
    DLMath::Shape3d _output_shape;
    /// \brief Kernel shape. Size: height_kernel * width_kernel.
    DLMath::Shape2d _kernel_shape;

    SizeType _n_filters;        ///< \brief Number of Convolutional filters.
    DLMath::Shape2d _stride;    ///< \brief Stride along axis.
    DLMath::Shape2d _padding;   ///< \brief Padding along axis.

    // == Layer parameters ==
    /// \brief Kernels of the layer. Size: height_k * width_k * n_filters.
    std::vector<NumType> _weights;
    /// \brief Biases of the layer. Size: n_filters.
    std::vector<NumType> _biases;

    // == Loss Gradients ==
    /// \brief Weight gradients. Size: height_k * width_k * n_filters.
    std::vector<NumType> _weight_gradients;
    /// \brief Biase gradients. Size: n_filters.
    std::vector<NumType> _bias_gradients;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_CONVOLUTIONAL_HPP
