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

/**
 * \brief Convolutional Layer.
 *
 * Input shape.
 * Size: height * width * channels (_input_size).
 *
 * Output shape.
 * Size: height_out * width_out * n_filters (_output_size).
 *  height_out  = ((h_in - h_kernel + (2 * h_padding)) / h_stride) + 1
 *  width_out   = ((w_in - w_kernel + (2 * w_padding)) / w_stride) + 1
 */
class ConvolutionalLayer : public FeedforwardLayer
{
public:
    static const std::string TYPE;

    ConvolutionalLayer(std::string name = std::string(),
           DLMath::Shape3d input_shape = {0, 0, 1},
           DLMath::Shape2d kernel_shape = {0}, SizeType n_filters = 0,
           DLMath::Shape2d stride = {1}, DLMath::Shape2d padding = {0});

    [[nodiscard]] inline const std::string& type() const override
    { return TYPE; }

    void init(
        InitializationFunction init = InitializationFunction::KAIMING,
        ProbabilityDensityFunction pdf = ProbabilityDensityFunction::NORMAL,
        RneType rne = RneType(std::random_device{}()))
        override;

    /**
     * \brief The input data should have size _input_size, that is the shape of
     * the input matrix height x width x channels.
     * \param inputs The input matrix of input size = height * width * channels.
     */
    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override;

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
    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override;

    /**
     * \brief The kernels entries + bias entries.
     * \return SizeType The amount of learnable parameters, given by the
     * n_filters of kernel size and the bias on the result of the filters.
     */
    SizeType param_count() const noexcept override
    {
        return (_kernel_shape.size()
            * _shared_fields->input_shape().channels() * _n_filters)
            + _n_filters;
    }

    NumType& param(SizeType index) override;
    NumType& gradient(SizeType index) override;

    [[nodiscard]] SharedPtr clone() const override
    {
        return std::make_shared<ConvolutionalLayer>(*this);
    }

    void print() const override;

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

    /**
     * \brief Save the layer infos and weights to disk.
     * \return Json Layer dump.
     */
    Json dump() const override;

    /**
     * \brief Load the layer infos and weights from disk.
     * \param in const Json& Json to read.
     */
    void load(const Json& in) override;

    static DLMath::Shape3d calculate_output_shape(
        DLMath::Shape3d input_shape, DLMath::Shape2d kernel_shape,
        DLMath::Shape2d stride, DLMath::Shape2d padding, SizeType n_filters);

protected:
    /**
     * \brief Input shape setter. In this layer, all the 3 fields contained in
     * DLMath::Shape3d are used to calculate the layer input size.
     * \param input_shape 3D object with input matrix shape.
     * The operation also performs a resize of the weights and its gradients.
     * Since input_shape == output_size in activation layer, it overrides also
     * the output_size.
     */
    void _set_input_shape(LayerShape input_shape) override;

private:
    /// \brief Kernel shape. Size: height_kernel * width_kernel.
    DLMath::Shape2d _kernel_shape;

    SizeType _n_filters;        ///< \brief Number of Convolutional filters.
    DLMath::Shape2d _stride;    ///< \brief Stride along axis.
    DLMath::Shape2d _padding;   ///< \brief Padding along axis.

    // == Layer parameters ==
    /// \brief Kernels of the layer.
    /// Size: height_k * width_k * channels * n_filters.
    SharedParams _weights;
    /// \brief Biases of the layer. Size: n_filters.
    SharedParams _biases;

    // == Loss Gradients ==
    /// \brief Weight gradients.
    /// Size: height_k * width_k * channels * n_filters.
    Params _weight_gradients;
    /// \brief Biase gradients. Size: n_filters.
    Params _bias_gradients;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_CONVOLUTIONAL_HPP
