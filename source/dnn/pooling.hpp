/***************************************************************************
 *            dnn/pooling.hpp
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

/*! \file  dnn/pooling.hpp
 *  \brief Pooling abstract class layer.
 */

#ifndef EDGE_LEARNING_DNN_POOLING_HPP
#define EDGE_LEARNING_DNN_POOLING_HPP

#include "feedforward.hpp"

#include "dlmath.hpp"

#include <string>
#include <vector>


namespace EdgeLearning {

class PoolingLayer : public FeedforwardLayer
{
public:
    PoolingLayer(Model& model,
                 Activation activation = Activation::ReLU,
                 DLMath::Shape3d input_shape = {0, 0, 1},
                 DLMath::Shape2d kernel_shape = {0},
                 DLMath::Shape2d stride = {1},
                 std::string name = std::string(),
                 std::string prefix_name = std::string());

    /**
     * \brief No initialization is needed for Pooling layers.
     * \param rne Not used.
     */
    void init(RneType& rne) override { (void) rne; };

    /**
     * \brief The input data should have size _input_size, that is the shape of
     * the input matrix height x width x channels.
     * \param inputs The input matrix of input size = height * width * channels.
     */
    virtual void forward(const NumType *inputs) override = 0;

    /**
     * \brief The gradient data should have size _output_size.
     * Compute dJ/dz = dJ/dg(z) * dg(z)/dz
     * where dJ/dg(z) is the input gradients, dg(z)/dz is the activation_grad 
     * computed in the function and dJ/dz will be the result saved in 
     * _activation_gradients.
     * \param gradients The gradients of the subsequent layer. The size of
     * the gradients is _output_size that is: h_out x w_out x n_filters, where
     *  h_out  = ((h_in - h_kernel) / h_stride) + 1
     *  w_out  = ((w_in - w_kernel) / w_stride) + 1
     */
    void reverse(const NumType *gradients) override = 0;

    /**
     * \brief The pooling Layer doesn't have any learnable parameters.
     * \return SizeType 0.
     */
    SizeType param_count() const noexcept override
    {
        return 0;
    }

    /**
     * \brief Pooling layers do not have params.
     * \param index Not used.
     * \return NumType* nullptr
     */
    NumType* param(SizeType index) override { (void) index; return nullptr; }

    /**
     * \brief Pooling layers do not have gradients.
     * \param index Not used.
     * \return NumType* nullptr
     */
    NumType* gradient(SizeType index) override
    {
        (void) index;
        return nullptr;
    }

    void print() const override;

    /**
     * \brief Input shape setter. In this layer, all the 3 fields contained in
     * DLMath::Shape3d are used to calculate the layer input size.
     * \param input_shape 3D object with input matrix shape.
     */
    void input_size(DLMath::Shape3d input_shape) override;

private:
    /// \brief Input shape. Size: height * width * channels (_input_size).
    DLMath::Shape3d _input_shape;
    /**
     * \brief Output shape.
     * Size: height_out * width_out * channels (_output_size).
     *  height_out  = ((h_in - h_kernel) / h_stride) + 1
     *  width_out   = ((w_in - w_kernel) / w_stride) + 1
     */
    DLMath::Shape3d _output_shape;
    /// \brief Kernel shape. Size: height_kernel * width_kernel.
    DLMath::Shape2d _kernel_shape;

    DLMath::Shape2d _stride;    ///< \brief Stride along axis.
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_POOLING_HPP
