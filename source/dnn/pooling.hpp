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

/**
 * \brief Pooling Layer.
 *
 * Input shape.
 * Size: height * width * channels (_input_size).
 *
 * Output shape.
 * Size: height_out * width_out * channels (_output_size).
 *  height_out  = ((h_in - h_kernel) / h_stride) + 1
 *  width_out   = ((w_in - w_kernel) / w_stride) + 1
 */
class PoolingLayer : public FeedforwardLayer
{
public:
    PoolingLayer(Model& model,
                 DLMath::Shape3d input_shape = {0, 0, 1},
                 DLMath::Shape2d kernel_shape = {0},
                 DLMath::Shape2d stride = {1},
                 std::string name = std::string(),
                 std::string prefix_name = std::string());

    /**
     * \brief No initialization is needed for Pooling layers.
     * \param init  Not used.
     * \param pdf   Not used.
     * \param rne   Not used.
     */
    void init(
        InitializationFunction init = InitializationFunction::KAIMING,
        ProbabilityDensityFunction pdf = ProbabilityDensityFunction::NORMAL,
        RneType rne = RneType(std::random_device{}()))
        override
    {
        (void) init;
        (void) pdf;
        (void) rne;
    };

    /**
     * \brief The pooling Layer doesn't have any learnable parameters.
     * \return SizeType 0.
     */
    [[nodiscard]] SizeType param_count() const noexcept override
    {
        return 0;
    }

    /**
     * \brief Pooling layers do not have params.
     * \param index Not used.
     * \return NumType* nullptr
     */
    NumType& param(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("Pooling layers do not have params");
    }

    /**
     * \brief Pooling layers do not have gradients.
     * \param index Not used.
     * \return NumType* nullptr
     */
    NumType& gradient(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("Pooling layers do not have gradients");
    }

    void print() const override;

    /**
     * \brief Input shape getter of Pooling layer.
     * \return DLMath::Shape3d the input shape 3D.
     */
    [[nodiscard]] const DLMath::Shape3d& input_shape() const override
    {
        return FeedforwardLayer::input_shape();
    }

    /**
     * \brief Input shape setter. In this layer, all the 3 fields contained in
     * DLMath::Shape3d are used to calculate the layer input size.
     * \param input_shape 3D object with input matrix shape.
     */
    void input_shape(DLMath::Shape3d input_shape) override;

    /**
     * \brief Kernel shape getter of Pooling layer.
     * \return DLMath::Shape2d the kernel shape 2D.
     */
    [[nodiscard]] const DLMath::Shape2d& kernel_shape() const
    { return _kernel_shape; }

protected:
    /// \brief Kernel shape. Size: height_kernel * width_kernel.
    DLMath::Shape2d _kernel_shape;

    DLMath::Shape2d _stride;    ///< \brief Stride along axis.
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_POOLING_HPP
