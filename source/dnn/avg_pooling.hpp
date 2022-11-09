/***************************************************************************
 *            dnn/avg_pooling.hpp
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

/*! \file  dnn/avg_pooling.hpp
 *  \brief Average Pooling class layer.
 */

#ifndef EDGE_LEARNING_DNN_AVG_POOLING_HPP
#define EDGE_LEARNING_DNN_AVG_POOLING_HPP

#include "pooling.hpp"

#include "dlmath.hpp"

#include <string>
#include <vector>


namespace EdgeLearning {

class AveragePoolingLayer : public PoolingLayer
{
public:
    static const std::string TYPE;

    AveragePoolingLayer(std::string name = std::string(),
                        DLMath::Shape3d input_shape = {0, 0, 1},
                        DLMath::Shape2d kernel_shape = {0},
                        DLMath::Shape2d stride = {1});

    [[nodiscard]] inline const std::string& type() const override
    { return TYPE; }

    /**
     * \brief The input data should have size _input_size, that is the shape of
     * the input matrix height x width x channels.
     * \param inputs const std::vector<NumType>& The input matrix of
     * input size = height * width * channels.
     * \return const std::vector<NumType>&
     */
    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override;

    /**
     * \brief The gradient data should have size _output_size.
     * Compute dJ/dz = dJ/dg(z) * dg(z)/dz
     * where dJ/dg(z) is the input gradients, dg(z)/dz is the activation_grad
     * computed in the function and dJ/dz will be the result saved in
     * _activation_gradients.
     * \param gradients const std::vector<NumType>& The gradients of the
     * subsequent layer. The size of the gradients is _output_size that
     * is: h_out x w_out x n_filters, where
     *  h_out  = ((h_in - h_kernel) / h_stride) + 1
     *  w_out  = ((w_in - w_kernel) / w_stride) + 1
     * \return const std::vector<NumType>&
     */
    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override;

    [[nodiscard]] SharedPtr clone() const override
    {
        return std::make_shared<AveragePoolingLayer>(*this);
    }

private:
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_AVG_POOLING_HPP
