/***************************************************************************
 *            dnn/concatenate.hpp
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

/*! \file  dnn/concatenate.hpp
 *  \brief Concatenate layer.
 */

#ifndef EDGE_LEARNING_DNN_CONCATENATE_HPP
#define EDGE_LEARNING_DNN_CONCATENATE_HPP

#include "feedforward.hpp"

#include <string>
#include <vector>


namespace EdgeLearning {

class ConcatenateLayer : public FeedforwardLayer
{
public:
    static const std::string TYPE;

    ConcatenateLayer(std::string name = std::string(),
                     std::vector<DLMath::Shape3d> shapes = {},
                     SizeType axis = 0);

    [[nodiscard]] inline const std::string& type() const override
    { return TYPE; }

    /**
     * \brief No initialization is needed for Pooling layers.
     * \param init  Not used.
     * \param pdf   Not used.
     * \param rne   Not used.
     */
    void init(
            InitializationFunction init = InitializationFunction::KAIMING,
            ProbabilityDensityFunction pdf = ProbabilityDensityFunction::NORMAL,
            RneType rne = RneType(std::random_device{}())) override
    {
        (void) init;
        (void) pdf;
        (void) rne;
    };

    /**
     * \brief The input data should have size _input_size.
     * \param inputs const std::vector<NumType>& Layer input in forward.
     */
    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override;

    /**
     * \brief The gradient data should have size _output_size.
     * Compute dJ/dz = dJ/dg(z) * dg(z)/dz
     * where dJ/dg(z) is the input gradients, dg(z)/dz is the activation_grad 
     * computed in the function and dJ/dz will be the result saved in 
     * _activation_gradients.
     * \param gradients const std::vector<NumType>& Layer gradients in backward.
     */
    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override;

    /**
     * \brief No params in concatenate layer.
     * \return SizeType 0.
     */
    [[nodiscard]] SizeType param_count() const noexcept override { return 0; }

    /**
     * \brief Concatenate layers do not have params.
     * \param index Not used.
     * \return NumType* nullptr
     */
    NumType& param(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("Concatenate layers do not have params");
    }

    /**
     * \brief Concatenate layers do not have gradients.
     * \param index Not used.
     * \return NumType* nullptr
     */
    NumType& gradient(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("Concatenate layers do not have gradients");
    }

    [[nodiscard]] SharedPtr clone() const override
    {
        return std::make_shared<ConcatenateLayer>(*this);
    }

    void print() const override;

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

protected:
    /**
     * \brief Setter of input_shape class field.
     * \param input_shape DLMath::Shape3d Shape param used to take the size and
     * assign it to input_shape.
     * The operation also performs a resize of the weights and its gradients.
     */
    void _set_input_shape(LayerShape input_shape) override;

private:
    SizeType _axis;
    SizeType _current_input_layer;
    DLMath::Shape3d _current_output_shape;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_CONCATENATE_HPP
