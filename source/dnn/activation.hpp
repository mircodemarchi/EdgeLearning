/***************************************************************************
 *            dnn/activation.hpp
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

/*! \file  dnn/activation.hpp
 *  \brief Dense layer.
 */

#ifndef EDGE_LEARNING_DNN_ACTIVATION_HPP
#define EDGE_LEARNING_DNN_ACTIVATION_HPP

#include "feedforward.hpp"

#include <string>
#include <vector>


namespace EdgeLearning {

class ActivationLayer : public FeedforwardLayer
{
public:
    ActivationLayer(Model& model, SizeType size = 0,
                    std::string name = std::string(),
                    std::string prefix_name = std::string());

    /**
     * \brief No initialization is needed for Activation layers.
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
     * \brief Activation layers do not have params.
     * \return SizeType 0
     */
    [[nodiscard]] SizeType param_count() const noexcept override { return 0; }

    /**
     * \brief Activation layers do not have params.
     * \param index Not used.
     * \return NumType* nullptr
     */
    NumType& param(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("Activation layers do not have params");
    }

    /**
     * \brief Activation layers do not have gradients.
     * \param index Not used.
     * \return NumType* nullptr
     */
    NumType& gradient(SizeType index) override
    {
        (void) index;
        throw std::runtime_error("Activation layers do not have gradients");
    }

    /**
     * \brief Activation layer info.
     */
    void print() const override;

    /**
     * \brief Getter of input_size class field.
     * \return The size of the layer input.
     */
    [[nodiscard]] virtual SizeType input_size() const override
    {
        return FeedforwardLayer::input_size();
    }

    /**
     * \brief Activation layer setter.
     * Since input_size == output_size in activation layer, it overrides also
     * the output_size.
     * \param input_size The input size to set.
     */
    void input_size(DLMath::Shape3d input_size) override;

private:
};

class ReluLayer : public ActivationLayer
{
public:
    static const std::string TYPE;
    ReluLayer(Model& model, std::string name = std::string(),
              SizeType size = 0);
    [[nodiscard]] inline const std::string& type() const override
    { return TYPE; }
    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override;
    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override;
private:
};

class SoftmaxLayer : public ActivationLayer
{
public:
    static const std::string TYPE;
    SoftmaxLayer(Model& model, std::string name = std::string(),
                 SizeType size = 0);
    [[nodiscard]] inline const std::string& type() const override
    { return TYPE; }
    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override;
    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override;
private:
};

class TanhLayer : public ActivationLayer
{
public:
    static const std::string TYPE;
    TanhLayer(Model& model, std::string name = std::string(),
              SizeType size = 0);
    [[nodiscard]] inline const std::string& type() const override
    { return TYPE; }
    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override;
    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override;
private:
};

class LinearLayer : public ActivationLayer
{
public:
    static const std::string TYPE;
    LinearLayer(Model& model, std::string name = std::string(),
                SizeType size = 0);
    [[nodiscard]] inline const std::string& type() const override
    { return TYPE; }
    const std::vector<NumType>& forward(
        const std::vector<NumType>& inputs) override;
    const std::vector<NumType>& backward(
        const std::vector<NumType>& gradients) override;
private:
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_ACTIVATION_HPP