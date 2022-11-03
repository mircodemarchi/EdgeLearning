/***************************************************************************
 *            middleware/layer_descriptor.hpp
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

/*! \file  middleware/layer_descriptor.hpp
 *  \brief High level interface layer descriptor.
 */


#ifndef EDGELEARNING_LAYER_DESCRIPTOR_HPP
#define EDGELEARNING_LAYER_DESCRIPTOR_HPP


#include "definitions.hpp"


namespace EdgeLearning {

/**
 * \brief Layer settings.
 * Class that group all the attributes that can be used to configure every
 * layer type.
 */
class LayerSetting {
public:
    /**
     * \brief Constructor for Input layer setting and Dense layer setting.
     * \param units LayerShape Number of units of the layer.
     */
    LayerSetting(LayerShape units);

    /**
     * \brief Constructor for Convolutional layer setting.
     * \param n_filters    SizeType Number of filters.
     * \param kernel_shape DLMath::Shape2d Kernel shape.
     * \param stride       DLMath::Shape2d Stride shape.
     * \param padding      DLMath::Shape2d Padding shape.
     */
    LayerSetting(SizeType n_filters,
                 DLMath::Shape2d kernel_shape,
                 DLMath::Shape2d stride,
                 DLMath::Shape2d padding);

    /**
     * \brief Constructor for Pooling layer setting.
     * \param kernel_shape DLMath::Shape2d Kernel shape.
     * \param stride       DLMath::Shape2d Stride shape.
     */
    LayerSetting(DLMath::Shape2d kernel_shape,
                 DLMath::Shape2d stride);

    /**
     * \brief Constructor for Dropout layer setting.
     * \param drop_probability NumType The dropout probability.
     */
    LayerSetting(NumType drop_probability);

    /**
     * \brief Default constructor.
     */
    LayerSetting();

    /**
     * \brief Getter of the number of units of the layer setting.
     * \return LayerShape Number of units.
     * Always 0 if the layer setting is not an Input or Dense layer.
     */
    LayerShape units() const;

    /**
     * \brief Setter of the number of units of the layer setting.
     * \param units LayerShape Number of units.
     */
    void units(LayerShape units);

    /**
     * \brief Getter of the number of filters of the layer setting.
     * \return SizeType Number of filters.
     * Always 0 if the layer setting is not a Convolutional layer.
     */
    SizeType n_filters() const;

    /**
     * \brief Setter of the number of filters of the layer setting.
     * \param n_filters SizeType Number of filters.
     */
    void n_filters(SizeType n_filters);

    /**
     * \brief Getter of the kernel shape of the layer setting.
     * \return const DLMath::Shape2d& The kernel shape.
     * Always 0 if the layer setting is not a Convolutional or Pooling layer.
     */
    const DLMath::Shape2d& kernel_shape() const;

    /**
     * \brief Setter of the kernel shape of the layer setting.
     * \param kernel_shape const DLMath::Shape2d& The kernel shape.
     */
    void kernel_shape(const DLMath::Shape2d& kernel_shape);

    /**
     * \brief Getter of the stride shape of the layer setting.
     * \return const DLMath::Shape2d& The stride shape.
     * Always 0 if the layer setting is not a Convolutional or Pooling layer.
     */
    const DLMath::Shape2d& stride() const;

    /**
     * \brief Setter of the stride shape of the layer setting.
     * \param stride const DLMath::Shape2d& The stride shape.
     */
    void stride(const DLMath::Shape2d& stride);

    /**
     * \brief Getter of the padding shape of the layer setting.
     * \return const DLMath::Shape2d& The padding shape.
     * Always 0 if the layer setting is not a Convolutional layer.
     */
    const DLMath::Shape2d& padding() const;

    /**
     * \brief Setter of the padding shape of the layer setting.
     * \param padding const DLMath::Shape2d& The padding shape.
     */
    void padding(const DLMath::Shape2d& padding);

    /**
     * \brief Getter of the dropout probability of the layer setting.
     * \return NumType The padding shape.
     * Always 0 if the layer setting is not a Dropout layer.
     */
    NumType drop_probability() const;

    /**
     * \brief Setter of the dropout probability of the layer setting.
     * \param drop_probability NumType The dropout probability.
     */
    void drop_probability(NumType drop_probability);

private:
    /**
     * \brief The hidden units for Dense layer and the input shape for
     * Input layer. Used only by Input and Dense layers.
     */
    LayerShape _units;

    /**
     * \brief The number of filters. Used only by Convolutional layer.
     */
    SizeType _n_filters;

    /**
     * \brief The shape of the kernel. Used only by Convolutional and Pooling
     * layers.
     */
    DLMath::Shape2d _kernel_shape;

    /**
     * \brief The shape of the stride. Used only by Convolutional and Pooling
     * layers.
     */
    DLMath::Shape2d _stride;

    /**
     * \brief The shape of the padding. Used only by Convolutional layer.
     */
    DLMath::Shape2d _padding;

    /**
     * \brief The dropout probability. Used only by Dropout layer.
     */
    NumType _drop_probability;
};

/**
 * \brief Minimal layer representation class.
 * If the layer type is LayerType::Input, then the layer shape and the
 * activation types are ignored.
 */
class LayerDescriptor {
public:

    /**
     * \brief Main constructor of the layer descriptor.
     * \param name            std::string The name of the layer.
     * \param type            LayerType The type of the layer.
     * \param setting         LayerShape The shape of the layer. default 0.
     * \param activation_type ActivationType The activation type of the layer.
     * Default activation type is linear (i.e. identity).
     */
    LayerDescriptor(
        std::string name,
        LayerType type,
        LayerSetting setting = {},
        ActivationType activation_type = ActivationType::Linear);

    /**
     * \brief Getter of the layer name.
     * \return const std::string& The layer name.
     */
    const std::string& name() const;

    /**
     * \brief Setter of the layer name.
     * \param name const std::string& The new layer name.
     */
    void name(const std::string& name);

    /**
     * \brief Getter of the layer type.
     * \return LayerType The layer type.
     */
    LayerType type() const;

    /**
     * \brief Setter of the layer type.
     * \param type LayerType The new layer type.
     */
    void type(LayerType type);

    /**
     * \brief Getter of the layer setting.
     * \return const LayerSetting& The layer setting.
     */
    const LayerSetting& setting() const;

    /**
     * \brief Setter of the layer setting.
     * \param setting const LayerSetting& The new layer setting.
     */
    void setting(const LayerSetting& setting);

    /**
     * \brief Getter of the layer activation type.
     * \return ActivationType The layer activation type.
     */
    ActivationType activation_type() const;

    /**
     * \brief Setter of the layer activation type.
     * \param activation_type ActivationType The new layer activation type.
     */
    void activation_type(ActivationType activation_type);

private:
    std::string _name;               ///< \brief Layer name.
    LayerType _type;                 ///< \brief Layer type.
    LayerSetting _setting;           ///< \brief Layer setting.
    ActivationType _activation_type; ///< \brief Layer activation type.
};

/**
 * \brief Input layer descriptor.
 */
class Input : public LayerDescriptor {
public:
    /**
     * \brief Constructor of Input layer descriptor.
     * \param name       std::string The name of the layer.
     * \param input_size LayerShape  The input shape of the layer.
     */
    Input(std::string name, LayerShape input_size);
};

/**
 * \brief Dense layer descriptor.
 */
class Dense : public LayerDescriptor {
public:
    /**
     * \brief Constructor of Dense layer descriptor.
     * \param name            std::string The name of the layer.
     * \param hidden_nodes    SizeType The hidden units of the layer.
     * \param activation_type ActivationType The activation type of the layer.
     */
    Dense(std::string name, SizeType hidden_nodes,
          ActivationType activation_type = ActivationType::Linear);
};

/**
 * \brief Convolutional layer descriptor.
 */
class Conv : public LayerDescriptor {
public:

    /**
     * \brief Convolutional layer settings.
     */
    struct ConvSetting {
        /**
         * \brief Constructor of Convolutional layer settings.
         * \param nf SizeType The number of convolutional filters.
         * \param ks DLMath::Shape2d The convolutional kernel shape.
         * \param s  DLMath::Shape2d The convolutional stride shape.
         * \param p  DLMath::Shape2d The convolutional padding shape.
         */
        ConvSetting(SizeType nf,
                    DLMath::Shape2d ks,
                    DLMath::Shape2d s = {1},
                    DLMath::Shape2d p = {0})
            : n_filters(nf)
            , kernel_shape(ks)
            , stride(s)
            , padding(p)
        { }

        SizeType n_filters;           ///< \brief The number of filters.
        DLMath::Shape2d kernel_shape; ///< \brief The kernel shape.
        DLMath::Shape2d stride;       ///< \brief The stride shape.
        DLMath::Shape2d padding;      ///< \brief The padding shape.
    };

    /**
     * \brief Constructor of Convolutional layer descriptor.
     * \param name            std::string The layer name.
     * \param setting         ConvSetting The Convolutional settings.
     * \param activation_type ActivationType The layer activation type.
     */
    Conv(std::string name, ConvSetting setting,
          ActivationType activation_type = ActivationType::Linear);
};

/**
 * \brief Max Pooling layer descriptor.
 */
class MaxPool : public LayerDescriptor {
public:

    /**
     * \brief Max Pooling layer settings.
     */
    struct MaxPoolSetting {

        /**
         * \brief Constructor of Max Pooling layer settings.
         * \param ks DLMath::Shape2d The max pooling kernel shape.
         * \param s  DLMath::Shape2d The max pooling stride shape.
         */
        MaxPoolSetting(DLMath::Shape2d ks, DLMath::Shape2d s = {1})
            : kernel_shape(ks)
            , stride(s)
        { }

        DLMath::Shape2d kernel_shape; ///< \brief The kernel shape.
        DLMath::Shape2d stride;       ///< \brief The stride shape.
    };

    /**
     * \brief Constructor of Max Pooling layer descriptor.
     * \param name            std::string The layer name.
     * \param setting         MaxPoolSetting The Max Pooling settings.
     * \param activation_type ActivationType The layer activation type.
     */
    MaxPool(std::string name, MaxPoolSetting setting,
            ActivationType activation_type = ActivationType::Linear);
};

/**
 * \brief Average Pooling layer descriptor.
 */
class AvgPool : public LayerDescriptor {
public:

    /**
     * \brief Average Pooling layer settings.
     */
    struct AvgPoolSetting {

        /**
         * \brief Constructor of Average Pooling layer settings.
         * \param ks DLMath::Shape2d The average pooling kernel shape.
         * \param s  DLMath::Shape2d The average pooling stride shape.
         */
        AvgPoolSetting(DLMath::Shape2d ks, DLMath::Shape2d s = {1})
            : kernel_shape(ks)
            , stride(s)
        { }

        DLMath::Shape2d kernel_shape; ///< \brief The kernel shape.
        DLMath::Shape2d stride;       ///< \brief The stride shape.
    };

    /**
     * \brief Constructor of Average Pooling layer descriptor.
     * \param name            std::string The layer name.
     * \param setting         AvgPoolSetting The Avg Pooling settings.
     * \param activation_type ActivationType The layer activation type.
     */
    AvgPool(std::string name, AvgPoolSetting setting,
            ActivationType activation_type = ActivationType::Linear);
};

/**
 * \brief Dropout layer descriptor.
 */
class Dropout : public LayerDescriptor {
public:

    /**
     * \brief Constructor of Dropout layer descriptor.
     * \param name              std::string The layer name.
     * \param drop_probability  NumType The drop probability.
     * \param activation_type   ActivationType The layer activation type.
     */
    Dropout(std::string name, NumType drop_probability,
            ActivationType activation_type = ActivationType::Linear);
};

} // namespace EdgeLearning

#endif //EDGELEARNING_LAYER_DESCRIPTOR_HPP
