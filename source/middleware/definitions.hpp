/***************************************************************************
 *            middleware/definitions.hpp
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

/*! \file  middleware/definitions.hpp
 *  \brief Simply replace me.
 */

#ifndef EDGE_LEARNING_MIDDLEWARE_DEFINITIONS_HPP
#define EDGE_LEARNING_MIDDLEWARE_DEFINITIONS_HPP

#include "nn.hpp"


namespace EdgeLearning {

// template <Framework F>
// struct MapType
// {
//     static ActivationType activation(ActivationType activation_type)
//     {
//         return activation_type;
//     }
// };


template <>
struct MapActivation<Framework::EDGE_LEARNING, ActivationType::ReLU> {
    using type = ReluLayer;
};

template <>
struct MapActivation<Framework::EDGE_LEARNING, ActivationType::ELU> {
    using type = EluLayer;
};

template <>
struct MapActivation<Framework::EDGE_LEARNING, ActivationType::Softmax> {
    using type = SoftmaxLayer;
};

template <>
struct MapActivation<Framework::EDGE_LEARNING, ActivationType::TanH> {
    using type = TanhLayer;
};

template <>
struct MapActivation<Framework::EDGE_LEARNING, ActivationType::Sigmoid> {
    using type = SigmoidLayer;
};

template <>
struct MapActivation<Framework::EDGE_LEARNING, ActivationType::Linear> {
    using type = LinearLayer;
};

template <>
struct MapLoss<Framework::EDGE_LEARNING, LossType::CCE> {
    using type = CCELossLayer;
    inline static const std::string name = "cce_loss";
};

template <>
struct MapLoss<Framework::EDGE_LEARNING, LossType::MSE> {
    using type = MSELossLayer;
    inline static const std::string name = "mse_loss";
};

template <>
struct MapOptimizer<Framework::EDGE_LEARNING, OptimizerType::GRADIENT_DESCENT> {
    using type = GDOptimizer;
};

template <>
struct MapOptimizer<Framework::EDGE_LEARNING, OptimizerType::ADAM> {
    using type = AdamOptimizer;
};

template <>
struct MapInit<Framework::EDGE_LEARNING, InitType::HE_INIT> {
    static const Model::InitializationFunction type
        = Model::InitializationFunction::KAIMING;
};

template <>
struct MapInit<Framework::EDGE_LEARNING, InitType::XAVIER_INIT> {
    static const Model::InitializationFunction type
        = Model::InitializationFunction::XAVIER;
};

template <>
struct MapInit<Framework::EDGE_LEARNING, InitType::AUTO> {
    static const Model::InitializationFunction type
        = Model::InitializationFunction::AUTO;
};

} // namespace EdgeLearning


#endif // EDGE_LEARNING_MIDDLEWARE_DEFINITIONS_HPP
