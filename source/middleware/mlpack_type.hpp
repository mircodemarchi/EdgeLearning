/***************************************************************************
 *            middleware/mlpack_type.hpp
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

/*! \file  middleware/mlpack_type.hpp
 *  \brief Simply replace me.
 */

#ifndef EDGE_LEARNING_MIDDLEWARE_TYPE_HPP
#define EDGE_LEARNING_MIDDLEWARE_TYPE_HPP

#include "nn.hpp"

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/init_rules/lecun_normal_init.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <ensmallen.hpp>


namespace EdgeLearning {

template <>
struct MapActivation<Framework::MLPACK, ActivationType::ReLU> {
    using type = mlpack::ann::ReLULayer<>;
};

template <>
struct MapActivation<Framework::MLPACK, ActivationType::Softmax> {
    using type = mlpack::ann::Softmax<>;
};

template <>
struct MapActivation<Framework::MLPACK, ActivationType::Linear> {
    using type = mlpack::ann::IdentityLayer<>;
};

template <>
struct MapLoss<Framework::MLPACK, LossType::CCE> {
    using type = mlpack::ann::CrossEntropyError<>;
    static constexpr std::string_view name = "cce_loss";
};

template <>
struct MapLoss<Framework::MLPACK, LossType::MSE> {
    using type = mlpack::ann::MeanSquaredError<>;
    static constexpr std::string_view name = "mse_loss";
};

template <>
struct MapOptimizer<Framework::MLPACK, OptimizerType::GRADIENT_DESCENT> {
    using type = ens::GradientDescent;
};

template <>
struct MapInit<Framework::MLPACK, InitType::HE_INIT> {
    using type = mlpack::ann::LecunNormalInitialization;
};

template <>
struct MapInit<Framework::MLPACK, InitType::XAVIER_INIT> {
    using type = mlpack::ann::LecunNormalInitialization;
};

template <>
struct MapInit<Framework::EDGE_LEARNING, InitType::AUTO> {
    using type = mlpack::ann::LecunNormalInitialization;
};

} // namespace EdgeLearning


#endif // EDGE_LEARNING_MIDDLEWARE_TYPE_HPP
