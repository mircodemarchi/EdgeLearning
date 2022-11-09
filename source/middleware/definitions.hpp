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

#include "dnn/dense.hpp"
#include "dnn/convolutional.hpp"
#include "dnn/dropout.hpp"
#include "dnn/avg_pooling.hpp"
#include "dnn/max_pooling.hpp"
#include "dnn/activation.hpp"
#include "dnn/mse_loss.hpp"
#include "dnn/cce_loss.hpp"
#include "dnn/gd_optimizer.hpp"
#include "dnn/adam_optimizer.hpp"


namespace EdgeLearning {

enum class Framework
{
    EDGE_LEARNING,
#if ENABLE_MLPACK
    MLPACK,
#endif
};

enum class ParallelizationLevel
{
    SEQUENTIAL,
    THREAD_PARALLELISM_ON_DATA_ENTRY,
    THREAD_PARALLELISM_ON_DATA_BATCH,
};

enum class LayerType
{
    Dense,
    Conv,
    MaxPool,
    AvgPool,
    Dropout,
    Input
};

enum class ActivationType
{
    ReLU,
    ELU,
    Softmax,
    TanH,
    Sigmoid,
    Linear,
    None
};

enum class LossType
{
    CCE,
    MSE,
};

enum class OptimizerType
{
    GRADIENT_DESCENT,
    ADAM
};

enum class InitType
{
    HE_INIT,
    XAVIER_INIT,
    AUTO,
};


// template <Framework F> struct MatType;

template <Framework F, ActivationType A> struct MapActivation;

template <Framework F, LayerType L> struct MapLayer;

template <Framework F, LossType LT> struct MapLoss;

template <Framework F, OptimizerType OT> struct MapOptimizer;

template <Framework F, InitType IT> struct MapInit;

template <
    Framework F,
    LossType LT,
    InitType IT,
    ParallelizationLevel PL,
    typename T>
struct MapModel;

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
struct MapLayer<Framework::EDGE_LEARNING, LayerType::Dense> {
    using type = DenseLayer;
};

template <>
struct MapLayer<Framework::EDGE_LEARNING, LayerType::Conv> {
    using type = ConvolutionalLayer;
};

template <>
struct MapLayer<Framework::EDGE_LEARNING, LayerType::MaxPool> {
    using type = MaxPoolingLayer;
};

template <>
struct MapLayer<Framework::EDGE_LEARNING, LayerType::AvgPool> {
    using type = AveragePoolingLayer;
};

template <>
struct MapLayer<Framework::EDGE_LEARNING, LayerType::Dropout> {
    using type = DropoutLayer;
};

template <>
struct MapLoss<Framework::EDGE_LEARNING, LossType::CCE> {
    using type = CategoricalCrossEntropyLossLayer;
    inline static const std::string name = "cce_loss";
};

template <>
struct MapLoss<Framework::EDGE_LEARNING, LossType::MSE> {
    using type = MeanSquaredLossLayer;
    inline static const std::string name = "mse_loss";
};

template <>
struct MapOptimizer<Framework::EDGE_LEARNING, OptimizerType::GRADIENT_DESCENT> {
    using type = GradientDescentOptimizer;
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
