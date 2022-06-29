/***************************************************************************
 *            middleware/nn.hpp
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

/*! \file  middleware/nn.hpp
 *  \brief Simply replace me.
 */

#ifndef EDGE_LEARNING_MIDDLEWARE_NN_HPP
#define EDGE_LEARNING_MIDDLEWARE_NN_HPP

#include "dnn/dense.hpp"
#include "dnn/activation.hpp"
#include "dnn/mse_loss.hpp"
#include "dnn/cce_loss.hpp"
#include "dnn/gd_optimizer.hpp"
#include "dnn/adam_optimizer.hpp"
#include "data/dataset.hpp"

#include <utility>
#include <vector>


namespace EdgeLearning {

enum class Framework
{
    EDGE_LEARNING,
#if ENABLE_MLPACK
    MLPACK,
#endif
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

struct LayerShape
{


};


using LayerDescriptor = std::tuple<std::string, SizeType, ActivationType>;
using LayerDescriptorVector = std::vector<LayerDescriptor>;

// template <Framework F> struct MatType;

template <Framework F, ActivationType A> struct MapActivation;

template <Framework F, LossType LT> struct MapLoss;

template <Framework F, OptimizerType OT> struct MapOptimizer;

template <Framework F, InitType IT> struct MapInit;

template <Framework F, LossType LT, OptimizerType OT, InitType IT, typename T>
struct MapModel;

template<typename T = NumType>
class NN {
public:
    NN(std::string name)
        : _name{std::move(name)}
    {
    }
    virtual ~NN() = default;
    virtual void add(LayerDescriptor ld) = 0;
    virtual Dataset<T> predict(Dataset<T>& data) = 0;
    virtual void fit(Dataset<T>& data,
                     SizeType epochs = 1,
                     SizeType batch_size = 1,
                     NumType learning_rate = 0.03) = 0;

protected:
    std::string _name;
};

} // namespace EdgeLearning


#endif // EDGE_LEARNING_MIDDLEWARE_NN_HPP
