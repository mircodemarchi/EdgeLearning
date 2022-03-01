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
#include "dnn/mse_loss.hpp"
#include "dnn/cce_loss.hpp"
#include "dnn/gd_optimizer.hpp"
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

using LayerDesc = std::tuple<std::string, SizeType, Activation>;
using LayerDescVec = std::vector<LayerDesc>;

using ActivationType = Activation;

enum class LossType
{
    CCE,
    MSE,
};

enum class OptimizerType
{
    GRADIENT_DESCENT,
};

enum class InitType
{
    HE_INIT,
    XAVIER_INIT,
    AUTO,
};

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
    virtual void add(LayerDesc ld) = 0;
    virtual Dataset<T> predict(Dataset<T>& data) = 0;
    virtual void fit(Dataset<T>& data,
                     SizeType epochs = 1,
                     SizeType batch_size = 1,
                     NumType learning_rate = 0.03) = 0;

protected:
    std::string _name;
};

template<
    Framework F = Framework::EDGE_LEARNING,
    LossType LT = LossType::MSE,
    OptimizerType OT = OptimizerType::GRADIENT_DESCENT,
    InitType IT = InitType::AUTO,
    typename T = NumType>
class FNN {
public:
    using ModelFNN = typename MapModel<F, LT, OT, IT, T>::fnn;

    FNN(LayerDescVec layers, std::string name)
        : _fnn_model{name}
        , _layers{layers}
    {

    }

    Dataset<T> predict(Dataset<T> &data)
    {
        return _fnn_model.predict(data);
    }

    void fit(Dataset<T> &data,
             SizeType epochs = 1,
             SizeType batch_size = 1,
             NumType learning_rate = 0.03)
    {
        for (const auto& e: _layers)
        {
            _fnn_model.add(e);
        }
        _fnn_model.train(data, epochs, batch_size, learning_rate);
    }

private:
    LayerDescVec _layers;
    ModelFNN _fnn_model;
};

} // namespace EdgeLearning


#endif // EDGE_LEARNING_MIDDLEWARE_NN_HPP
