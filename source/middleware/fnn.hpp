/***************************************************************************
 *            middleware/fnn.hpp
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

/*! \file  middleware/fnn.hpp
 *  \brief Feedforward Neural Network.
 */

#ifndef EDGE_LEARNING_MIDDLEWARE_FNN_HPP
#define EDGE_LEARNING_MIDDLEWARE_FNN_HPP

#include "definitions.hpp"
#if ENABLE_MLPACK
#include "mlpack_fnn.hpp"
#endif

#include <map>
#include <tuple>
#include <utility>
#include <vector>


namespace EdgeLearning {

template<
    LossType LT = LossType::MSE,
    OptimizerType OT = OptimizerType::GRADIENT_DESCENT,
    InitType IT = InitType::AUTO,
    typename T = NumType>
class EdgeFNN : public NN<T> {
public:
    EdgeFNN(std::string name)
        : NN<T>(name)
        , _m(NN<T>::_name)
        , _output_size{0}
    {

    }

    void add(LayerDescriptor ld) override
    {
        auto layer_name = std::get<0>(ld);
        auto layer_size = std::get<1>(ld);
        auto layer_activation = std::get<2>(ld);

        if (_output_size != 0)
        {
            Layer::SharedPtr prev_layer;
            if (!_m.layers().empty())
            {
                prev_layer = _m.layers().back();
            }
            auto layer = _m.template add_layer<DenseLayer>(
                layer_name, layer_activation,
                layer_size, _output_size);
            if (prev_layer)
            {
                _m.create_edge(prev_layer, layer);
            }
        }
        _output_size = layer_size;
    }

    void fit(Dataset<T> &data,
             SizeType epochs = 1,
             SizeType batch_size = 1,
             NumType learning_rate = 0.03) override
    {
        // Add Loss layer.
        if (_m.layers().empty())
        {
            throw std::runtime_error(
                "The FNN has no layer: call add before fit");
        }
        auto prev_layer = _m.layers().back();
        auto loss_layer_name = MapLoss<Framework::EDGE_LEARNING, LT>::name;
        auto loss_layer = _m.template add_loss<
            typename MapLoss<Framework::EDGE_LEARNING, LT>::type>(
                loss_layer_name, prev_layer->output_size(), batch_size);
        _m.create_back_arc(prev_layer, loss_layer);

        // Train.
        using optimizer_type = typename MapOptimizer<
            Framework::EDGE_LEARNING, OT>::type;
        auto o = optimizer_type(learning_rate);
        for (SizeType e = 0; e < epochs; ++e)
        {
            for (SizeType i = 0; i < data.size();)
            {
                for (SizeType b = 0; b < batch_size
                                     && i < data.size(); ++b, ++i)
                {
                    _m.step(data.trainset(i).data(),
                           data.labels(i).data());
                }
                _m.train(o);
            }
        }
    }

    Dataset<T> predict(Dataset<T> &data) override
    {
        std::vector<T> ret;
        ret.resize(data.size() * data.feature_size());

        auto output_size = _m.output_size();
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            auto res = _m.predict(data.entry(i).data());
            std::copy(res, res + output_size,
                      ret.begin() + long(i * data.feature_size()));
        }
        return Dataset<T>(ret, data.feature_size());
    }

private:
    Model _m;
    SizeType _output_size;
};

template <LossType LT, OptimizerType OT, InitType IT, typename T>
struct MapModel<Framework::EDGE_LEARNING, LT, OT, IT, T> {
    using loss_type = typename MapLoss<Framework::EDGE_LEARNING, LT>::type;
    using optimizer_type = typename MapOptimizer<
        Framework::EDGE_LEARNING, OT>::type;
    static const InitType init_type = MapInit<
        Framework::EDGE_LEARNING, IT>::type;
    using type = Model;
    using fnn = EdgeFNN<LT, OT, IT, T>;
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

    FNN(LayerDescriptorVector layers, std::string name)
        : _layers{std::move(layers)}
        , _fnn_model{name}
    {
        for (const auto& e: _layers)
        {
            _fnn_model.add(e);
        }
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
        _fnn_model.fit(data, epochs, batch_size, learning_rate);
    }

private:
    LayerDescriptorVector _layers;
    ModelFNN _fnn_model;
};

#if ENABLE_MLPACK
template<
    LossType LT = LossType::MSE,
    OptimizerType OT = OptimizerType::GRADIENT_DESCENT,
    InitType IT = InitType::AUTO,
    typename T = NumType>
using CompileFNN = FNN<Framework::MLPACK, LT, OT, IT, T>;
#else
template<
    LossType LT = LossType::MSE,
    OptimizerType OT = OptimizerType::GRADIENT_DESCENT,
    InitType IT = InitType::AUTO,
    typename T = NumType>
using CompileFNN = FNN<Framework::EDGE_LEARNING, LT, OT, IT, T>;
#endif

} // namespace EdgeLearning

#endif // EDGE_LEARNING_MIDDLEWARE_FNN_HPP
