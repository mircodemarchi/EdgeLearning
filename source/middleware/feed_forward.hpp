/***************************************************************************
 *            time_estimator.hpp
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

/*! \file time_estimator.hpp
 *  \brief Task execution time estimator model.
 */

#ifndef EDGE_LEARNING_MIDDLEWARE_FEED_FORWARD_HPP
#define EDGE_LEARNING_MIDDLEWARE_FEED_FORWARD_HPP

#include "dnn/dense.hpp"
#include "dnn/mse_loss.hpp"
#include "dnn/cce_loss.hpp"
#include "dnn/gd_optimizer.hpp"
#include "middleware/dataset.hpp"

#include <map>
#include <tuple>


namespace EdgeLearning {

enum class LossType
{
    CCE,
    MSE
};

enum class OptimizerType
{
    GradientDescent
};

class FeedForward {
public:
    FeedForward(std::map<std::string, std::tuple<SizeType, Activation>> layers,
                LossType loss = LossType::MSE,
                OptimizerType optimizer = OptimizerType::GradientDescent,
                std::string name = std::string());

    template<typename T = double>
    void fit(Dataset<T>& data, SizeType epochs = 1,
             NumType learning_rate = 0.03, SizeType batch_size = 1)
    {
        auto input_size = data.feature_size();
#if ENABLE_MLPACK
#else
        std::vector<Layer::SharedPtr> l;
        auto prev_layer_size = input_size;
        for (const auto& e: _layers)
        {
            auto curr_layer_size = std::get<0>(e.second);
            auto curr_layer_activation = std::get<1>(e.second);
            l.push_back(
                _m.add_layer<DenseLayer>(
                    e.first, curr_layer_activation,
                    curr_layer_size, prev_layer_size)
            );
            prev_layer_size = curr_layer_size;
        }

        std::shared_ptr<LossLayer> loss_layer;
        auto output_size = prev_layer_size;
        switch(_loss)
        {
            case LossType::CCE: {
                loss_layer = _m.add_loss<CCELossLayer>(
                    "cce_loss", output_size, batch_size);
                break;
            }
            case LossType::MSE:
            default: {
                loss_layer = _m.add_loss<MSELossLayer>(
                    "mse_loss", output_size, batch_size);
                break;
            }
        }

        for (SizeType i = 0; i < l.size() - 1; ++i)
        {
            _m.create_edge(l[i], l[i + 1]);
        }
        _m.create_edge(l[l.size() - 1], loss_layer);

        std::shared_ptr<Optimizer> o;
        switch(_optimizer)
        {
            case OptimizerType::GradientDescent:
            default: {
                o = std::make_shared<GDOptimizer>(learning_rate);
                break;
            }
        }

        for (SizeType e = 0; e < epochs; ++e)
        {
            for (SizeType i = 0; i < data.size();)
            {
                for (SizeType b = 0; b < batch_size && i < data.size(); ++b, ++i)
                {
                    _m.step(data.trainset(i).data(),
                            data.labels(i).data());
                }
                _m.train(*o);
            }
        }
#endif
    }

    template<typename T = double>
    std::vector<T> predict(Dataset<T>& data)
    {

    }

private:
    std::map<std::string, std::tuple<SizeType, Activation>> _layers;
    LossType _loss;
    OptimizerType _optimizer;
    std::string _name;

#if ENABLE_MLPACK
#else
    Model _m;
#endif
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_MIDDLEWARE_FEED_FORWARD_HPP
