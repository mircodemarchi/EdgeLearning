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
                SizeType input_size,
                LossType loss = LossType::MSE,
                SizeType batch_size = 1,
                std::string name = std::string());

    template<typename T = double>
    void fit(Dataset<T>& data, SizeType epochs = 1,
             OptimizerType optimizer = OptimizerType::GradientDescent,
             NumType learning_rate = 0.03)
    {
#if ENABLE_MLPACK
#else
        _optimizer = optimizer;
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
                for (SizeType b = 0; b < _batch_size
                     && i < data.size(); ++b, ++i)
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
    std::vector<T> predict(const Dataset<T>& data)
    {
        std::vector<T> ret;
        ret.resize(data.size() * data.feature_size());

        auto output_size = _m.output_size();
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            auto res = _m.predict(
                    data.entry(i).data());
            std::copy(res, res + output_size,
                      ret.begin() + (i * data.feature_size()));
        }
        return ret;
    }

private:
    std::map<std::string, std::tuple<SizeType, Activation>> _layers;
    // SizeType _input_size;
    LossType _loss;
    SizeType _batch_size;
    OptimizerType _optimizer;
    std::string _name;

#if ENABLE_MLPACK
#else
    Model _m;
#endif
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_MIDDLEWARE_FEED_FORWARD_HPP
