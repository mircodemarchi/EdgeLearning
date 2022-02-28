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

#ifndef EDGE_LEARNING_MIDDLEWARE_FNN_HPP
#define EDGE_LEARNING_MIDDLEWARE_FNN_HPP

#include "middleware/type.hpp"

#include <map>
#include <tuple>


namespace EdgeLearning {

template<LossType LT = LossType::MSE, typename T = NumType>
class FFNN {
public:
    using LayerDesc = std::tuple<std::string, SizeType, Activation>;
    using LayerDescVec = std::vector<LayerDesc>;

    FFNN(LayerDescVec layers, std::string name)
        : _layers{std::move(layers)}
        , _name{name}
    {
#if ENABLE_MLPACK

#else

#endif
    }

    template<OptimizerType OT = OptimizerType::GradientDescent>
    ModelType<LT> fit(Dataset<T>& data,
                      SizeType epochs = 1,
                      SizeType batch_size = 1,
                      NumType learning_rate = 0.03)
    {
        auto prev_layer_size = data.trainset_idx().size();
#if ENABLE_MLPACK
        ModelType<LT> m;
        for (const auto& e: _layers)
        {
            auto curr_layer_name = std::get<0>(e);
            auto curr_layer_size = std::get<1>(e);
            auto curr_layer_activation = std::get<2>(e);
            m.template Add<mlpack::ann::Linear<>>(
                prev_layer_size, curr_layer_size);
            switch (curr_layer_activation) {
                case Activation::ReLU:
                {
                    m.template Add<mlpack::ann::ReLULayer<>>();
                    break;
                }
                case Activation::Softmax:
                {
                    m.template Add<mlpack::ann::Softmax<>>();
                    break;
                }
                case Activation::Linear:
                default:
                {
                    m.template Add<mlpack::ann::IdentityLayer<>>();
                    break;
                }
            }
            prev_layer_size = curr_layer_size;
        }

        ens::GradientDescent o(0.03, data.size());
        auto dataset = data.template to_arma<arma::Mat<T>>();
        arma::mat trainLabels = dataset.row(dataset.n_rows - 1);
        dataset.shed_row(dataset.n_rows - 1);
        m.template Train(dataset, trainLabels, o);
#else
        ModelType m(_name);
        std::vector<Layer::SharedPtr> l;
        for (const auto& e: _layers)
        {
            auto curr_layer_name = std::get<0>(e);
            auto curr_layer_size = std::get<1>(e);
            auto curr_layer_activation = std::get<2>(e);
            l.push_back(
                m.add_layer<DenseLayer>(
                    curr_layer_name, curr_layer_activation,
                    curr_layer_size, prev_layer_size)
            );
            prev_layer_size = curr_layer_size;
        }

        auto output_size = prev_layer_size;
        auto loss_layer_name = MapLoss<LT>::name;
        auto loss_layer = m.add_loss<MapLoss<LT>::type>(
            loss_layer_name, output_size, batch_size);

        for (SizeType i = 0; i < l.size() - 1; ++i)
        {
            m.create_edge(l[i], l[i + 1]);
        }
        m.create_back_arc(l[l.size() - 1], loss_layer);

        auto o = MapOptimizer<OT>::type(learning_rate);
        for (SizeType e = 0; e < epochs; ++e)
        {
            for (SizeType i = 0; i < data.size();)
            {
                for (SizeType b = 0; b < batch_size
                     && i < data.size(); ++b, ++i)
                {
                    m.step(data.trainset(i).data(),
                           data.labels(i).data());
                }
                m.train(o);
            }
        }
        return m;
#endif
    }

    Dataset<T> predict(Dataset<T>& data, ModelType<LT>& m)
    {
#if ENABLE_MLPACK
        return  Dataset<T>(
            std::vector<T>(data.size() * data.feature_size()),
            data.feature_size());
#else
        std::vector<T> ret;
        ret.resize(data.size() * data.feature_size());

        auto output_size = m.output_size();
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            auto res = m.predict(
                    data.entry(i).data());
            std::copy(res, res + output_size,
                      ret.begin() + long(i * data.feature_size()));
        }
        return Dataset<T>(ret, data.feature_size());
#endif
    }

private:
    LayerDescVec _layers;
    std::string _name;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_MIDDLEWARE_FNN_HPP
