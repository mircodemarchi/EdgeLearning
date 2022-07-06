/***************************************************************************
 *            middleware/mlpack_fnn.hpp
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

/*! \file  middleware/mlpack_fnn.hpp
 *  \brief MLPACK implementation of Feedforward Neural Network.
 */

#ifndef EDGE_LEARNING_MIDDLEWARE_MLPACK_FNN_HPP
#define EDGE_LEARNING_MIDDLEWARE_MLPACK_FNN_HPP

#include "nn.hpp"
#include "mlpack_definitions.hpp"

#include <map>
#include <tuple>
#include <vector>


namespace EdgeLearning {

template<
    LossType LT = LossType::MSE,
    OptimizerType OT = OptimizerType::GRADIENT_DESCENT,
    InitType IT = InitType::AUTO,
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL,
    typename T = NumType>
class MlpackFNN : public NN<T> {
public:
    MlpackFNN(std::string name)
        : NN<T>{name}
        , _m{}
        , _output_size{0}
        , _layers_name{}
    {

    }

    void fit(Dataset<T>& data,
             SizeType epochs = 1,
             SizeType batch_size = 1,
             NumType learning_rate = 0.03) override
    {
        if (_layers_name.empty())
        {
            throw std::runtime_error(
                "The FNN has no layer: call add before fit");
        }
        ens::GradientDescent o(learning_rate, data.size());
        auto trainset = data.trainset().template to_arma<arma::Mat<T>>();
        auto labels = data.labels().template to_arma<arma::Mat<T>>();

        for (SizeType i = 0; i < epochs; ++i)
        {
            _m.Train(trainset, labels, o);
        }
    }

    Dataset<T> predict(Dataset<T>& data) override
    {
        arma::Mat<T> prediction;
        _m.Predict(data.template to_arma<arma::Mat<T>>(), prediction);
        arma::Mat<T> predition_t = prediction.t();
        predition_t.reshape(1, predition_t.n_rows * predition_t.n_cols);
        auto predition_vec = arma::conv_to<std::vector<T>>::from(predition_t);
        return Dataset<T>(predition_vec, prediction.n_rows);
    }

    void add(LayerDescriptor ld) override
    {
        auto layer_name = std::get<0>(ld);
        auto layer_size = std::get<1>(ld);
        auto layer_activation = std::get<2>(ld);

        _layers_name.push_back(layer_name);
        if (_output_size != 0)
        {
            _m.template Add<mlpack::ann::Linear<>>(_output_size, layer_size);
            switch (layer_activation) {
                case ActivationType::ReLU:
                {
                    _m.template Add<mlpack::ann::ReLULayer<>>();
                    break;
                }
                case ActivationType::ELU:
                {
                    _m.template Add<mlpack::ann::ReLULayer<>>();
                    break;
                }
                case ActivationType::Softmax:
                {
#if __unix__
                    _m.template Add<mlpack::ann::LogSoftMax<>>();
#else
                    _m.template Add<mlpack::ann::Softmax<>>();
#endif
                    break;
                }
                case ActivationType::TanH:
                {
                    _m.template Add<mlpack::ann::TanHLayer<>>();
                    break;
                }
                case ActivationType::Sigmoid:
                {
                    _m.template Add<mlpack::ann::SigmoidLayer<>>();
                    break;
                }
                case ActivationType::Linear:
                default:
                {
                    _m.template Add<mlpack::ann::IdentityLayer<>>();
                    break;
                }
            }
        }
        _output_size = layer_size;
    }

private:
    mlpack::ann::FFN<
        typename MapLoss<Framework::MLPACK, LT>::type,
        typename MapInit<Framework::MLPACK, IT>::type> _m;
    SizeType _output_size;
    std::vector<std::string> _layers_name;
};

template <
    LossType LT,
    OptimizerType OT,
    InitType IT,
    ParallelizationLevel PL,
    typename T>
struct MapModel<Framework::MLPACK, LT, OT, IT, PL, T> {
    using loss_type = typename MapLoss<Framework::MLPACK, LT>::type;
    using optimizer_type = typename MapOptimizer<Framework::MLPACK, OT>::type;
    using init_type = typename MapInit<Framework::MLPACK, IT>::type;
    using type = mlpack::ann::FFN<loss_type, init_type>;
    using fnn = MlpackFNN<LT, OT, IT, PL, T>;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_MIDDLEWARE_MLPACK_FNN_HPP
