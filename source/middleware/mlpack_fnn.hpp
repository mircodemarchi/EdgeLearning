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

#ifndef EDGE_LEARNING_MIDDLEWARE_FNN_HPP
#define EDGE_LEARNING_MIDDLEWARE_FNN_HPP

#include "nn.hpp"
#include "mlpack_type.hpp"

#include <map>
#include <tuple>
#include <vector>


namespace EdgeLearning {

template<
    LossType LT = LossType::MSE,
    OptimizerType OT = OptimizerType::GRADIENT_DESCENT,
    InitType IT = InitType::AUTO,
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
        ens::GradientDescent o(learning_rate, data.size());
        auto dataset = data.template to_arma<arma::Mat<T>>();
        arma::mat trainLabels = dataset.row(dataset.n_rows - 1);
        dataset.shed_row(dataset.n_rows - 1);
        for (SizeType i = 0; i < epochs; ++i)
        {
            _m.template Train(dataset, trainLabels, o);
        }
    }

    Dataset<T> predict(Dataset<T>& data)
    {
        return Dataset<T>(
            std::vector<T>(data.size() * data.feature_size()),
            data.feature_size());
    }

    void add(LayerDesc ld) override
    {
        auto layer_name = std::get<0>(ld);
        auto layer_size = std::get<1>(ld);
        auto layer_activation = std::get<2>(ld);

        _layers_name.push_back(layer_name);
        if (_output_size != 0)
        {
            _m.template Add(mlpack::ann::Linear<>(_output_size, layer_size));
            switch (layer_activation) {
                case Activation::ReLU:
                {
                    _m.template Add<mlpack::ann::ReLULayer<>>();
                    break;
                }
                case Activation::Softmax:
                {
                    _m.template Add<mlpack::ann::Softmax<>>();
                    break;
                }
                case Activation::Linear:
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
    MlpackFNN<LT, OT, IT, T> _m;
    SizeType _output_size;
    std::vector<std::string> _layers_name;
};

template <LossType LT, OptimizerType OT, InitType IT, typename T>
struct MapModel<Framework::MLPACK, LT, OT, IT, T> {
    using loss_type = typename MapLoss<Framework::MLPACK, LT>::type;
    using optimizer_type = typename MapOptimizer<Framework::MLPACK, OT>::type;
    using init_type = typename MapInit<Framework::MLPACK, IT>::type;
    using type = mlpack::ann::FFN<loss_type, init_type>;
    using fnn = MlpackFNN<LT, OT, IT, T>;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_MIDDLEWARE_FNN_HPP