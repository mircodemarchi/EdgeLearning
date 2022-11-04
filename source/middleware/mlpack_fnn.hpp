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

#include "mlpack_definitions.hpp"
#include "layer_descriptor.hpp"

#include <map>
#include <tuple>
#include <vector>


namespace EdgeLearning {

template<
    LossType LT = LossType::MSE,
    InitType IT = InitType::AUTO,
    typename T = NumType>
class MlpackFNN : public StaticNeuralNetwork<T> {
public:
    MlpackFNN(std::string name)
        : StaticNeuralNetwork<T>{name}
        , _m{}
        , _input_shape{0}
        , _output_shape{0}
        , _layers_name{}
        , _is_first_add{true}
    {

    }

    void fit(Dataset<T>& data,
             OptimizerType optimizer = OptimizerType::ADAM,
             SizeType epochs = 1,
             SizeType batch_size = 1,
             NumType learning_rate = 0.03) override
    {
        if (_layers_name.empty())
        {
            throw std::runtime_error(
                "The FNN has no layer: call add before fit");
        }
        auto trainset = data.trainset().template to_arma<arma::Mat<T>>();
        auto labels = data.labels().template to_arma<arma::Mat<T>>();
        _fit(optimizer, trainset, labels, epochs, batch_size, learning_rate);
    }

    Dataset<T> predict(Dataset<T>& data) override
    {
        arma::Mat<T> prediction;
        _m.Predict(data.template to_arma<arma::Mat<T>>(), prediction);
        auto prediction_vec = arma::conv_to<std::vector<T>>::from(
            prediction.as_col());
        return Dataset<T>(prediction_vec, prediction.n_rows);
    }

    void add(LayerDescriptor ld) override
    {
        if (_is_first_add)
        {
            _is_first_add = false;
            if (ld.type() == LayerType::Input)
            {
                _input_shape = ld.setting().units();
                _output_shape = ld.setting().units();
                return;
            }
        }

        _layers_name.push_back(ld.name());
        _output_shape = _add_layer(ld, _output_shape);
        _add_activation_layer(ld);
    }

    SizeType input_size() override { return _input_shape.size(); }
    SizeType output_size() override { return _output_shape.size(); }

private:
    void _fit(OptimizerType optimizer,
              const arma::Mat<T> trainset, const arma::Mat<T> labels,
              SizeType epochs, SizeType batch_size, NumType learning_rate)
    {
        ens::StoreBestCoordinates<arma::mat> bestCoordinates;
        switch (optimizer) {
            case OptimizerType::GRADIENT_DESCENT:
            {
                using optimizer_type = typename MapOptimizer<
                    Framework::MLPACK, OptimizerType::GRADIENT_DESCENT>::type;
                auto o = optimizer_type(learning_rate, batch_size,
                                        epochs * trainset.n_cols, 0.0, false);
                _m.Train(trainset, labels, o,
                         ens::PrintLoss(), ens::ProgressBar(), bestCoordinates);
                break;
            }
            case OptimizerType::ADAM:
            default:
            {
                using optimizer_type = typename MapOptimizer<
                    Framework::MLPACK, OptimizerType::ADAM>::type;
                auto o = optimizer_type(learning_rate, batch_size,
                                        0.9, 0.999, 1e-8,
                                        epochs * trainset.n_cols, 0.0, false);
                _m.Train(trainset, labels, o,
                         ens::PrintLoss(), ens::ProgressBar(), bestCoordinates);
                break;
            }
        }
        _m.Parameters() = bestCoordinates.BestCoordinates();
    }

    LayerShape _add_layer(const LayerDescriptor& ld, const LayerShape& input_shape)
    {
        switch (ld.type())
        {
            case LayerType::Input:
            {
                throw std::runtime_error("Model structure error: "
                                         "Input layer have to be "
                                         "put as first layer");
            }
            case LayerType::Conv:
            {
                using layer_type = typename MapLayer<
                    Framework::MLPACK, LayerType::Conv>::type;
                auto layer = new layer_type(
                    input_shape.channels(),
                    ld.setting().n_filters(),
                    ld.setting().kernel_shape().width(),
                    ld.setting().kernel_shape().height(),
                    ld.setting().stride().width(),
                    ld.setting().stride().height(),
                    ld.setting().padding().width(),
                    ld.setting().padding().height(),
                    input_shape.width(),
                    input_shape.height());
                _m.Add(layer);
                return ConvolutionalLayer::calculate_output_shape(
                    input_shape.shape(),
                    ld.setting().kernel_shape(),
                    ld.setting().stride(),
                    ld.setting().padding(),
                    ld.setting().n_filters());
            }
            case LayerType::MaxPool:
            {
                using layer_type = typename MapLayer<
                    Framework::MLPACK, LayerType::MaxPool>::type;
                auto layer = new layer_type(
                    ld.setting().kernel_shape().width(),
                    ld.setting().kernel_shape().height(),
                    ld.setting().stride().width(),
                    ld.setting().stride().height());
                _m.Add(layer);
                return MaxPoolingLayer::calculate_output_shape(
                    input_shape.shape(),
                    ld.setting().kernel_shape(),
                    ld.setting().stride());
            }
            case LayerType::AvgPool:
            {
                using layer_type = typename MapLayer<
                    Framework::MLPACK, LayerType::AvgPool>::type;
                auto layer = new layer_type(
                    ld.setting().kernel_shape().width(),
                    ld.setting().kernel_shape().height(),
                    ld.setting().stride().width(),
                    ld.setting().stride().height());
                _m.Add(layer);
                return AvgPoolingLayer::calculate_output_shape(
                    input_shape.shape(),
                    ld.setting().kernel_shape(),
                    ld.setting().stride());
            }
            case LayerType::Dropout:
            {
                using layer_type = typename MapLayer<
                    Framework::MLPACK, LayerType::Dropout>::type;
                auto layer = new layer_type(
                    ld.setting().drop_probability());
                _m.Add(layer);
                return input_shape;
            }
            case LayerType::Dense:
            default:
            {
                using layer_type = typename MapLayer<
                    Framework::MLPACK, LayerType::Dense>::type;
                auto layer = new layer_type(
                    input_shape.size(),
                    ld.setting().units().size());
                _m.Add(layer);
                return ld.setting().units();
            }
        }
    }

    void _add_activation_layer(const LayerDescriptor& ld)
    {
        switch (ld.activation_type())
        {
            case ActivationType::ReLU:
            {
                using activation_type = typename MapActivation<
                    Framework::MLPACK, ActivationType::ReLU>::type;
                _m.template Add<activation_type>();
                break;
            }
            case ActivationType::ELU:
            {
                using activation_type = typename MapActivation<
                    Framework::MLPACK, ActivationType::ELU>::type;
                _m.template Add<activation_type>();
                break;
            }
            case ActivationType::Softmax:
            {
                using activation_type = typename MapActivation<
                    Framework::MLPACK, ActivationType::Softmax>::type;
                _m.template Add<activation_type>();
                break;
            }
            case ActivationType::TanH:
            {
                using activation_type = typename MapActivation<
                    Framework::MLPACK, ActivationType::TanH>::type;
                _m.template Add<activation_type>();
                break;
            }
            case ActivationType::Sigmoid:
            {
                using activation_type = typename MapActivation<
                    Framework::MLPACK, ActivationType::Sigmoid>::type;
                _m.template Add<activation_type>();
                break;
            }
            case ActivationType::Linear:
            default:
            {
                using activation_type = typename MapActivation<
                    Framework::MLPACK, ActivationType::Linear>::type;
                _m.template Add<activation_type>();
                break;
            }
        }
    }

    mlpack::ann::FFN<
        typename MapLoss<Framework::MLPACK, LT>::type,
        typename MapInit<Framework::MLPACK, IT>::type> _m;
    LayerShape _input_shape;
    LayerShape _output_shape;
    std::vector<std::string> _layers_name;
    bool _is_first_add;
};

template <
    LossType LT,
    InitType IT,
    ParallelizationLevel PL,
    typename T>
struct MapModel<Framework::MLPACK, LT, IT, PL, T> {
    using loss_type = typename MapLoss<Framework::MLPACK, LT>::type;
    using init_type = typename MapInit<Framework::MLPACK, IT>::type;
    using type = mlpack::ann::FFN<loss_type, init_type>;
    using feedforward_model = MlpackFNN<LT, IT, T>;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_MIDDLEWARE_MLPACK_FNN_HPP
