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

#include "nn.hpp"
#include "layer_descriptor.hpp"
#if ENABLE_MLPACK
#include "mlpack_fnn.hpp"
#endif

#include "betterthreads/task_manager.hpp"

#include <map>
#include <tuple>
#include <utility>
#include <vector>
#include <future>


namespace EdgeLearning {

template<
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL,
    typename T = NumType>
struct Training { };

template<typename T>
struct Training<ParallelizationLevel::SEQUENTIAL, T>
{
public:
    static void run(Model& model, Dataset<T> &data, Optimizer& o,
             SizeType epochs, SizeType batch_size)
    {
        for (SizeType e = 0; e < epochs; ++e)
        {
            for (SizeType i = 0; i < data.size();)
            {
                model.reset_score();
                for (SizeType b = 0; b < batch_size
                                     && i < data.size(); ++b, ++i)
                {
                    model.step(data.input(i), data.label(i));
                }
                model.train(o);
                std::cout << "step " << i << std::endl;
            }
        }
    }
};

template<typename T>
struct Training<ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_ENTRY, T>
{
public:
    static void run(Model& model, Dataset<T> &data, Optimizer& o,
                    SizeType epochs, SizeType batch_size)
    {
        auto& tm = BetterThreads::TaskManager::instance();
        tm.set_maximum_concurrency();
        for (SizeType e = 0; e < epochs; ++e)
        {
            for (SizeType i = 0; i < data.size();)
            {
                model.reset_score();

                std::vector<BetterThreads::Future<Model>> futures;
                for (SizeType b = 0; b < batch_size
                                     && i < data.size(); ++b, ++i)
                {
                    futures.push_back(tm.enqueue(
                        [&](SizeType idx) {
                            Model m(model);
                            m.step(data.input(idx), data.label(idx));
                            return m;
                        }, i));
                    std::cout << "step " << i << std::endl;
                }

                for (auto& f: futures) {
                    Model m = f.get();
                    model.train(o, m);
                }
            }
        }
    }
};

template<typename T>
struct Training<ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_BATCH, T>
{
public:
    static void run(Model& model, const Dataset<T>& data, Optimizer& o,
                    SizeType epochs, SizeType batch_size)
    {
        auto& tm = BetterThreads::TaskManager::instance();
        tm.set_maximum_concurrency();
        for (SizeType e = 0; e < epochs; ++e)
        {
            for (SizeType i = 0; i < data.size();)
            {
                model.reset_score();

                std::vector<BetterThreads::Future<Model>> futures;
                for (SizeType t = 0; t < tm.concurrency(); ++t)
                {
                    futures.push_back(tm.enqueue(
                        [&, batch_size](SizeType idx)
                        {
                            Model m(model);
                            for (SizeType b = 0;
                                 b < batch_size && idx < data.size();
                                 ++b, ++idx)
                            {
                                m.step(data.input(idx),
                                       data.label(idx));
                            }
                            return m;
                        }, i));
                    std::cout << "step " << i << std::endl;
                    i += batch_size;
                }

                for (auto& f: futures) {
                    Model m = f.get();
                    model.train(o, m);
                }
            }
        }
    }
};

template<
    LossType LT = LossType::MSE,
    InitType IT = InitType::AUTO,
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL,
    typename T = NumType>
class EdgeFeedforwardNeuralNetwork : public StaticNeuralNetwork<T> {
public:
    EdgeFeedforwardNeuralNetwork(std::string name)
        : StaticNeuralNetwork<T>(name)
        , _m(StaticNeuralNetwork<T>::_name)
        , _output_shape{0}
        , _is_first_add{true}
    { }

    void add(LayerDescriptor ld) override
    {
        if (_is_first_add)
        {
            _is_first_add = false;
            if (ld.type() == LayerType::Input)
            {
                _output_shape = ld.setting().units();
                return;
            }
        }

        auto layer = _add_layer(ld, _output_shape);
        if (!layer) return;
        _output_shape = layer->output_shape();
        if (_m.layers().size() > 1)
        {
            _m.create_edge(_m.layers()[_m.layers().size() - 2], layer);
        }
        auto activation_layer = _add_activation_layer(ld, _output_shape);
        if (!activation_layer) return;
        _m.create_edge(layer, activation_layer);
    }

    void fit(Dataset<T>& data,
             OptimizerType optimizer = OptimizerType::ADAM,
             SizeType epochs = 1,
             SizeType batch_size = 1,
             NumType learning_rate = 0.03,
             RneType::result_type seed = 0) override
    {
        if (_m.layers().empty())
        {
            throw std::runtime_error(
                "The FNN has no layer: call add before fit");
        }

        // Add Loss layer.
        if (_m.loss_layers().empty())
        {
            using LossLayerType = typename MapLoss<Framework::EDGE_LEARNING, LT>::type;
            auto loss_layer_name = MapLoss<Framework::EDGE_LEARNING, LT>::name;
            for (auto output_layer: _m.output_layers())
            {
                auto loss_layer = _m.add_loss<LossLayerType>(
                    loss_layer_name, output_layer->output_size(), batch_size);
                _m.create_loss_edge(output_layer, loss_layer);
            }
        }

        // Train.
        _m.init(MapInit<Framework::EDGE_LEARNING, IT>::type,
                Model::ProbabilityDensityFunction::NORMAL, seed);
        _fit(optimizer, data, epochs, batch_size, learning_rate);
    }

    Dataset<T> predict(Dataset<T> &data) override
    {
        auto output_size = _m.output_size();
        std::vector<T> ret;
        ret.resize(data.size() * output_size);

        for (std::size_t i = 0; i < data.size(); ++i)
        {
            auto res = _m.predict(data.entry(i));
            std::copy(res.begin(), res.end(),
                      ret.begin() + long(i * output_size));
        }
        return Dataset<T>(ret, output_size);
    }

    SizeType input_size() override { return _m.input_size(); }
    SizeType output_size() override { return _m.output_size(); }

private:
    void _fit(OptimizerType optimizer,
              Dataset<T>& data,
              SizeType epochs, SizeType batch_size, NumType learning_rate)
    {
        switch (optimizer) {
            case OptimizerType::GRADIENT_DESCENT:
            {
                using optimizer_type = typename MapOptimizer<
                    Framework::EDGE_LEARNING, OptimizerType::GRADIENT_DESCENT>::type;
                auto o = optimizer_type(learning_rate);
                Training<PL, T>::run(_m, data, o, epochs, batch_size);
                break;
            }
            case OptimizerType::ADAM:
            default:
            {
                using optimizer_type = typename MapOptimizer<
                    Framework::EDGE_LEARNING, OptimizerType::ADAM>::type;
                auto o = optimizer_type(learning_rate);
                Training<PL, T>::run(_m, data, o, epochs, batch_size);
                break;
            }
        }
    }

    Layer::SharedPtr _add_layer(const LayerDescriptor& ld,
                                const LayerShape& input_shape)
    {
        const auto& layer_name = ld.name();
        switch (ld.type())
        {
            case LayerType::Input:
            {
                throw std::runtime_error("Model structure error: "
                                         "Input layer have to be "
                                         "put as first layer");
            }
            case LayerType::Conv: {
                auto layer = _m.template add_layer<ConvolutionalLayer>(
                    layer_name, input_shape.shape(),
                    ld.setting().kernel_shape(),
                    ld.setting().n_filters(),
                    ld.setting().stride(),
                    ld.setting().padding());
                return layer;
            }
            case LayerType::MaxPool:
            {
                auto layer = _m.template add_layer<MaxPoolingLayer>(
                    layer_name, input_shape.shape(),
                    ld.setting().kernel_shape(),
                    ld.setting().stride());
                return layer;
            }
            case LayerType::AvgPool:
            {
                auto layer = _m.template add_layer<AveragePoolingLayer>(
                    layer_name, input_shape.shape(),
                    ld.setting().kernel_shape(),
                    ld.setting().stride());
                return layer;
            }
            case LayerType::Dropout: {
                auto layer = _m.template add_layer<DropoutLayer>(
                    layer_name, input_shape.size(),
                    ld.setting().drop_probability());
                return layer;
            }
            case LayerType::Dense:
            default:
            {
                auto layer = _m.template add_layer<DenseLayer>(
                    layer_name, input_shape.size(),
                    ld.setting().units().size());
                return layer;
            }
        }
    }

    Layer::SharedPtr _add_activation_layer(const LayerDescriptor& ld,
                                           const LayerShape& input_shape)
    {
        const auto& layer_name = ld.name();

        switch (ld.activation_type())
        {
            case ActivationType::ReLU:
            {
                auto activation_layer = _m.template add_layer<ReluLayer>(
                    layer_name, input_shape.size());
                return activation_layer;
            }
            case ActivationType::ELU:
            {
                auto activation_layer = _m.template add_layer<EluLayer>(
                    layer_name, input_shape.size());
                return activation_layer;
            }
            case ActivationType::Softmax:
            {
                auto activation_layer = _m.template add_layer<SoftmaxLayer>(
                    layer_name, input_shape.size());
                return activation_layer;
            }
            case ActivationType::TanH:
            {
                auto activation_layer = _m.template add_layer<TanhLayer>(
                    layer_name, input_shape.size());
                return activation_layer;
            }
            case ActivationType::Sigmoid:
            {
                auto activation_layer = _m.template add_layer<SigmoidLayer>(
                    layer_name, input_shape.size());
                return activation_layer;
            }
            case ActivationType::Linear:
            default:
            {
                auto activation_layer = _m.template add_layer<LinearLayer>(
                    layer_name, input_shape.size());
                return activation_layer;
            }
        }
    }

    Model _m;
    LayerShape _output_shape;
    bool _is_first_add;
};

template <
    LossType LT,
    InitType IT,
    ParallelizationLevel PL,
    typename T>
struct MapModel<Framework::EDGE_LEARNING, LT, IT, PL, T> {
    using loss_type = typename MapLoss<Framework::EDGE_LEARNING, LT>::type;
    static const Model::InitializationFunction init_type = MapInit<
        Framework::EDGE_LEARNING, IT>::type;
    static const ParallelizationLevel parallelization_level = PL;
    using type = Model;
    using feedforward_model = EdgeFeedforwardNeuralNetwork<LT, IT, PL, T>;
};


template<
    Framework F = Framework::EDGE_LEARNING,
    LossType LT = LossType::MSE,
    InitType IT = InitType::AUTO,
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL,
    typename T = NumType>
class FeedforwardNeuralNetwork {
public:
    using SubModelType = typename MapModel<F, LT, IT, PL, T>::feedforward_model;
    using EvaluationResult = typename SubModelType::EvaluationResult;

    FeedforwardNeuralNetwork(NeuralNetworkDescriptor layers, std::string name)
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
             OptimizerType optimizer = OptimizerType::GRADIENT_DESCENT,
             SizeType epochs = 1,
             SizeType batch_size = 1,
             NumType learning_rate = 0.03,
             RneType::result_type seed = 0)
    {
        _fnn_model.fit(data, optimizer,
                       epochs, batch_size, learning_rate, seed);
    }

    EvaluationResult evaluate(Dataset<T>& data)
    {
        return _fnn_model.template evaluate_static<LT>(data);
    }

private:
    NeuralNetworkDescriptor _layers;
    SubModelType _fnn_model;
};

#if ENABLE_MLPACK
template<
    LossType LT = LossType::MSE,
    InitType IT = InitType::AUTO,
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL,
    typename T = NumType>
using CompileFeedforwardNeuralNetwork = FeedforwardNeuralNetwork<Framework::MLPACK, LT, IT, PL, T>;
#else
template<
    LossType LT = LossType::MSE,
    InitType IT = InitType::AUTO,
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL,
    typename T = NumType>
using CompileFeedforwardNeuralNetwork = FeedforwardNeuralNetwork<Framework::EDGE_LEARNING, LT, IT, PL, T>;
#endif

template<
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL,
    typename T = NumType>
class DynamicFeedforwardNeuralNetwork {
public:

    template <
        Framework MM_F,
        LossType MM_LT,
        InitType MM_IT,
        ParallelizationLevel MM_PL,
        typename MM_T>
    struct MapModelFeedforward {
        using type = typename MapModel<MM_F, MM_LT, MM_IT, MM_PL, MM_T>::feedforward_model;
    };

    template<Framework F>
    using SubModelType = DynamicNeuralNetwork<MapModelFeedforward, F, PL, T>;

    using EvaluationResult = typename NeuralNetwork<T>::EvaluationResult;

    DynamicFeedforwardNeuralNetwork(NeuralNetworkDescriptor layers, std::string name)
        : _fnn_models{}
    {
        auto _edge_learning_fnn_model = std::make_shared<
            SubModelType<Framework::EDGE_LEARNING>>(layers, name);
        _fnn_models[Framework::EDGE_LEARNING] = _edge_learning_fnn_model;

#if ENABLE_MLPACK
        auto _mlpack_fnn_model = std::make_shared<
            SubModelType<Framework::MLPACK>>(layers, name);
        _fnn_models[Framework::MLPACK] = _mlpack_fnn_model;
#endif
    }

    Dataset<T> predict(Framework framework, Dataset<T> &data)
    {
        return _fnn_models[framework]->predict(data);
    }

    std::map<Framework, Dataset<T>> predict(Dataset<T> &data)
    {
        std::map<Framework, Dataset<T>> ret;
        for (const auto& e: _fnn_models)
        {
            ret[e.first] = predict(e.first, data);
        }
        return ret;
    }

    void fit(Framework framework, Dataset<T> &data,
             SizeType epochs = 1,
             SizeType batch_size = 1,
             NumType learning_rate = 0.03,
             RneType::result_type seed = 0)
    {
        _fnn_models[framework]->fit(data, epochs,
                                    batch_size, learning_rate, seed);
    }

    void fit(Dataset<T> &data,
             SizeType epochs = 1,
             SizeType batch_size = 1,
             NumType learning_rate = 0.03,
             RneType::result_type seed = 0)
    {
        for (const auto& e: _fnn_models)
        {
            fit(e.first, data, epochs, batch_size, learning_rate, seed);
        }
    }

    void compile(Framework framework, LossType loss = LossType::MSE,
                 OptimizerType optimizer = OptimizerType::ADAM,
                 InitType init = InitType::AUTO)
    {
        _fnn_models[framework]->compile(loss, optimizer, init);
    }

    void compile(LossType loss = LossType::MSE,
                 OptimizerType optimizer = OptimizerType::ADAM,
                 InitType init = InitType::AUTO)
    {
        for (const auto& e: _fnn_models)
        {
            compile(e.first, loss, optimizer, init);
        }
    }

    EvaluationResult evaluate(
        Framework framework, Dataset<T>& data)
    {
        return _fnn_models[framework]->evaluate(data);
    }

    std::map<Framework, EvaluationResult> evaluate(
        Dataset<T>& data)
    {
        std::map<Framework, typename NeuralNetwork<T>::EvaluationResult > ret;
        for (const auto& e: _fnn_models)
        {
            ret[e.first] = evaluate(e.first, data);
        }
        return ret;
    }

private:
    std::map<Framework, typename CompileNeuralNetwork<T>::SharedPtr> _fnn_models;
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_MIDDLEWARE_FNN_HPP
