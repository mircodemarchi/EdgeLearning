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

#include "concmanager/include/task_manager.hpp"

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
                for (SizeType b = 0; b < batch_size
                                     && i < data.size(); ++b, ++i)
                {
                    model.step(data.trainset(i), data.labels(i));
                }
                model.train(o);
                model.reset_score();
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
        auto& tm = ConcManager::TaskManager::instance();
        tm.set_maximum_concurrency();
        for (SizeType e = 0; e < epochs; ++e)
        {
            for (SizeType i = 0; i < data.size();)
            {
                std::vector<ConcManager::Future<Model>> futures;
                for (SizeType b = 0; b < batch_size
                                     && i < data.size(); ++b, ++i)
                {
                    futures.push_back(tm.enqueue(
                        [](Model m,
                              std::vector<NumType> train_entry,
                              std::vector<NumType> label_entry) {
                            m.step(train_entry, label_entry);
                            return m;
                        }, model, data.trainset(i), data.labels(i)));
                }

                for (auto& f: futures) {
                    Model m = f.get();
                    model.train(o, m);
                }
                model.reset_score();
            }
        }
    }
};

template<typename T>
struct Training<ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_BATCH, T>
{
public:
    static void run(Model& model, Dataset<T> &data, Optimizer& o,
                    SizeType epochs, SizeType batch_size)
    {
        auto& tm = ConcManager::TaskManager::instance();
        tm.set_maximum_concurrency();
        for (SizeType e = 0; e < epochs; ++e)
        {
            for (SizeType i = 0; i < data.size();)
            {
                std::vector<ConcManager::Future<Model>> futures;
                for (SizeType t = 0; t < tm.maximum_concurrency(); ++t)
                {
                    futures.push_back(tm.enqueue(
                        [batch_size](Model m,
                           Dataset<T> &data_ref,
                           SizeType idx)
                        {
                            for (SizeType b = 0;
                                 b < batch_size && idx < data_ref.size();
                                 ++b, ++idx)
                            {
                                m.step(data_ref.trainset(idx),
                                       data_ref.labels(idx));
                            }
                            return m;
                        }, model, data, i + (t * batch_size)));
                }

                for (auto& f: futures) {
                    Model m = f.get();
                    model.train(o, m);
                }
                model.reset_score();

                i += batch_size * tm.maximum_concurrency();
            }
        }
    }
};

template<
    LossType LT = LossType::MSE,
    OptimizerType OT = OptimizerType::GRADIENT_DESCENT,
    InitType IT = InitType::AUTO,
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL,
    typename T = NumType>
class EdgeFNN : public NN<T> {
public:
    EdgeFNN(std::string name)
        : NN<T>(name)
        , _m(NN<T>::_name)
        , _output_size{0}
    { }

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
                layer_name, _output_size, layer_size);
            if (prev_layer)
            {
                _m.create_edge(prev_layer, layer);
            }
            switch (layer_activation) {
                case ActivationType::ReLU:
                {
                    auto activation_layer = _m.template add_layer<ReluLayer>(
                        layer_name, layer_size);
                    _m.create_edge(layer, activation_layer);
                    break;
                }
                case ActivationType::ELU:
                {
                    auto activation_layer = _m.template add_layer<EluLayer>(
                        layer_name, layer_size);
                    _m.create_edge(layer, activation_layer);
                    break;
                }
                case ActivationType::Softmax:
                {
                    auto activation_layer = _m.template add_layer<SoftmaxLayer>(
                        layer_name, layer_size);
                    _m.create_edge(layer, activation_layer);
                    break;
                }
                case ActivationType::TanH:
                {
                    auto activation_layer = _m.template add_layer<TanhLayer>(
                        layer_name, layer_size);
                    _m.create_edge(layer, activation_layer);
                    break;
                }
                case ActivationType::Sigmoid:
                {
                    auto activation_layer = _m.template add_layer<SigmoidLayer>(
                        layer_name, layer_size);
                    _m.create_edge(layer, activation_layer);
                    break;
                }
                case ActivationType::Linear:
                default:
                {
                    auto activation_layer = _m.template add_layer<LinearLayer>(
                        layer_name, layer_size);
                    _m.create_edge(layer, activation_layer);
                    break;
                }
            }
        }
        _output_size = layer_size;
    }

    void fit(Dataset<T>& data,
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
        _m.create_loss_edge(prev_layer, loss_layer);

        // Train.
        using optimizer_type = typename MapOptimizer<
            Framework::EDGE_LEARNING, OT>::type;
        auto o = optimizer_type(learning_rate);
        _m.init();
        Training<PL, T>::run(_m, data, o, epochs, batch_size);
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
    Model _m;
    SizeType _output_size;
};

template <
    LossType LT,
    OptimizerType OT,
    InitType IT,
    ParallelizationLevel PL,
    typename T>
struct MapModel<Framework::EDGE_LEARNING, LT, OT, IT, PL, T> {
    using loss_type = typename MapLoss<Framework::EDGE_LEARNING, LT>::type;
    using optimizer_type = typename MapOptimizer<
        Framework::EDGE_LEARNING, OT>::type;
    static const InitType init_type = MapInit<
        Framework::EDGE_LEARNING, IT>::type;
    using type = Model;
    using fnn = EdgeFNN<LT, OT, IT, PL, T>;
};


template<
    Framework F = Framework::EDGE_LEARNING,
    LossType LT = LossType::MSE,
    OptimizerType OT = OptimizerType::ADAM,
    InitType IT = InitType::AUTO,
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL,
    typename T = NumType>
class FNN {
public:
    using ModelFNN = typename MapModel<F, LT, OT, IT, PL, T>::fnn;

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

    typename ModelFNN::EvaluationResult evaluate(Dataset<T>& data)
    {
        return _fnn_model.template evaluate<LT>(data);
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
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL,
    typename T = NumType>
using CompileFNN = FNN<Framework::MLPACK, LT, OT, IT, PL, T>;
#else
template<
    LossType LT = LossType::MSE,
    OptimizerType OT = OptimizerType::GRADIENT_DESCENT,
    InitType IT = InitType::AUTO,
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL,
    typename T = NumType>
using CompileFNN = FNN<Framework::EDGE_LEARNING, LT, OT, IT, PL, T>;
#endif

} // namespace EdgeLearning

#endif // EDGE_LEARNING_MIDDLEWARE_FNN_HPP
