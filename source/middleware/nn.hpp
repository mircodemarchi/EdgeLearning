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

#include "definitions.hpp"
#include "layer_descriptor.hpp"

#include "data/dataset.hpp"

#include <utility>
#include <vector>


namespace EdgeLearning {


using NeuralNetworkDescriptor = std::vector<LayerDescriptor>;

/**
 * \brief High level interface of a neural network.
 * \tparam T Learning Parameters type.
 */
template<typename T = NumType>
class NeuralNetwork {
public:
    using SharedPtr = std::shared_ptr<NeuralNetwork>;

    /**
     * \brief Structure for result of model evaluation.
     */
    struct EvaluationResult {
        /**
         * \brief Initiliaze the performance metrics with loss and accuracy.
         * \param l Loss value.
         * \param a Accuracy value.
         */
        EvaluationResult(T l, T a)
            : loss(l)
            , accuracy(a)
            , accuracy_perc(a * 100.0)
            , error_rate(1.0 - a)
            , error_rate_perc((1.0 - a) * 100.0)
        { }

        EvaluationResult()
            : EvaluationResult(T(0), T(0))
        { }

        T loss;             ///< \brief Average loss.
        T accuracy;         ///< \brief Accuracy: correct / tot.
        T accuracy_perc;    ///< \brief Accuracy in percentage.
        T error_rate;       ///< \brief Error rate: 1 - accuracy.
        T error_rate_perc;  ///< \brief Error rate in percentage.
    };

    /**
     * \brief High level constructor with the model name.
     * \param name std::string The name of the model.
     */
    NeuralNetwork(std::string name)
        : _name{std::move(name)}
    {
    }

    /**
     * \brief Default deconstruct.
     */
    virtual ~NeuralNetwork() = default;

    /**
     * \brief Perform the prediction of a given dataset with the current
     * values of the model parameters.
     * \param data Dataset<T>& The data to predict.
     * \return Dataset<T> The labels predicted.
     */
    virtual Dataset<T> predict(Dataset<T>& data) = 0;

    /**
     * \brief Getter of the input size of the model.
     * \return SizeType Model input size.
     */
    virtual SizeType input_size() = 0;

    /**
     * \brief Getter of the output size of the model.
     * \return SizeType Model output size.
     */
    virtual SizeType output_size() = 0;

    EvaluationResult evaluate(Dataset<T>& data, LossType loss)
    {
        switch (loss) {
            case LossType::CCE:
            {
                return evaluate_static<LossType::CCE>(data);
            }
            case LossType::MSE:
            default:
            {
                return evaluate_static<LossType::MSE>(data);
            }
        }
    }

    /**
     * \brief Compute the performance metrics of the model given a dataset.
     * \tparam LT  The loss enumeration type.
     * \param data Dataset<T>& The dataset used for the evaluation.
     * \return EvaluationResult The resulting performance metrics.
     */
    template <LossType LT>
    EvaluationResult evaluate_static(Dataset<T>& data)
    {
        using loss_type = typename MapLoss<Framework::EDGE_LEARNING, LT>::type;
        loss_type loss("evaluation_loss", output_size(), 1);

        auto data_train = data.trainset();
        auto result = predict(data_train);
        for (std::size_t i = 0; i < result.size(); ++i)
        {
            loss.set_target(data.labels(i));
            loss.forward(result.entry(i));
        }

        return { loss.avg_loss(), loss.accuracy() };
    }

protected:
    std::string _name; ///< \brief The model name.
};

template<typename T = NumType>
class StaticNeuralNetwork : public NeuralNetwork<T> {
public:
    using NeuralNetwork<T>::NeuralNetwork;
    using SharedPtr = std::shared_ptr<StaticNeuralNetwork>;

    /**
     * \brief Add a layer to the model.
     * \param ld LayerDescriptor The layer descriptor.
     */
    virtual void add(LayerDescriptor ld) = 0;

    /**
     * \brief Perform the training of the model with the given dataset.
     * The process will change the value of the model parameters.
     * \param data          The labelled data to use for training.
     * \param optimizer     The optimizer to use for training.
     * \param epochs        The number of iterations over the dataset.
     * \param batch_size    The number of entries evaluated for gradient before
     *                      optimization.
     * \param learning_rate The optimization step size.
     */
    virtual void fit(Dataset<T>& data,
                     OptimizerType optimizer = OptimizerType::ADAM,
                     SizeType epochs = 1,
                     SizeType batch_size = 1,
                     NumType learning_rate = 0.03) = 0;

};

template<typename T = NumType>
class CompileNeuralNetwork : public NeuralNetwork<T> {
public:
    using NeuralNetwork<T>::NeuralNetwork;
    using SharedPtr = std::shared_ptr<CompileNeuralNetwork>;

    /**
     * \brief Perform the training of the model with the given dataset.
     * The process will change the value of the model parameters.
     * \param data          The labelled data to use for training.
     * \param optimizer     The optimizer to use for training.
     * \param epochs        The number of iterations over the dataset.
     * \param batch_size    The number of entries evaluated for gradient before
     *                      optimization.
     * \param learning_rate The optimization step size.
     */
    virtual void fit(Dataset<T>& data,
                     SizeType epochs = 1,
                     SizeType batch_size = 1,
                     NumType learning_rate = 0.03) = 0;

    virtual void compile(LossType loss = LossType::MSE,
                         OptimizerType optimizer = OptimizerType::ADAM,
                         InitType init = InitType::AUTO) = 0;

    virtual typename NeuralNetwork<T>::EvaluationResult evaluate(
        Dataset<T>& data) = 0;

};

template<
    template<Framework, LossType, InitType, ParallelizationLevel, typename>
    typename MM,
    Framework F,
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL,
    typename T = NumType>
class DynamicNeuralNetwork : public CompileNeuralNetwork<T> {
public:
    DynamicNeuralNetwork(NeuralNetworkDescriptor layers, std::string name)
        : CompileNeuralNetwork<T>(name)
        , _model_ptr()
        , _layers(layers)
        , _optimizer()
        , _loss()
    { }

    void compile(LossType loss = LossType::MSE,
                 OptimizerType optimizer = OptimizerType::ADAM,
                 InitType init = InitType::AUTO) override
    {
        _optimizer = optimizer;
        _loss = loss;
        if (loss == LossType::MSE && init == InitType::HE_INIT)
        {
            _model_ptr = std::make_shared<
                typename MM<F, LossType::MSE, InitType::HE_INIT, PL, T>::type>(
                CompileNeuralNetwork<T>::_name);
        }
        else if (loss == LossType::MSE && init == InitType::XAVIER_INIT)
        {
            _model_ptr = std::make_shared<
                typename MM<F, LossType::MSE, InitType::XAVIER_INIT, PL, T>::type>(
                CompileNeuralNetwork<T>::_name);
        }
        else if (loss == LossType::MSE && init == InitType::AUTO)
        {
            _model_ptr = std::make_shared<
                typename MM<F, LossType::MSE, InitType::AUTO, PL, T>::type>(
                CompileNeuralNetwork<T>::_name);
        }
        else if (loss == LossType::CCE && init == InitType::HE_INIT)
        {
            _model_ptr = std::make_shared<
                typename MM<F, LossType::CCE, InitType::HE_INIT, PL, T>::type>(
                CompileNeuralNetwork<T>::_name);
        }
        else if (loss == LossType::CCE && init == InitType::XAVIER_INIT)
        {
            _model_ptr = std::make_shared<
                typename MM<F, LossType::CCE, InitType::XAVIER_INIT, PL, T>::type>(
                CompileNeuralNetwork<T>::_name);
        }
        else if (loss == LossType::CCE && init == InitType::AUTO)
        {
            _model_ptr = std::make_shared<
                typename MM<F, LossType::CCE, InitType::AUTO, PL, T>::type>(
                CompileNeuralNetwork<T>::_name);
        }
        else
        {
            throw std::runtime_error("LossType and InitType not recognized");
        }

        for (const auto& l: _layers)
        {
            _model_ptr->add(l);
        }
    }

    Dataset<T> predict(Dataset<T>& data) override
    {
        if (!_model_ptr)
        {
            compile();
        }
        return _model_ptr->predict(data);
    }

    void fit(Dataset<T>& data,
             SizeType epochs = 1,
             SizeType batch_size = 1,
             NumType learning_rate = 0.03) override
    {
        if (!_model_ptr)
        {
            throw std::runtime_error("Training error: you need to call "
                                     "the compile method before fit");
        }
        return _model_ptr->fit(data, _optimizer,
                               epochs, batch_size, learning_rate);
    }

    SizeType input_size() override
    {
        if (!_model_ptr)
        {
            compile();
        }
        return _model_ptr->input_size();
    }

    SizeType output_size() override
    {
        if (!_model_ptr)
        {
            compile();
        }
        return _model_ptr->output_size();
    }

    typename NeuralNetwork<T>::EvaluationResult evaluate(Dataset<T>& data) override
    {
        if (!_model_ptr)
        {
            throw std::runtime_error("Evaluate error: you need to "
                                     "call the compile method before evaluate");
        }
        return _model_ptr->evaluate(data, _loss);
    }

private:
    typename StaticNeuralNetwork<T>::SharedPtr _model_ptr;
    NeuralNetworkDescriptor _layers;
    OptimizerType _optimizer;
    LossType _loss;
};


} // namespace EdgeLearning


#endif // EDGE_LEARNING_MIDDLEWARE_NN_HPP
