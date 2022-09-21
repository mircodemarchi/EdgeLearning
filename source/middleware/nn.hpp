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

#include "dnn/dense.hpp"
#include "dnn/activation.hpp"
#include "dnn/mse_loss.hpp"
#include "dnn/cce_loss.hpp"
#include "dnn/gd_optimizer.hpp"
#include "dnn/adam_optimizer.hpp"
#include "data/dataset.hpp"

#include <utility>
#include <vector>


namespace EdgeLearning {

enum class Framework
{
    EDGE_LEARNING,
#if ENABLE_MLPACK
    MLPACK,
#endif
};

enum class ParallelizationLevel
{
    SEQUENTIAL,
    THREAD_PARALLELISM_ON_DATA_ENTRY,
    THREAD_PARALLELISM_ON_DATA_BATCH,
};

enum class ActivationType
{
    ReLU,
    ELU,
    Softmax,
    TanH,
    Sigmoid,
    Linear,
    None
};

enum class LossType
{
    CCE,
    MSE,
};

enum class OptimizerType
{
    GRADIENT_DESCENT,
    ADAM
};

enum class InitType
{
    HE_INIT,
    XAVIER_INIT,
    AUTO,
};

using LayerDescriptor = std::tuple<std::string, SizeType, ActivationType>;
using LayerDescriptorVector = std::vector<LayerDescriptor>;

// template <Framework F> struct MatType;

template <Framework F, ActivationType A> struct MapActivation;

template <Framework F, LossType LT> struct MapLoss;

template <Framework F, OptimizerType OT> struct MapOptimizer;

template <Framework F, InitType IT> struct MapInit;

template <
    Framework F,
    LossType LT,
    OptimizerType OT,
    InitType IT,
    ParallelizationLevel PL,
    typename T>
struct MapModel;

/**
 * \brief High level interface of a neural network.
 * \tparam T Learning Parameters type.
 */
template<typename T = NumType>
class NN {
public:

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
    NN(std::string name)
        : _name{std::move(name)}
    {
    }

    /**
     * \brief Default deconstruct.
     */
    virtual ~NN() = default;

    /**
     * \brief Add a layer to the model.
     * \param ld LayerDescriptor The layer descriptor.
     */
    virtual void add(LayerDescriptor ld) = 0;

    /**
     * \brief Perform the prediction of a given dataset with the current
     * values of the model parameters.
     * \param data Dataset<T>& The data to predict.
     * \return Dataset<T> The labels predicted.
     */
    virtual Dataset<T> predict(Dataset<T>& data) = 0;

    /**
     * \brief Perform the training of the model with the given dataset.
     * The process will change the value of the model parameters.
     * \param data          The labelled data to use for training.
     * \param epochs        The number of iterations over the dataset.
     * \param batch_size    The number of entries evaluated for gradient before
     *                      optimization.
     * \param learning_rate The optimization step size.
     */
    virtual void fit(Dataset<T>& data,
                     SizeType epochs = 1,
                     SizeType batch_size = 1,
                     NumType learning_rate = 0.03) = 0;

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

    /**
     * \brief Compute the performance metrics of the model given a dataset.
     * \tparam LT  The loss enumeration type.
     * \param data Dataset<T>& The dataset used for the evaluation.
     * \return EvaluationResult The resulting performance metrics.
     */
    template <LossType LT>
    EvaluationResult evaluate(Dataset<T>& data)
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

} // namespace EdgeLearning


#endif // EDGE_LEARNING_MIDDLEWARE_NN_HPP
