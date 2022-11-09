/***************************************************************************
 *            profile_fnn.hpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 * This file is part of Opera, under the MIT license.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef EDGE_LEARNING_PROFILE_FNN_HPP
#define EDGE_LEARNING_PROFILE_FNN_HPP

#include "profile.hpp"
#include "profile_dataset.hpp"


using namespace EdgeLearning;
namespace fs = std::filesystem;

template <LossType LT> struct MapOutputActivation;

template <>
struct MapOutputActivation<LossType::CCE>
{
    inline static const ActivationType type = ActivationType::Softmax;
};

template <>
struct MapOutputActivation<LossType::MSE>
{
    inline static const ActivationType type = ActivationType::Linear;
};

template <
    LossType LT,
    OptimizerType OT,
    ParallelizationLevel PL = ParallelizationLevel::SEQUENTIAL>
class ProfileFNN : public ProfileNN
{
public:
    ProfileFNN(std::string profile_name,
               ProfileDataset::Type dataset_type,
               std::vector<NeuralNetworkDescriptor> hidden_layers_descriptor_vec,
               TrainingSetting default_setting)
        : ProfileNN(
            100,
#if ENABLE_MLPACK
            "profile_mlpack_fnn_" + profile_name
#else
            "profile_edgelearning_fnn_" + profile_name
#endif
            )
        , _profile_name(profile_name)
        , _dataset_type(dataset_type)
        , _hidden_layers_descriptor_vec(hidden_layers_descriptor_vec)
        , _default_setting(default_setting)
    { }

    virtual void run() {
        EDGE_LEARNING_PROFILE_TITLE(
            "FNN training and prediction process when "
            "solving a " + _profile_name + " problem");
        std::cout << "*** Dataset: " + std::string(ProfileDataset(_dataset_type))
                  << " ***" << std::endl;
        auto data = ProfileDataset(_dataset_type).load_dataset();

        for (const auto& nn_descriptor: _hidden_layers_descriptor_vec)
        {
            EDGE_LEARNING_PROFILE_CALL(profile_on_fixed_parameters(nn_descriptor, data));
            EDGE_LEARNING_PROFILE_CALL(profile_on_parallelism_level(nn_descriptor, data));
            EDGE_LEARNING_PROFILE_CALL(profile_on_epochs_amount(nn_descriptor, data));
            EDGE_LEARNING_PROFILE_CALL(profile_on_training_set(nn_descriptor, data));
        }

        EDGE_LEARNING_PROFILE_CALL(profile_on_layers_amount(data));
        EDGE_LEARNING_PROFILE_CALL(profile_on_layers_shape(data));
    }

private:
    using ProfileCompileFNN = CompileFeedforwardNeuralNetwork<LT, InitType::AUTO, PL>;

    void profile_on_fixed_parameters(
        NeuralNetworkDescriptor nn_descriptor, ProfileDataset::Info& data)
    {
        auto data_training = data.train;
        auto data_evaluation = data.evaluation;
        auto data_testing = data.test;
        auto input_size = data.input_shape;
        auto output_size = data_training.label_idx().size();

        NeuralNetworkDescriptor layers_descriptor(nn_descriptor);
        layers_descriptor.insert(
            layers_descriptor.begin(),
            Input{"input_layer", input_size});
        layers_descriptor.push_back(
            Dense{"output_layer", output_size, MapOutputActivation<LT>::type});

        training_testing<ProfileCompileFNN>(
            "training and testing of sequential model with "
            "default parameters: { "
            "epochs: " + std::to_string(_default_setting.epochs) +
            ", batch_size: " + std::to_string(_default_setting.batch_size) +
            ", learning_rate: " + std::to_string(_default_setting.learning_rate) +
            " }",
            "training_testing_default_parameters",
            1, data_training, data_evaluation, data_testing,
            layers_descriptor,
            OT,
            _default_setting.epochs,
            _default_setting.batch_size,
            _default_setting.learning_rate);
    }

    void profile_on_parallelism_level(
        NeuralNetworkDescriptor nn_descriptor, ProfileDataset::Info& data)
    {
        auto data_training = data.train;
        auto data_evaluation = data.evaluation;
        auto data_testing = data.test;
        auto input_size = data.input_shape;
        auto output_size = data_training.label_idx().size();

        NeuralNetworkDescriptor layers_descriptor(nn_descriptor);
        layers_descriptor.insert(
            layers_descriptor.begin(),
            Input{"input_layer", input_size});
        layers_descriptor.push_back(
            Dense{"output_layer", output_size, MapOutputActivation<LT>::type});

        std::vector<SizeType> batch_sizes = {1, 4, 16, 32, 64, 128};
        std::vector<NumType> learning_rates = {0.3, 0.1, 0.03, 0.01};

        using ProfileCompileFNNSequential = CompileFeedforwardNeuralNetwork<
            LT,
            InitType::AUTO,
            ParallelizationLevel::SEQUENTIAL>;

        using ProfileCompileFNNThreadOnEntry = CompileFeedforwardNeuralNetwork<
            LT,
            InitType::AUTO,
            ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_ENTRY>;

        using ProfileCompileFNNThreadOnBatch = CompileFeedforwardNeuralNetwork<
            LT,
            InitType::AUTO,
            ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_BATCH>;

        for (const auto& batch_size: batch_sizes)
        {
            training_testing<ProfileCompileFNNSequential>(
                "training and testing sequential model with batch size: "
                    + std::to_string(batch_size),
                "training_testing_sequential_on_batch_size"
                    + std::to_string(batch_size),
                1, data_training, data_evaluation, data_testing,
                layers_descriptor,
                OT,
                _default_setting.epochs,
                batch_size,
                _default_setting.learning_rate);

            training_testing<ProfileCompileFNNThreadOnEntry>(
                "training and testing thread parallelism on data entry model "
                "with batch size: " + std::to_string(batch_size),
                "training_testing_thread_parallelism_entry_on_batch_size"
                    + std::to_string(batch_size),
                1, data_training, data_evaluation, data_testing,
                layers_descriptor,
                OT,
                _default_setting.epochs,
                batch_size,
                _default_setting.learning_rate);

            training_testing<ProfileCompileFNNThreadOnBatch>(
                "training and testing thread parallelism on data batch model "
                "with batch size: " + std::to_string(batch_size),
                "training_testing_thread_parallelism_batch_on_batch_size"
                    + std::to_string(batch_size),
                1, data_training, data_evaluation, data_testing,
                layers_descriptor,
                OT,
                _default_setting.epochs,
                batch_size,
                _default_setting.learning_rate);
        }

        for (const auto& learning_rate: learning_rates)
        {
            training_testing<ProfileCompileFNNSequential>(
                "training and testing sequential model with learning rate: "
                    + std::to_string(learning_rate),
                "training_testing_sequential_on_learning_rate"
                    + std::to_string(learning_rate),
                1, data_training, data_evaluation, data_testing,
                layers_descriptor,
                OT,
                _default_setting.epochs,
                _default_setting.batch_size,
                learning_rate);

            training_testing<ProfileCompileFNNThreadOnEntry>(
                "training and testing thread parallelism on data entry model "
                "with learning rate: " + std::to_string(learning_rate),
                "training_testing_thread_parallelism_entry_on_learning_rate"
                    + std::to_string(learning_rate),
                1, data_training, data_evaluation, data_testing,
                layers_descriptor,
                OT,
                _default_setting.epochs,
                _default_setting.batch_size,
                learning_rate);

            training_testing<ProfileCompileFNNThreadOnBatch>(
                "training and testing thread parallelism on data batch model "
                "with learning rate: " + std::to_string(learning_rate),
                "training_testing_thread_parallelism_batch_on_learning_rate"
                    + std::to_string(learning_rate),
                1, data_training, data_evaluation, data_testing,
                layers_descriptor,
                OT,
                _default_setting.epochs,
                _default_setting.batch_size,
                learning_rate);
        }
    }

    /**
     * \brief Profile the training and the prediction phase of a FNN model
     * on epochs incrementation.
     */
    void profile_on_epochs_amount(
        NeuralNetworkDescriptor nn_descriptor, ProfileDataset::Info& data)
    {
        auto data_training = data.train;
        auto data_evaluation = data.evaluation;
        auto data_testing = data.test;
        auto input_size = data.input_shape;
        auto output_size = data_training.label_idx().size();

        NeuralNetworkDescriptor layers_descriptor(nn_descriptor);
        layers_descriptor.insert(
            layers_descriptor.begin(),
            Input{"input_layer", input_size});
        layers_descriptor.push_back(
            Dense{"output_layer", output_size, MapOutputActivation<LT>::type});

        for (SizeType e = 1; e <= _default_setting.epochs; ++e)
        {
            training_testing<ProfileCompileFNN>(
                "training and testing epochs amount: " + std::to_string(e),
                "training_on_epochs_amount" + std::to_string(e),
                100, data_training, data_evaluation, data_testing,
                layers_descriptor,
                OT, e,
                _default_setting.batch_size,
                _default_setting.learning_rate);
        }

        predict<ProfileCompileFNN>(
            "prediction after training with epochs amount: "
                + std::to_string(_default_setting.epochs),
            "prediction",
            100, data_training,
            layers_descriptor,
            OT,
            _default_setting.epochs,
            _default_setting.batch_size,
            _default_setting.learning_rate);
    }

    /**
     * \brief Profile the training and the prediction phase of a FNN model
     * on different training set amount and fixed epoch amount.
     */
    void profile_on_training_set(
        NeuralNetworkDescriptor nn_descriptor, ProfileDataset::Info& data)
    {
        auto data_training = data.train;
        auto data_evaluation = data.evaluation;
        auto data_testing = data.test;
        auto training_set_size = data_training.size();
        auto input_size = data.input_shape;
        auto output_size = data_training.label_idx().size();

        NeuralNetworkDescriptor layers_descriptor(nn_descriptor);
        layers_descriptor.insert(
            layers_descriptor.begin(),
            Input{"input_layer", input_size});
        layers_descriptor.push_back(
            Dense{"output_layer", output_size, MapOutputActivation<LT>::type});

        std::vector<std::size_t> training_set_sizes = {
            10, 50, 100, 200, 300, 400, 600, 800, 1000, 10000
        };
        for (const auto& size: training_set_sizes)
        {
            auto curr_size = std::min(training_set_size, size);
            auto subdata_training = data_training.subdata(0, curr_size);
            auto subdata_evaluation = data_evaluation.subdata(
                0, std::min(curr_size, data_evaluation.size()));
            auto subdata_testing = data_testing.subdata(
                0, std::min(curr_size, data_testing.size()));
            training_testing<ProfileCompileFNN>(
                "training and testing with dataset size (#entries): "
                    + std::to_string(curr_size),
                "training_testing_on_dataset_size" + std::to_string(curr_size),
                100, subdata_training, subdata_evaluation, subdata_testing,
                layers_descriptor,
                OT,
                _default_setting.epochs,
                _default_setting.batch_size,
                _default_setting.learning_rate);
            if (curr_size == training_set_size) break;
        }

        for (const auto& size: training_set_sizes)
        {
            auto curr_size = std::min(training_set_size, size);
            auto subset = data_training.subdata(0, curr_size);
            predict<ProfileCompileFNN>(
                "prediction with dataset size (#entries): "
                    + std::to_string(curr_size),
                "prediction_on_dataset_size" + std::to_string(curr_size),
                100, subset,
                layers_descriptor,
                OT,
                _default_setting.epochs,
                _default_setting.batch_size,
                _default_setting.learning_rate);
            if (curr_size == training_set_size) break;
        }
    }

    /**
     * \brief Profile the training and the prediction phase of a FNN model
     * on incremental layers amount.
     */
    void profile_on_layers_amount(ProfileDataset::Info& data) {
        auto data_training = data.train;
        auto data_evaluation = data.evaluation;
        auto data_testing = data.test;
        auto input_size = data.input_shape;
        auto output_size = data_training.label_idx().size();

        NeuralNetworkDescriptor layers_descriptor(
            {
                Input{"input_layer",   input_size},
                Dense{"output_layer",  output_size, MapOutputActivation<LT>::type },
            }
        );

        std::vector<std::size_t> layers_amounts = {
            1, 2, 5, 10, 20, 50, 100
        };
        for (const auto& amount: layers_amounts)
        {
            auto curr_ld = layers_descriptor;
            for (std::size_t i = 0; i < amount; ++i)
            {
                std::string name = "hidden_layer" + std::to_string(amount - 1);
                curr_ld.insert(
                    layers_descriptor.end() - 1,
                    Dense{name, 32, ActivationType::ReLU});
            }

            training_testing<ProfileCompileFNN>(
                "training and testing with hidden layers amount: "
                    + std::to_string(amount),
                "training_testing_on_hidden_layers_amount"
                    + std::to_string(amount),
                100, data_training, data_evaluation, data_testing,
                curr_ld,
                OT,
                _default_setting.epochs,
                _default_setting.batch_size,
                _default_setting.learning_rate);

            predict<ProfileCompileFNN>(
                "prediction with hidden layers amount: "
                    + std::to_string(amount),
                "prediction_on_hidden_layers_amount" + std::to_string(amount),
                100, data_training,
                curr_ld,
                OT,
                _default_setting.epochs,
                _default_setting.batch_size,
                _default_setting.learning_rate);
        }
    }

    /**
     * \brief Profile the training phase of a FNN model
     * on different hidden layer shape.
     */
    void profile_on_layers_shape(ProfileDataset::Info& data) {
        auto data_training = data.train;
        auto data_evaluation = data.evaluation;
        auto data_testing = data.test;
        auto input_size = data.input_shape;
        auto output_size = data_training.label_idx().size();

        std::vector<std::size_t> layers_shapes = {
            10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000
        };
        for (const auto& shape: layers_shapes)
        {
            NeuralNetworkDescriptor layers_descriptor(
                {
                    Input{"input_layer",   input_size},
                    Dense{"hidden_layer0", shape,       ActivationType::ReLU   },
                    Dense{"output_layer",  output_size, MapOutputActivation<LT>::type},
                }
            );
            training_testing<ProfileCompileFNN>(
                "training and testing with hidden layers shape: "
                    + std::to_string(shape),
                "training_testing_on_hidden_layers_shape"
                    + std::to_string(shape),
                100, data_training, data_evaluation, data_testing,
                layers_descriptor,
                OT,
                _default_setting.epochs,
                _default_setting.batch_size,
                _default_setting.learning_rate);

            predict<ProfileCompileFNN>(
                "prediction with hidden layers shape: " + std::to_string(shape),
                "prediction_on_hidden_layers_shape" + std::to_string(shape),
                100, data_training,
                layers_descriptor,
                OT,
                _default_setting.epochs,
                _default_setting.batch_size,
                _default_setting.learning_rate);
        }
    }

    std::string _profile_name;
    ProfileDataset::Type _dataset_type;
    std::vector<NeuralNetworkDescriptor> _hidden_layers_descriptor_vec;
    TrainingSetting _default_setting;
};

#endif // EDGE_LEARNING_PROFILE_FNN_HPP