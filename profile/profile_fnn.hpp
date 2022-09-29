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
               std::vector<ProfileDataset::Type> dataset_types,
               NNDescriptor hidden_layers_descriptor,
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
        , _dataset_types(dataset_types)
        , _hidden_layers_descriptor(hidden_layers_descriptor)
        , _default_setting(default_setting)
    { }

    virtual void run() {
        EDGE_LEARNING_PROFILE_TITLE(
            "FNN training and prediction process when "
            "solving a " + _profile_name + " problem");
        for (const auto& dt: _dataset_types)
        {
            std::cout << "*** Dataset: " + std::string(ProfileDataset(dt))
                      << " ***" << std::endl;
            profile_on_fixed_parameters(dt);
            EDGE_LEARNING_PROFILE_CALL(profile_on_parallelism_level(dt));
            EDGE_LEARNING_PROFILE_CALL(profile_on_epochs_amount(dt));
            EDGE_LEARNING_PROFILE_CALL(profile_on_training_set(dt));
            EDGE_LEARNING_PROFILE_CALL(profile_on_layers_amount(dt));
            EDGE_LEARNING_PROFILE_CALL(profile_on_layers_shape(dt));
        }
    }

private:
    using ProfileCompileFNN = CompileFNN<LT, OT, InitType::AUTO, PL>;

    void profile_on_fixed_parameters(ProfileDataset::Type pd_type) {
        auto data = ProfileDataset(pd_type).load_dataset();
        auto data_training = std::get<0>(data);
        auto data_validation = std::get<1>(data);
        auto data_testing = std::get<2>(data);
        // auto input_size = data_training.trainset_idx().size();
        auto output_size = data_training.labels_idx().size();

        NNDescriptor layers_descriptor(_hidden_layers_descriptor);
        layers_descriptor.insert(
            layers_descriptor.begin(),
            Input{"input_layer", DLMath::Shape3d{28, 28, 1}});
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
            1, data_training, data_validation, data_testing,
            layers_descriptor,
            _default_setting.epochs,
            _default_setting.batch_size,
            _default_setting.learning_rate);
    }

    void profile_on_parallelism_level(ProfileDataset::Type pd_type) {
        auto data = ProfileDataset(pd_type).load_dataset();
        auto data_training = std::get<0>(data);
        auto data_validation = std::get<1>(data);
        auto data_testing = std::get<2>(data);
        // auto input_size = data_training.trainset_idx().size();
        auto output_size = data_training.labels_idx().size();

        NNDescriptor layers_descriptor(_hidden_layers_descriptor);
        layers_descriptor.insert(
            layers_descriptor.begin(),
            Input{"input_layer", DLMath::Shape3d{28, 28, 1}});
        layers_descriptor.push_back(
            Dense{"output_layer", output_size, MapOutputActivation<LT>::type});

        std::vector<SizeType> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128};

        using ProfileCompileFNNSequential = CompileFNN<
            LT,
            OT,
            InitType::AUTO,
            ParallelizationLevel::SEQUENTIAL>;

        using ProfileCompileFNNThreadOnEntry = CompileFNN<
            LT,
            OT,
            InitType::AUTO,
            ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_ENTRY>;

        using ProfileCompileFNNThreadOnBatch = CompileFNN<
            LT,
            OT,
            InitType::AUTO,
            ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_BATCH>;

        for (const auto& batch_size: batch_sizes)
        {
            training_testing<ProfileCompileFNNSequential>(
                "training and testing sequential model with batch_size: "
                    + std::to_string(batch_size),
                "training_testing_sequential_on_batch_size"
                    + std::to_string(batch_size),
                1, data_training, data_validation, data_testing,
                layers_descriptor,
                _default_setting.epochs,
                batch_size,
                _default_setting.learning_rate);
        }

        for (const auto& batch_size: batch_sizes)
        {
            training_testing<ProfileCompileFNNThreadOnEntry>(
                "training and testing thread parallelism on data entry model "
                "with batch_size: " + std::to_string(batch_size),
                "training_testing_thread_parallelism_entry_on_batch_size"
                    + std::to_string(batch_size),
                1, data_training, data_validation, data_testing,
                layers_descriptor,
                _default_setting.epochs,
                batch_size,
                _default_setting.learning_rate);
        }

        for (const auto& batch_size: batch_sizes)
        {
            training_testing<ProfileCompileFNNThreadOnBatch>(
                "training and testing thread parallelism on data batch model "
                "with batch_size: " + std::to_string(batch_size),
                "training_testing_thread_parallelism_batch_on_batch_size"
                    + std::to_string(batch_size),
                1, data_training, data_validation, data_testing,
                layers_descriptor,
                _default_setting.epochs,
                batch_size,
                _default_setting.learning_rate);
        }
    }

    /**
     * \brief Profile the training and the prediction phase of a FNN model
     * on epochs incrementation.
     */
    void profile_on_epochs_amount(ProfileDataset::Type pd_type) {
        auto data = ProfileDataset(pd_type).load_dataset();
        auto data_training = std::get<0>(data);
        auto data_validation = std::get<1>(data);
        auto data_testing = std::get<2>(data);
        auto input_size = data_training.trainset_idx().size();
        auto output_size = data_training.labels_idx().size();

        NNDescriptor layers_descriptor(_hidden_layers_descriptor);
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
                100, data_training, data_validation, data_testing,
                layers_descriptor,
                e,
                _default_setting.batch_size,
                _default_setting.learning_rate);
        }

        predict<ProfileCompileFNN>(
            "prediction after training with epochs amount: "
                + std::to_string(_default_setting.epochs),
            "prediction",
            100, data_training,
            layers_descriptor,
            _default_setting.epochs,
            _default_setting.batch_size,
            _default_setting.learning_rate);
    }

    /**
     * \brief Profile the training and the prediction phase of a FNN model
     * on different training set amount and fixed epoch amount.
     */
    void profile_on_training_set(ProfileDataset::Type pd_type) {
        auto data = ProfileDataset(pd_type).load_dataset();
        auto data_training = std::get<0>(data);
        auto data_validation = std::get<1>(data);
        auto data_testing = std::get<2>(data);
        auto training_set_size = data_training.size();
        auto input_size = data_training.trainset_idx().size();
        auto output_size = data_training.labels_idx().size();

        NNDescriptor layers_descriptor(_hidden_layers_descriptor);
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
            auto subdata_validation = data_validation.subdata(
                0, std::min(curr_size, data_validation.size()));
            auto subdata_testing = data_testing.subdata(
                0, std::min(curr_size, data_testing.size()));
            training_testing<ProfileCompileFNN>(
                "training and testing with dataset size (#entries): "
                    + std::to_string(curr_size),
                "training_testing_on_dataset_size" + std::to_string(curr_size),
                100, subdata_training, subdata_validation, subdata_testing,
                layers_descriptor,
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
    void profile_on_layers_amount(ProfileDataset::Type pd_type) {
        auto data = ProfileDataset(pd_type).load_dataset();
        auto data_training = std::get<0>(data);
        auto data_validation = std::get<1>(data);
        auto data_testing = std::get<2>(data);
        auto input_size = data_training.trainset_idx().size();
        auto output_size = data_training.labels_idx().size();

        NNDescriptor layers_descriptor(
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
                100, data_training, data_validation, data_testing,
                curr_ld,
                _default_setting.epochs,
                _default_setting.batch_size,
                _default_setting.learning_rate);

            predict<ProfileCompileFNN>(
                "prediction with hidden layers amount: "
                    + std::to_string(amount),
                "prediction_on_hidden_layers_amount" + std::to_string(amount),
                100, data_training,
                curr_ld, _default_setting.epochs, _default_setting.batch_size, _default_setting.learning_rate);
        }
    }

    /**
     * \brief Profile the training phase of a FNN model
     * on different hidden layer shape.
     */
    void profile_on_layers_shape(ProfileDataset::Type pd_type) {
        auto data = ProfileDataset(pd_type).load_dataset();
        auto data_training = std::get<0>(data);
        auto data_validation = std::get<1>(data);
        auto data_testing = std::get<2>(data);
        auto input_size = data_training.trainset_idx().size();
        auto output_size = data_training.labels_idx().size();

        std::vector<std::size_t> layers_shapes = {
            10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000
        };
        for (const auto& shape: layers_shapes)
        {
            NNDescriptor layers_descriptor(
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
                100, data_training, data_validation, data_testing,
                layers_descriptor,
                _default_setting.epochs,
                _default_setting.batch_size,
                _default_setting.learning_rate);

            predict<ProfileCompileFNN>(
                "prediction with hidden layers shape: " + std::to_string(shape),
                "prediction_on_hidden_layers_shape" + std::to_string(shape),
                100, data_training,
                layers_descriptor,
                _default_setting.epochs,
                _default_setting.batch_size,
                _default_setting.learning_rate);
        }
    }

    std::string _profile_name;
    std::vector<ProfileDataset::Type> _dataset_types;
    NNDescriptor _hidden_layers_descriptor;
    TrainingSetting _default_setting;
};

#endif // EDGE_LEARNING_PROFILE_FNN_HPP