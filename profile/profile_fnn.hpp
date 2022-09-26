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


template <LossType LT, OptimizerType OT>
class ProfileFNN : public DNNProfiler
{
public:
    ProfileFNN(std::string profile_name,
               std::vector<ProfileDataset::Type> dataset_types)
        : DNNProfiler(
            100,
#if ENABLE_MLPACK
            "profile_mlpack_fnn_" + profile_name
#else
            "profile_edgelearning_fnn_" + profile_name
#endif
            )
        , _profile_name(profile_name)
        , _dataset_types(dataset_types)
    { }

    virtual void run() {
        EDGE_LEARNING_PROFILE_TITLE(
            "FNN training and prediction process when "
            "solving a " + _profile_name + " problem");
        for (const auto& dt: _dataset_types)
        {
            std::cout << "*** Dataset: " + std::string(ProfileDataset(dt))
                      << " ***" << std::endl;
            EDGE_LEARNING_PROFILE_CALL(profile_on_epochs_amount(dt));
            return;
            EDGE_LEARNING_PROFILE_CALL(profile_on_parallelism_level(dt));
            EDGE_LEARNING_PROFILE_CALL(profile_on_training_set(dt));
            EDGE_LEARNING_PROFILE_CALL(profile_on_layers_amount(dt));
            EDGE_LEARNING_PROFILE_CALL(profile_on_layers_shape(dt));
        }
    }

private:

    using ProfileCompileFNN = CompileFNN<
        LT,
        OT,
        InitType::AUTO,
        ParallelizationLevel::SEQUENTIAL>;

    void profile_on_parallelism_level(ProfileDataset::Type pd_type) {
        const SizeType EPOCHS        = 1;
        const NumType  LEARNING_RATE = 0.03;

        auto data = ProfileDataset(pd_type).load_dataset();
        auto data_training = std::get<0>(data);
        auto data_validation = std::get<1>(data);
        auto data_testing = std::get<2>(data);
        auto input_size = data_training.trainset_idx().size();
        auto output_size = data_training.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  ActivationType::Linear },
                {"hidden_layer0", 32,          ActivationType::ReLU },
                {"output_layer",  output_size,
                 _profile_name == "classification" ?
                 ActivationType::Softmax : ActivationType::Linear },
            }
        );

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
            training<ProfileCompileFNNSequential>(
                "training sequential model with batch_size: "
                    + std::to_string(batch_size),
                "training_sequential_on_batch_size"
                    + std::to_string(batch_size),
                1, data_training, data_validation,
                layers_descriptor, EPOCHS, batch_size, LEARNING_RATE);

            testing<ProfileCompileFNNSequential>(
                "testing sequential model with batch_size: "
                + std::to_string(batch_size),
                "testing_sequential_on_batch_size"
                + std::to_string(batch_size),
                1, data_training, data_testing,
                layers_descriptor, EPOCHS, batch_size, LEARNING_RATE);
        }

        for (const auto& batch_size: batch_sizes)
        {
            training<ProfileCompileFNNThreadOnEntry>(
                "training thread parallelism on data entry model with batch_size: "
                    + std::to_string(batch_size),
                "training_thread_parallelism_entry_on_batch_size"
                    + std::to_string(batch_size),
                1, data_training, data_validation,
                layers_descriptor, EPOCHS, batch_size, LEARNING_RATE);

            testing<ProfileCompileFNNThreadOnEntry>(
                "testing thread parallelism on data entry model with batch_size: "
                    + std::to_string(batch_size),
                "testing_thread_parallelism_entry_on_batch_size"
                    + std::to_string(batch_size),
                1, data_training, data_testing,
                layers_descriptor, EPOCHS, batch_size, LEARNING_RATE);
        }

        for (const auto& batch_size: batch_sizes)
        {
            training<ProfileCompileFNNThreadOnBatch>(
                "training thread parallelism on data batch model with batch_size: "
                    + std::to_string(batch_size),
                "training_thread_parallelism_batch_on_batch_size"
                    + std::to_string(batch_size),
                1, data_training, data_validation,
                layers_descriptor, EPOCHS, batch_size, LEARNING_RATE);

            testing<ProfileCompileFNNThreadOnBatch>(
                "testing thread parallelism on data batch model with batch_size: "
                    + std::to_string(batch_size),
                "testing_thread_parallelism_batch_on_batch_size"
                    + std::to_string(batch_size),
                1, data_training, data_testing,
                layers_descriptor, EPOCHS, batch_size, LEARNING_RATE);
        }
    }

    /**
     * \brief Profile the training and the prediction phase of a FNN model
     * on epochs incrementation.
     */
    void profile_on_epochs_amount(ProfileDataset::Type pd_type) {
        const SizeType BATCH_SIZE    = 1;
        const SizeType EPOCHS        = 20;
        const NumType  LEARNING_RATE = 0.03;

        auto data = ProfileDataset(pd_type).load_dataset();
        auto data_training = std::get<0>(data);
        auto data_validation = std::get<1>(data);
        auto data_testing = std::get<2>(data);
        auto input_size = data_training.trainset_idx().size();
        auto output_size = data_training.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  ActivationType::Linear },
                {"hidden_layer0", 32,          ActivationType::ReLU },
                {"output_layer",  output_size,
                 _profile_name == "classification" ?
                 ActivationType::Softmax : ActivationType::Linear },
            }
        );

        for (SizeType e = 1; e <= EPOCHS; ++e)
        {
            training<ProfileCompileFNN>(
                "training epochs amount: " + std::to_string(e),
                "training_on_epochs_amount" + std::to_string(e),
                1, data_training, data_validation,
                layers_descriptor, e, BATCH_SIZE, LEARNING_RATE);
        }

        predict<ProfileCompileFNN>(
            "prediction after training with epochs amount: "
                + std::to_string(EPOCHS),
            "prediction",
            100, data_training,
            layers_descriptor, EPOCHS, BATCH_SIZE, LEARNING_RATE);

        testing<ProfileCompileFNN>(
            "testing after training with epochs amount: "
                + std::to_string(EPOCHS),
            "testing",
            1, data_training, data_testing,
            layers_descriptor, EPOCHS, BATCH_SIZE, LEARNING_RATE);
    }

    /**
     * \brief Profile the training and the prediction phase of a FNN model
     * on different training set amount and fixed epoch amount.
     */
    void profile_on_training_set(ProfileDataset::Type pd_type) {
        const SizeType BATCH_SIZE            = 1;
        const SizeType EPOCHS                = 5;
        const NumType  LEARNING_RATE         = 0.03;

        auto data = ProfileDataset(pd_type).load_dataset();
        auto data_training = std::get<0>(data);
        auto data_validation = std::get<1>(data);
        auto data_testing = std::get<2>(data);
        auto training_set_size = data_training.size();
        auto input_size = data_training.trainset_idx().size();
        auto output_size = data_training.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  ActivationType::Linear },
                {"hidden_layer0", 32,          ActivationType::ReLU },
                {"output_layer",  output_size,
                 _profile_name == "classification" ?
                 ActivationType::Softmax : ActivationType::Linear },
            }
        );

        std::vector<std::size_t> training_set_sizes = {
            10, 50, 100, 200, 300, 400, 600, 800, 1000, 10000
        };
        for (const auto& size: training_set_sizes)
        {
            auto curr_size = std::min(training_set_size, size);
            auto subset = data_training.subdata(0, curr_size);
            training<ProfileCompileFNN>(
                "training with dataset size (#entries): " + std::to_string(curr_size),
                "training_on_dataset_size" + std::to_string(curr_size),
                100, subset, data_validation,
                layers_descriptor, EPOCHS, BATCH_SIZE, LEARNING_RATE);
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
                layers_descriptor, EPOCHS, BATCH_SIZE, LEARNING_RATE);
            if (curr_size == training_set_size) break;
        }

        for (const auto& size: training_set_sizes)
        {
            auto curr_size = std::min(training_set_size, size);
            auto subset = data_training.subdata(0, curr_size);
            testing<ProfileCompileFNN>(
                "testing with dataset size (#entries): "
                + std::to_string(curr_size),
                "testing_on_dataset_size" + std::to_string(curr_size),
                1, subset, data_testing,
                layers_descriptor, EPOCHS, BATCH_SIZE, LEARNING_RATE);
            if (curr_size == training_set_size) break;
        }
    }

    /**
     * \brief Profile the training and the prediction phase of a FNN model
     * on incremental layers amount.
     */
    void profile_on_layers_amount(ProfileDataset::Type pd_type) {
        const SizeType BATCH_SIZE    = 1;
        const SizeType EPOCHS        = 5;
        const SizeType LAYERS_AMOUNT = 10;
        const NumType  LEARNING_RATE = 0.03;

        auto data = ProfileDataset(pd_type).load_dataset();
        auto data_training = std::get<0>(data);
        auto data_validation = std::get<1>(data);
        auto data_testing = std::get<2>(data);
        auto input_size = data_training.trainset_idx().size();
        auto output_size = data_training.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  ActivationType::Linear },
                {"output_layer",  output_size,
                 _profile_name == "classification" ?
                 ActivationType::Softmax : ActivationType::Linear },
            }
        );

        for (std::size_t amount = 1; amount <= LAYERS_AMOUNT; ++amount)
        {
            layers_descriptor.insert(
                layers_descriptor.end() - 1,
                {
                    "hidden_layer" + std::to_string(amount-1),
                    15,
                    ActivationType::ReLU
                });
            training<ProfileCompileFNN>(
                "training with hidden layers amount: " + std::to_string(amount),
                "training_on_hidden_layers_amount" + std::to_string(amount),
                100, data_training, data_validation,
                layers_descriptor, EPOCHS, BATCH_SIZE, LEARNING_RATE);

            predict<ProfileCompileFNN>(
                "prediction with hidden layers amount: "
                    + std::to_string(amount),
                "prediction_on_hidden_layers_amount" + std::to_string(amount),
                100, data_training,
                layers_descriptor, EPOCHS, BATCH_SIZE, LEARNING_RATE);

            testing<ProfileCompileFNN>(
                "testing with hidden layers amount: " + std::to_string(amount),
                "testing_on_hidden_layers_amount" + std::to_string(amount),
                1, data_training, data_testing,
                layers_descriptor, EPOCHS, BATCH_SIZE, LEARNING_RATE);
        }
    }

    /**
     * \brief Profile the training phase of a FNN model
     * on different hidden layer shape.
     */
    void profile_on_layers_shape(ProfileDataset::Type pd_type) {
        const SizeType BATCH_SIZE    = 1;
        const NumType  LEARNING_RATE = 0.03;
        const SizeType EPOCHS        = 5;
        const SizeType LAYERS_MAX_SIZE = 20;

        auto data = ProfileDataset(pd_type).load_dataset();
        auto data_training = std::get<0>(data);
        auto data_validation = std::get<1>(data);
        auto data_testing = std::get<2>(data);
        auto input_size = data_training.trainset_idx().size();
        auto output_size = data_training.labels_idx().size();

        for (std::size_t shape = 10; shape <= LAYERS_MAX_SIZE; ++shape)
        {
            LayerDescriptorVector layers_descriptor(
                {
                    {"input_layer", input_size, ActivationType::Linear},
                    {"hidden_layer0", shape, ActivationType::ReLU},
                    {"output_layer", output_size,
                     _profile_name == "classification" ?
                     ActivationType::Softmax : ActivationType::Linear},
                }
            );
            training<ProfileCompileFNN>(
                "training with hidden layers shape: " + std::to_string(shape),
                "training_on_hidden_layers_shape" + std::to_string(shape),
                100, data_training, data_validation,
                layers_descriptor, EPOCHS, BATCH_SIZE, LEARNING_RATE);

            predict<ProfileCompileFNN>(
                "prediction with hidden layers shape: " + std::to_string(shape),
                "prediction_on_hidden_layers_shape" + std::to_string(shape),
                100, data_training,
                layers_descriptor, EPOCHS, BATCH_SIZE, LEARNING_RATE);

            testing<ProfileCompileFNN>(
                "testing with hidden layers shape: " + std::to_string(shape),
                "testing_on_hidden_layers_shape" + std::to_string(shape),
                1, data_training, data_testing,
                layers_descriptor, EPOCHS, BATCH_SIZE, LEARNING_RATE);
        }
    }

    std::string _profile_name;
    std::vector<ProfileDataset::Type> _dataset_types;
};

#endif // EDGE_LEARNING_PROFILE_FNN_HPP