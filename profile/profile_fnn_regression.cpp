/***************************************************************************
 *            profile_fnn.cpp
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

#include "profile.hpp"

#include "parser/csv.hpp"
#include "data/dataset.hpp"
#include "data/path.hpp"
#include "middleware/fnn.hpp"


using namespace EdgeLearning;
namespace fs = std::filesystem;

struct ProfileRegressionFNN : Profiler
{
    ProfileRegressionFNN()
        : Profiler(
            100,
#if ENABLE_MLPACK
            "profile_mlpack_fnn_regression"
#else
            "profile_edgelearning_fnn_regression"
#endif
            )
    {
    }

    void run() {
        EDGE_LEARNING_PROFILE_TITLE("FNN training and prediction process when "
                                    "solving a Regression problem");
        EDGE_LEARNING_PROFILE_CALL(profile_on_epochs_amount());
        EDGE_LEARNING_PROFILE_CALL(profile_on_parallelism_level());
        EDGE_LEARNING_PROFILE_CALL(profile_on_training_set());
        EDGE_LEARNING_PROFILE_CALL(profile_on_layers_amount());
        EDGE_LEARNING_PROFILE_CALL(profile_on_layers_shape());
    }

private:
    const std::string DATA_TRAINING_FN = "execution-time.csv";
    const std::filesystem::path DATA_TRAINING_FP = std::filesystem::path(__FILE__).parent_path()
                                      / ".." / "data" / DATA_TRAINING_FN;

    using ProfileCompileFNN = CompileFNN<
        LossType::MSE,
        OptimizerType::GRADIENT_DESCENT,
        InitType::AUTO,
        ParallelizationLevel::SEQUENTIAL>;

    void profile_on_parallelism_level() {
        const SizeType EPOCHS        = 1;
        const NumType  LEARNING_RATE = 0.03;

        auto csv = CSV(
            DATA_TRAINING_FP.string(),
            { Type::AUTO },
            ',', {4});
        auto data = Dataset<NumType>::parse(csv);
        auto input_size = data.trainset_idx().size();
        auto output_size = data.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  ActivationType::Linear },
                {"hidden_layer0", 15,          ActivationType::ReLU },
                {"hidden_layer1", 15,          ActivationType::ReLU },
                {"hidden_layer2", 15,          ActivationType::ReLU },
                {"hidden_layer3", 15,          ActivationType::ReLU },
                {"hidden_layer4", 15,          ActivationType::ReLU },
                {"hidden_layer5", 15,          ActivationType::ReLU },
                {"output_layer",  output_size, ActivationType::Linear },
            }
        );

        std::vector<SizeType> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128};

        using ProfileCompileFNNSequential = CompileFNN<
            LossType::MSE,
            OptimizerType::GRADIENT_DESCENT,
            InitType::AUTO,
            ParallelizationLevel::SEQUENTIAL>;

        using ProfileCompileFNNThreadOnEntry = CompileFNN<
            LossType::MSE,
            OptimizerType::GRADIENT_DESCENT,
            InitType::AUTO,
            ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_ENTRY>;

        using ProfileCompileFNNThreadOnBatch = CompileFNN<
            LossType::MSE,
            OptimizerType::GRADIENT_DESCENT,
            InitType::AUTO,
            ParallelizationLevel::THREAD_PARALLELISM_ON_DATA_BATCH>;

        for (const auto& batch_size: batch_sizes)
        {
            profile(
                "training sequential model with batch_size: " + std::to_string(batch_size),
                [&](SizeType i){
                    (void) i;
                    ProfileCompileFNNSequential m(
                        layers_descriptor, "regressor_model");
                    m.fit(data, EPOCHS, batch_size, LEARNING_RATE);
                }, 20, "training_sequential_on_batch_size" + std::to_string(batch_size));
        }

        for (const auto& batch_size: batch_sizes)
        {
            profile(
                "training thread parallelism on data entry model with batch_size: " + std::to_string(batch_size),
                [&](SizeType i){
                    (void) i;
                    ProfileCompileFNNThreadOnEntry m(
                        layers_descriptor, "regressor_model");
                    m.fit(data, EPOCHS, batch_size, LEARNING_RATE);
                }, 20, "training_thread_parallelism_entry_on_batch_size" + std::to_string(batch_size));
        }

        for (const auto& batch_size: batch_sizes)
        {
            profile(
                "training thread parallelism on data batch model with batch_size: " + std::to_string(batch_size),
                [&](SizeType i){
                    (void) i;
                    ProfileCompileFNNThreadOnBatch m(
                        layers_descriptor, "regressor_model");
                    m.fit(data, EPOCHS, batch_size, LEARNING_RATE);
                }, 20, "training_thread_parallelism_batch_on_batch_size" + std::to_string(batch_size));
        }
    }

    /**
     * \brief Profile the training and the prediction phase of a FNN model to
     * solve a regression problem on epochs incrementation.
     */
    void profile_on_epochs_amount() {
        const SizeType BATCH_SIZE    = 1;
        const SizeType EPOCHS        = 20;
        const NumType  LEARNING_RATE = 0.03;

        auto csv = CSV(
            DATA_TRAINING_FP.string(),
            { Type::AUTO },
            ',', {4});
        auto data = Dataset<NumType>::parse(csv);
        auto input_size = data.trainset_idx().size();
        auto output_size = data.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  ActivationType::Linear },
                {"hidden_layer0", 15,          ActivationType::ReLU },
                {"hidden_layer1", 15,          ActivationType::ReLU },
                {"hidden_layer2", 15,          ActivationType::ReLU },
                {"hidden_layer3", 15,          ActivationType::ReLU },
                {"hidden_layer4", 15,          ActivationType::ReLU },
                {"hidden_layer5", 15,          ActivationType::ReLU },
                {"output_layer",  output_size, ActivationType::Linear },
            }
        );

        for (SizeType e = 1; e <= EPOCHS; ++e)
        {
            profile("training epochs amount: " + std::to_string(e),
                    [&](SizeType i){
                        (void) i;
                        ProfileCompileFNN m(
                            layers_descriptor, "regressor_model");
                        m.fit(data, e, BATCH_SIZE, LEARNING_RATE);
                    }, 100, "training_on_epochs_amount" + std::to_string(e));
        }

        ProfileCompileFNN m(layers_descriptor, "regressor_model");
        m.fit(data, EPOCHS, BATCH_SIZE, LEARNING_RATE);
        profile("prediction after training with epochs amount: "
                    + std::to_string(EPOCHS),
                [&](SizeType i){
                    (void) i;
                    auto input = data.trainset();
                    auto prediction = m.predict(input);
                    (void) prediction;
                }, 100, "prediction");
    }

    /**
     * \brief Profile the training and the prediction phase of a FNN model to
     * solve a regression problem on different training set amount and fixed
     * epoch amount.
     */
    void profile_on_training_set() {
        const SizeType BATCH_SIZE            = 1;
        const SizeType EPOCHS                = 5;
        const NumType  LEARNING_RATE         = 0.03;

        auto csv = CSV(
            DATA_TRAINING_FP.string(),
            { Type::AUTO },
            ',', {4});
        auto data = Dataset<NumType>::parse(csv);
        auto training_set_size = data.size();
        auto input_size = data.trainset_idx().size();
        auto output_size = data.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  ActivationType::Linear },
                {"hidden_layer0", 15,          ActivationType::ReLU },
                {"hidden_layer1", 15,          ActivationType::ReLU },
                {"hidden_layer2", 15,          ActivationType::ReLU },
                {"hidden_layer3", 15,          ActivationType::ReLU },
                {"hidden_layer4", 15,          ActivationType::ReLU },
                {"hidden_layer5", 15,          ActivationType::ReLU },
                {"output_layer",  output_size, ActivationType::Linear },
            }
        );

        std::vector<std::size_t> training_set_sizes = {
            10, 50, 100, 200, 300, 400, 600, 800, 1000, 10000
        };
        for (const auto& size: training_set_sizes)
        {
            auto curr_size = std::min(training_set_size, size);
            profile("training with dataset size (#entries): "
                        + std::to_string(curr_size),
                    [&](SizeType i) {
                        (void) i;
                        ProfileCompileFNN m(
                            layers_descriptor, "regressor_model");
                        auto subset = data.subdata(0, curr_size);
                        m.fit(subset, EPOCHS, BATCH_SIZE, LEARNING_RATE);
                    }, 100,
                    "training_on_dataset_size" + std::to_string(curr_size));
            if (curr_size == training_set_size) break;
        }

        ProfileCompileFNN m(layers_descriptor, "regressor_model");
        m.fit(data, EPOCHS, BATCH_SIZE, LEARNING_RATE);
        for (const auto& size: training_set_sizes)
        {
            auto curr_size = std::min(training_set_size, size);
            profile("prediction with dataset size (#entries): "
                        + std::to_string(curr_size),
                    [&](SizeType i) {
                        (void) i;
                        auto input = data.subdata(0, curr_size)
                            .trainset();
                        auto prediction = m.predict(input);
                        (void) prediction;
                    },
                    100,
                    "prediction_on_dataset_size" + std::to_string(curr_size));
            if (curr_size == training_set_size) break;
        }
    }

    /**
     * \brief Profile the training and the prediction phase of a FNN model to
     * solve a regression problem on incremental layers amount.
     */
    void profile_on_layers_amount() {
        const SizeType BATCH_SIZE    = 1;
        const SizeType EPOCHS        = 5;
        const SizeType LAYERS_AMOUNT = 10;
        const NumType  LEARNING_RATE = 0.03;

        auto csv = CSV(
            DATA_TRAINING_FP.string(),
            { Type::AUTO },
            ',', {4});
        auto data = Dataset<NumType>::parse(csv);
        auto input_size = data.trainset_idx().size();
        auto output_size = data.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  ActivationType::Linear },
                {"output_layer",  output_size, ActivationType::Linear },
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
            profile("training with hidden layers amount: "
                        + std::to_string(amount),
                    [&](SizeType i){
                        (void) i;
                        ProfileCompileFNN m(
                            layers_descriptor, "regressor_model");
                        m.fit(data, EPOCHS, BATCH_SIZE, LEARNING_RATE);
                    }, 100,
                    "training_on_hidden_layers_amount"
                        + std::to_string(amount));
            ProfileCompileFNN m(layers_descriptor, "regressor_model");
            m.fit(data, EPOCHS, BATCH_SIZE, LEARNING_RATE);
            profile("prediction with hidden layers amount: "
                        + std::to_string(amount),
                    [&](SizeType i){
                        (void) i;
                        auto input = data.trainset();
                        auto prediction = m.predict(input);
                        (void) prediction;
                    }, 100,
                    "prediction_on_hidden_layers_amount"
                        + std::to_string(amount));
        }
    }

    /**
     * \brief Profile the training phase of a FNN model to solve a regression
     * problem on different hidden layer shape.
     */
    void profile_on_layers_shape() {
        const SizeType BATCH_SIZE    = 1;
        const NumType  LEARNING_RATE = 0.03;
        const SizeType EPOCHS        = 5;
        const SizeType LAYERS_MAX_SIZE = 20;

        auto csv = CSV(
            DATA_TRAINING_FP.string(),
            { Type::AUTO },
            ',', {4});
        auto data = Dataset<NumType>::parse(csv);
        auto input_size = data.trainset_idx().size();
        auto output_size = data.labels_idx().size();

        for (std::size_t shape = 10; shape <= LAYERS_MAX_SIZE; ++shape)
        {
            LayerDescriptorVector layers_descriptor(
                {
                    {"input_layer", input_size, ActivationType::Linear},
                    {"hidden_layer0", shape, ActivationType::ReLU},
                    {"hidden_layer1", shape, ActivationType::ReLU},
                    {"hidden_layer2", shape, ActivationType::ReLU},
                    {"hidden_layer3", shape, ActivationType::ReLU},
                    {"hidden_layer4", shape, ActivationType::ReLU},
                    {"hidden_layer5", shape, ActivationType::ReLU},
                    {"output_layer", output_size, ActivationType::Linear},
                }
            );
            profile("training with hidden layers shape: "
                        + std::to_string(shape),
                    [&](SizeType i){
                        (void) i;
                        ProfileCompileFNN m(
                            layers_descriptor, "regressor_model");
                        m.fit(data, EPOCHS, BATCH_SIZE, LEARNING_RATE);
                    }, 100,
                    "training_on_hidden_layers_shape" + std::to_string(shape));
            ProfileCompileFNN m(layers_descriptor, "regressor_model");
            m.fit(data, EPOCHS, BATCH_SIZE, LEARNING_RATE);
            profile("prediction with hidden layers shape: "
                        + std::to_string(shape),
                    [&](SizeType i){
                        (void) i;
                        auto input = data.trainset();
                        auto prediction = m.predict(input);
                        (void) prediction;
                    }, 100,
                    "prediction_on_hidden_layers_shape"
                        + std::to_string(shape));
        }
    }
};

int main() {
    ProfileRegressionFNN().run();
}