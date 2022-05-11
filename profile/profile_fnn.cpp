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
#include "middleware/fnn.hpp"

#include <filesystem>


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
        EDGE_LEARNING_PROFILE_CALL(profile_on_training_set());
        EDGE_LEARNING_PROFILE_CALL(profile_on_layers_amount());
        EDGE_LEARNING_PROFILE_CALL(profile_on_layers_shape());
    }

private:
    const std::string DATA_TRAINING_FN = "execution-time.csv";
    const fs::path DATA_TRAINING_FP = fs::path(__FILE__).parent_path()
                                      / ".." / "data" / DATA_TRAINING_FN;

    /**
     * \brief Profile the training and the prediction phase of a FNN model to
     * solve a regression problem on epochs incrementation.
     */
    void profile_on_epochs_amount() {
        const SizeType BATCH_SIZE    = 1;
        const SizeType EPOCHS        = 20;
        const NumType  LEARNING_RATE = 0.03;

        auto csv = CSV(DATA_TRAINING_FP.string());
        auto labels_idx = std::set<SizeType>{4};
        auto vec = csv.to_vec<NumType>();
        vec.erase(vec.begin());
        auto data = Dataset<NumType>(vec, csv.cols_size(), 1, labels_idx);
        auto input_size = data.trainset_idx().size();
        auto output_size = data.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  ActivationType::Linear },
                {"hidden_layer0", 10,          ActivationType::ReLU },
                {"hidden_layer1", 20,          ActivationType::ReLU },
                {"hidden_layer2", 25,          ActivationType::ReLU },
                {"hidden_layer3", 28,          ActivationType::ReLU },
                {"hidden_layer4", 30,          ActivationType::ReLU },
                {"hidden_layer5", 28,          ActivationType::ReLU },
                {"hidden_layer6", 25,          ActivationType::ReLU },
                {"hidden_layer7", 20,          ActivationType::ReLU },
                {"hidden_layer8", 10,          ActivationType::ReLU },
                {"output_layer",  output_size, ActivationType::Linear },
            }
        );

        for (SizeType e = 1; e < EPOCHS; ++e)
        {
            profile("training epochs amount: " + std::to_string(e),
                    [&](SizeType i){
                        (void) i;
                        CompileFNN<LossType::MSE,
                            OptimizerType::GRADIENT_DESCENT,
                            InitType::AUTO>
                            m(layers_descriptor, "regressor_model");
                        m.fit(data, e, BATCH_SIZE, LEARNING_RATE);
                    }, 100, "training_on_epochs_amount" + std::to_string(e));
        }

        CompileFNN<LossType::MSE,
            OptimizerType::GRADIENT_DESCENT,
            InitType::AUTO>
            m(layers_descriptor, "regressor_model");
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
        const SizeType MIN_SET_SIZE = 10;
        const SizeType MAX_SET_SIZE = 1000;

        auto csv = CSV(DATA_TRAINING_FP.string());
        auto labels_idx = std::set<SizeType>{4};
        auto vec = csv.to_vec<NumType>();
        vec.erase(vec.begin());
        auto data = Dataset<NumType>(vec, csv.cols_size(), 1, labels_idx);
        auto training_set_size = data.size() - MIN_SET_SIZE;
        auto input_size = data.trainset_idx().size();
        auto output_size = data.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  ActivationType::Linear },
                {"hidden_layer0", 10,          ActivationType::ReLU },
                {"hidden_layer1", 20,          ActivationType::ReLU },
                {"hidden_layer2", 25,          ActivationType::ReLU },
                {"hidden_layer3", 28,          ActivationType::ReLU },
                {"hidden_layer4", 30,          ActivationType::ReLU },
                {"hidden_layer5", 28,          ActivationType::ReLU },
                {"hidden_layer6", 25,          ActivationType::ReLU },
                {"hidden_layer7", 20,          ActivationType::ReLU },
                {"hidden_layer8", 10,          ActivationType::ReLU },
                {"output_layer",  output_size, ActivationType::Linear },
            }
        );

        auto max_size_training_set = std::min(
            training_set_size, MAX_SET_SIZE);
        for (std::int64_t div_factor = 10; div_factor >= 0; div_factor-=2)
        {
            auto curr_size = MIN_SET_SIZE
                + max_size_training_set / static_cast<std::size_t>(
                    div_factor + 1);
            profile("training with dataset size (#entries): "
                        + std::to_string(curr_size),
                    [&](SizeType i) {
                        (void) i;
                        CompileFNN<LossType::MSE,
                            OptimizerType::GRADIENT_DESCENT,
                            InitType::AUTO>
                            m(layers_descriptor, "regressor_model");
                        auto subset = data.subdata(0, curr_size);
                        m.fit(subset, EPOCHS, BATCH_SIZE, LEARNING_RATE);
                    }, 100,
                    "training_on_dataset_size" + std::to_string(curr_size));
        }

        CompileFNN<LossType::MSE,
            OptimizerType::GRADIENT_DESCENT,
            InitType::AUTO>
            m(layers_descriptor, "regressor_model");
        m.fit(data, EPOCHS, BATCH_SIZE, LEARNING_RATE);
        for (std::int64_t div_factor = 10; div_factor >= 0; div_factor-=2)
        {
            auto curr_size = MIN_SET_SIZE
                + max_size_training_set / static_cast<std::size_t>(
                    div_factor + 1);
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
        }
    }

    /**
     * \brief Profile the training and the prediction phase of a FNN model to
     * solve a regression problem on incremental layers amount.
     */
    void profile_on_layers_amount() {
        const SizeType BATCH_SIZE    = 1;
        const SizeType EPOCHS        = 5;
        const SizeType LAYERS_AMOUNT = 15;
        const NumType  LEARNING_RATE = 0.03;

        auto csv = CSV(DATA_TRAINING_FP.string());
        auto labels_idx = std::set<SizeType>{4};
        auto vec = csv.to_vec<NumType>();
        vec.erase(vec.begin());
        auto data = Dataset<NumType>(vec, csv.cols_size(), 1, labels_idx);
        auto input_size = data.trainset_idx().size();
        auto output_size = data.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  ActivationType::Linear },
                {"output_layer",  output_size, ActivationType::Linear },
            }
        );

        for (std::size_t amount = 0; amount < LAYERS_AMOUNT; ++amount)
        {
            layers_descriptor.insert(
                layers_descriptor.end() - 1,
                {
                    "hidden_layer" + std::to_string(amount),
                    input_size * 2,
                    ActivationType::ReLU
                });
            profile("training with hidden layers amount: "
                        + std::to_string(amount),
                    [&](SizeType i){
                        (void) i;
                        CompileFNN<LossType::MSE,
                            OptimizerType::GRADIENT_DESCENT,
                            InitType::AUTO>
                            m(layers_descriptor, "regressor_model");
                        m.fit(data, EPOCHS, BATCH_SIZE, LEARNING_RATE);
                    }, 100,
                    "training_on_hidden_layers_amount"
                        + std::to_string(amount));
            CompileFNN<LossType::MSE,
                OptimizerType::GRADIENT_DESCENT,
                InitType::AUTO>
                m(layers_descriptor, "regressor_model");
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

        auto csv = CSV(DATA_TRAINING_FP.string());
        auto labels_idx = std::set<SizeType>{4};
        auto vec = csv.to_vec<NumType>();
        vec.erase(vec.begin());
        auto data = Dataset<NumType>(vec, csv.cols_size(), 1, labels_idx);
        auto input_size = data.trainset_idx().size();
        auto output_size = data.labels_idx().size();

        for (std::size_t shape = 10; shape < LAYERS_MAX_SIZE; ++shape)
        {
            LayerDescriptorVector layers_descriptor(
                {
                    {"input_layer", input_size, ActivationType::Linear},
                    {
                        "hidden_layer0",
                        static_cast<SizeType>(shape / 2),
                        ActivationType::ReLU
                    },
                    {"hidden_layer1", shape, ActivationType::ReLU},
                    {
                        "hidden_layer2",
                        static_cast<SizeType>(shape / 2),
                        ActivationType::ReLU
                    },
                    {"output_layer", output_size, ActivationType::Linear},
                }
            );
            profile("training with hidden layers shape: "
                        + std::to_string(shape),
                    [&](SizeType i){
                        (void) i;
                        CompileFNN<LossType::MSE,
                            OptimizerType::GRADIENT_DESCENT,
                            InitType::AUTO>
                            m(layers_descriptor, "regressor_model");
                        m.fit(data, EPOCHS, BATCH_SIZE, LEARNING_RATE);
                    }, 100,
                    "training_on_hidden_layers_shape" + std::to_string(shape));
            CompileFNN<LossType::MSE,
                OptimizerType::GRADIENT_DESCENT,
                InitType::AUTO>
                m(layers_descriptor, "regressor_model");
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