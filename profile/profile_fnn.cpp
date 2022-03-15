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

struct ProfileRegressorFNN : Profiler
{
    ProfileRegressorFNN() : Profiler(100000) { }

    void run() {
        profile_on_epochs_amount();
        profile_on_training_set();
        profile_on_layers_amount();
        profile_on_layers_shape();
    }

private:
    const std::string DATA_TRAINING_FN = "execution-time.csv";
    const fs::path data_training_fp = fs::path(__FILE__).parent_path()
        / ".." / "data" / DATA_TRAINING_FN;

    void profile_on_epochs_amount() {
        const SizeType BATCH_SIZE    = 1;
        const SizeType EPOCHS        = 20;
        const NumType  LEARNING_RATE = 0.03;

        auto csv = CSV(data_training_fp);
        auto labels_idx = std::set<SizeType>{4};
        auto vec = csv.to_vec<NumType>();
        vec.erase(vec.begin());
        auto data = Dataset<NumType>(vec, csv.cols_size(), 1, labels_idx);
        auto input_size = data.trainset_idx().size();
        auto output_size = data.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  Activation::Linear },
                {"hidden_layer0", 10,          Activation::ReLU },
                {"hidden_layer1", 20,          Activation::ReLU },
                {"hidden_layer2", 25,          Activation::ReLU },
                {"hidden_layer3", 28,          Activation::ReLU },
                {"hidden_layer4", 30,          Activation::ReLU },
                {"hidden_layer5", 28,          Activation::ReLU },
                {"hidden_layer6", 25,          Activation::ReLU },
                {"hidden_layer7", 20,          Activation::ReLU },
                {"hidden_layer8", 10,          Activation::ReLU },
                {"output_layer",  output_size, Activation::Linear },
            }
        );

        profile("Increment the amount of epochs for regression",
                [&](SizeType i){
                    auto epochs = i + 1;
                    CompileFNN<LossType::MSE,
                        OptimizerType::GRADIENT_DESCENT,
                        InitType::AUTO>
                        m(layers_descriptor, "regressor_model");
                    m.fit(data, epochs, BATCH_SIZE, LEARNING_RATE);
                },
                EPOCHS);
    }

    void profile_on_training_set() {
        const SizeType BATCH_SIZE            = 1;
        const SizeType EPOCHS                = 20;
        const NumType  LEARNING_RATE         = 0.03;
        const SizeType MIN_TRAINING_SET_SIZE = 10;

        auto csv = CSV(data_training_fp);
        auto labels_idx = std::set<SizeType>{4};
        auto vec = csv.to_vec<NumType>();
        vec.erase(vec.begin());
        auto data = Dataset<NumType>(vec, csv.cols_size(), 1, labels_idx);
        auto training_set_size = data.size() - MIN_TRAINING_SET_SIZE;
        auto input_size = data.trainset_idx().size();
        auto output_size = data.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  Activation::Linear },
                {"hidden_layer0", 10,          Activation::ReLU },
                {"hidden_layer1", 20,          Activation::ReLU },
                {"hidden_layer2", 25,          Activation::ReLU },
                {"hidden_layer3", 28,          Activation::ReLU },
                {"hidden_layer4", 30,          Activation::ReLU },
                {"hidden_layer5", 28,          Activation::ReLU },
                {"hidden_layer6", 25,          Activation::ReLU },
                {"hidden_layer7", 20,          Activation::ReLU },
                {"hidden_layer8", 10,          Activation::ReLU },
                {"output_layer",  output_size, Activation::Linear },
            }
        );

        profile("Increment the training-set size for regression",
                [&](SizeType i){
                    CompileFNN<LossType::MSE,
                        OptimizerType::GRADIENT_DESCENT,
                        InitType::AUTO>
                        m(layers_descriptor, "regressor_model");
                    auto trainsubset = data.subdata(
                        0, MIN_TRAINING_SET_SIZE + i);
                    m.fit(trainsubset, EPOCHS, BATCH_SIZE, LEARNING_RATE);
                },
                training_set_size);
    }

    void profile_on_layers_amount() {
        const SizeType BATCH_SIZE    = 1;
        const SizeType EPOCHS        = 10;
        const SizeType LAYERS_AMOUNT = 100;
        const NumType  LEARNING_RATE = 0.03;

        auto csv = CSV(data_training_fp);
        auto labels_idx = std::set<SizeType>{4};
        auto vec = csv.to_vec<NumType>();
        vec.erase(vec.begin());
        auto data = Dataset<NumType>(vec, csv.cols_size(), 1, labels_idx);
        auto input_size = data.trainset_idx().size();
        auto output_size = data.labels_idx().size();

        LayerDescriptorVector layers_descriptor(
            {
                {"input_layer",   input_size,  Activation::Linear },
                {"output_layer",  output_size, Activation::Linear },
            }
        );

        profile("Increment the amount of FNN layers for regression",
                [&](SizeType i){
                    layers_descriptor.insert(
                        layers_descriptor.end() - 1,
                        {
                            "hidden_layer" + std::to_string(i),
                            input_size * 2,
                            Activation::ReLU
                        });
                    CompileFNN<LossType::MSE,
                        OptimizerType::GRADIENT_DESCENT,
                        InitType::AUTO>
                        m(layers_descriptor, "regressor_model");
                    m.fit(data, EPOCHS, BATCH_SIZE, LEARNING_RATE);
                },
                LAYERS_AMOUNT);
    }

    void profile_on_layers_shape() {
        const SizeType BATCH_SIZE    = 1;
        const NumType  LEARNING_RATE = 0.03;
        const SizeType LAYERS_MAX_SIZE = 100;

        auto csv = CSV(data_training_fp);
        auto labels_idx = std::set<SizeType>{4};
        auto vec = csv.to_vec<NumType>();
        vec.erase(vec.begin());
        auto data = Dataset<NumType>(vec, csv.cols_size(), 1, labels_idx);
        auto input_size = data.trainset_idx().size();
        auto output_size = data.labels_idx().size();

        profile("Increment the shape of layers for regression",
                [&](SizeType i){
                    LayerDescriptorVector layers_descriptor(
                        {
                            {"input_layer",   input_size,  Activation::Linear },
                            {
                                "hidden_layer0",
                                10 + static_cast<NumType>(i / 2),
                                Activation::ReLU
                            },
                            {"hidden_layer1", 10 + i,  Activation::ReLU   },
                            {
                                "hidden_layer2",
                                10 + static_cast<NumType>(i / 2),
                                Activation::ReLU
                            },
                            {"output_layer",  output_size, Activation::Linear },
                        }
                    );

                    CompileFNN<LossType::MSE,
                        OptimizerType::GRADIENT_DESCENT,
                        InitType::AUTO>
                        m(layers_descriptor, "regressor_model");
                    m.fit(data, 10, BATCH_SIZE, LEARNING_RATE);
                },
                LAYERS_MAX_SIZE);
    }

};

int main() {
    ProfileRegressorFNN().run();
}