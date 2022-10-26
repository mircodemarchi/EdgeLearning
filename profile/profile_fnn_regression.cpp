/***************************************************************************
 *            profile_fnn_regression.cpp
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

#include "profile_fnn.hpp"


const NeuralNetworkDescriptor execution_time_hidden_layers_descriptor(
    {
        Dense{"hidden_layer0", 32, ActivationType::ReLU },
    }
);

template <OptimizerType OT>
class ProfileFNNRegression : public ProfileFNN<LossType::MSE, OT>
{
public:
    ProfileFNNRegression(
        ProfileDataset::Type dataset_type,
        std::vector<NeuralNetworkDescriptor> hidden_layers_descriptor_vec,
        ProfileNN::TrainingSetting default_setting)
        : ProfileFNN<LossType::MSE, OT>(
            "regression",
            dataset_type,
            hidden_layers_descriptor_vec,
            default_setting)
    { }
};

int main() {
    SizeType EPOCHS = 20;
    SizeType BATCH_SIZE = 128;
    NumType LEARNING_RATE = 0.01;

    ProfileFNNRegression<OptimizerType::GRADIENT_DESCENT>(
        ProfileDataset::Type::CSV_EXECUTION_TIME,
        {execution_time_hidden_layers_descriptor},
        {EPOCHS, BATCH_SIZE, LEARNING_RATE}).run();
}