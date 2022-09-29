/***************************************************************************
 *            profile_fnn_classification.cpp
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


const NNDescriptor mnist_hidden_layers_descriptor(
    {
        Conv{"hidden_layer0", {32, {3,3}}, ActivationType::ReLU },
    }
);

template <OptimizerType OT>
class ProfileFNNClassification : public ProfileFNN<LossType::CCE, OT>
{
public:
    ProfileFNNClassification(
        std::vector<ProfileDataset::Type> dataset_types,
        NNDescriptor hidden_layers_descriptor,
        ProfileNN::TrainingSetting default_setting)
        : ProfileFNN<LossType::CCE, OT>(
            "classification",
            dataset_types,
            hidden_layers_descriptor,
            default_setting)
    { }
};

int main() {
    SizeType EPOCHS = 20;
    SizeType BATCH_SIZE = 128;
    NumType LEARNING_RATE = 0.01;

    std::vector<ProfileDataset::Type> dataset_types({
        ProfileDataset::Type::MNIST,
    });
    ProfileFNNClassification<OptimizerType::GRADIENT_DESCENT>(
        dataset_types,
        mnist_hidden_layers_descriptor,
        {EPOCHS, BATCH_SIZE, LEARNING_RATE}).run();
}