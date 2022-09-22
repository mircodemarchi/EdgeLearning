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


template <OptimizerType OT>
class ProfileFNNRegression : public ProfileFNN<LossType::MSE, OT>
{
public:
    ProfileFNNRegression(
        std::vector<ProfileDataset::Type> dataset_types)
        : ProfileFNN<LossType::MSE, OT>("regression", dataset_types)
    { }
};

int main() {
    std::vector<ProfileDataset::Type> dataset_types({
        ProfileDataset::Type::CSV_EXECUTION_TIME,
    });
    ProfileFNNRegression<OptimizerType::GRADIENT_DESCENT>(dataset_types)
        .run();
}