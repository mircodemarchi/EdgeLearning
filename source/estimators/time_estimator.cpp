/***************************************************************************
 *            time_estimator.cpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne.
 *
 *  Ariadne is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "estimators/time_estimator.hpp"

namespace Ariadne {

TimeEstimatorModel::TimeEstimatorModel() : model() {
    data_training_fp = std::filesystem::path(__FILE__).parent_path() / ".." / ".." / "data" / DATA_TRAINING_FN;
}

TimeEstimatorModel::~TimeEstimatorModel() = default;

void TimeEstimatorModel::load_data() {
    data::Load(this->data_training_fp, this->data, true, true, arma::csv_ascii);
}


} // namespace Ariadne
