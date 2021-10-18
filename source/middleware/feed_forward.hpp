/***************************************************************************
 *            time_estimator.hpp
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

/*! \file time_estimator.hpp
 *  \brief Task execution time estimator model.
 */

#ifndef EDGE_LEARNING_ESTIMATORS_TIME_ESTIMATOR_HPP
#define EDGE_LEARNING_ESTIMATORS_TIME_ESTIMATOR_HPP

#if ENABLE_MLPACK

#include <filesystem>

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

namespace EdgeLearning {

using namespace mlpack;
using namespace mlpack::ann;

const std::string DATA_TRAINING_FN = "execution-time.csv";

class TimeEstimatorModel {
  public:
    TimeEstimatorModel();
    ~TimeEstimatorModel();

    void load_data();
  private:
    FFN<> model;
    arma::mat data;
    std::filesystem::path data_training_fp;
};

} // namespace EdgeLearning

#endif // ENABLE_MLPACK

#endif // EDGE_LEARNING_ESTIMATORS_TIME_ESTIMATOR_HPP
