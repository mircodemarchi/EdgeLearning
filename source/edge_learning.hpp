/***************************************************************************
 *            edge_learning.hpp
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

/*! \file  edge_learning.hpp
 *  \brief Top-level header file includes all user headers.
 */

#ifndef EDGE_LEARNING_EDGE_LEARNING_HPP
#define EDGE_LEARNING_EDGE_LEARNING_HPP

#include "type.hpp"
#include "dnn/dlmath.hpp"
#include "dnn/model.hpp"
#include "dnn/layer.hpp"
#include "dnn/optimizer.hpp"
#include "dnn/cce_loss.hpp"
#include "dnn/mse_loss.hpp"
#include "dnn/dense.hpp"
#include "dnn/activation.hpp"
#include "dnn/recurrent.hpp"
#include "dnn/convolutional.hpp"
#include "dnn/pooling.hpp"
#include "dnn/max_pooling.hpp"
#include "dnn/avg_pooling.hpp"
#include "dnn/dropout.hpp"
#include "dnn/optimizer.hpp"
#include "dnn/gd_optimizer.hpp"
#include "middleware/definitions.hpp"
#include "middleware/nn.hpp"
#include "middleware/fnn.hpp"
#include "middleware/rnn.hpp"
#include "data/dataset.hpp"
#include "parser/type_checker.hpp"
#include "parser/parser.hpp"
#include "parser/csv.hpp"
#include "parser/mnist.hpp"

namespace EdgeLearning { } // namespace EdgeLearning

#endif // EDGE_LEARNING_EDGE_LEARNING_HPP
