/***************************************************************************
 *            ariadnedl.hpp
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

/*! \file ariadnedl.hpp
 *  \brief Top-level header file includes all user headers.
 */

#ifndef ARIADNE_ARIADNEDL_HPP
#define ARIADNE_ARIADNEDL_HPP

#include "dnn/type.hpp"
#include "dnn/dlmath.hpp"
#include "dnn/model.hpp"
#include "dnn/layer.hpp"
#include "dnn/optimizer.hpp"
#include "dnn/cce_loss.hpp"
#include "dnn/mse_loss.hpp"
#include "dnn/dense.hpp"
#include "dnn/recurrent.hpp"
#include "dnn/gd_optimizer.hpp"
#include "middleware/feed_forward.hpp"
#include "middleware/recurrent.hpp"
#include "parser/parser.hpp"
#include "parser/csv.hpp"

namespace Ariadne { } // namespace Ariadne

#endif // ARIADNE_ARIADNEDL_HPP
