/***************************************************************************
 *            time_estimator.cpp
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

#include "middleware/ffnn.hpp"

#include <utility>

namespace EdgeLearning {

FFNN::FFNN(
    LayerDescVec layers,
    SizeType input_size,
    LossType loss,
    SizeType batch_size,
    std::string name)
    : _layers{std::move(layers)}
    // , _input_size{input_size}
    , _loss{loss}
    , _batch_size{batch_size}
    , _name{name}
#if ENABLE_MLPACK
#else
    , _m{name}
#endif
{
#if ENABLE_MLPACK

#else
        std::vector<Layer::SharedPtr> l;
        auto prev_layer_size = input_size;
        for (const auto& e: _layers)
        {
            auto curr_layer_name = std::get<0>(e);
            auto curr_layer_size = std::get<1>(e);
            auto curr_layer_activation = std::get<2>(e);
            l.push_back(
                    _m.add_layer<DenseLayer>(
                            curr_layer_name, curr_layer_activation,
                            curr_layer_size, prev_layer_size)
            );
            prev_layer_size = curr_layer_size;
        }

        std::shared_ptr<LossLayer> loss_layer;
        auto output_size = prev_layer_size;
        switch(_loss)
        {
            case LossType::CCE: {
                loss_layer = _m.add_loss<CCELossLayer>(
                        "cce_loss", output_size, batch_size);
                break;
            }
            case LossType::MSE:
            default: {
                loss_layer = _m.add_loss<MSELossLayer>(
                        "mse_loss", output_size, batch_size);
                break;
            }
        }

        for (SizeType i = 0; i < l.size() - 1; ++i)
        {
            _m.create_edge(l[i], l[i + 1]);
        }
        _m.create_edge(l[l.size() - 1], loss_layer);
#endif
}

} // namespace EdgeLearning
