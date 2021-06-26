/***************************************************************************
 *            model.hpp
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

/*! \file model.hpp
 *  \brief Deep Neural Network Model.
 */

#ifndef ARIADNE_DNN_MODEL_HPP
#define ARIADNE_DNN_MODEL_HPP

#include "layer.hpp"
#include "optimizer.hpp"
#include "type.hpp"

#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace Ariadne {

/**
 * \brief Base class of a neural network model.
 */
class Model
{
public:
    Model(std::string name);

    /**
     * \brief Append a layer to the model, forward its parameters to the layer 
     * constructor and return its reference.
     * \tparam Layer_t The class name of the layer to append.
     * \tparam T       The list types of arguments to forward to the layer 
     *                 constructor.
     * \param args The list of arguments that will be forwarded to the layer 
     * constructor.
     * \return Layer_t& The reference to the layer inserted.
     */
    template <class Layer_t, typename... T>
    Layer_t& add_node(T&&... args)
    {
        _layers.push_back(
            std::make_unique<Layer_t>(*this, std::forward<T>(args)...)
        );
        return reinterpret_cast<Layer_t&>(*_layers.back());
    }

    /**
     * \brief Create a dependency between two constituent layers.
     * \param dst Destination layer.
     * \param src Source layer.
     */
    void create_edge(Layer& dst, Layer& src);

    /**
     * \brief Initialize the parameters of all nodes with the provided seed. 
     * If the seed is 0 a new random seed is chosen instead. 
     * \param seed Seed provided.  
     * \return RneType::result_type Seed used.
     */
    RneType::result_type init(RneType::result_type seed = 0);

    /**
     * \brief Adjust all model parameters of constituent layers using the 
     * provided optimizer. 
     * \param optimizer Provided optimizer.
     */
    void train(Optimizer& optimizer);

    /**
     * \brief Model name provided for debugging purposes.
     * \return std::string const& Model name string.
     */
    [[nodiscard]] std::string const& name() const noexcept
    {
        return _name;
    }

    /**
     * \brief Print.
     */
    void print() const;

    /**
     * \brief Save the model weights to disk.
     * 
     * To save the model to disk, we employ a very simple scheme. All nodes are
     * looped through in the order they were added to the model. Then, all
     * advertised learnable parameters are serialized in host byte-order to the
     * supplied output stream.
     *
     * This simplistic method of saving the model to disk isn't very
     * robust or practical in the real world. It contains no reflection data 
     * about the topology of the model. Furthermore, the data will be parsed 
     * incorrectly if the program is recompiled to operate with a different 
     * precision. 
     * 
     * \param out Out file stream
     */
    void save(std::ofstream& out);

    /**
     * \brief Load the model weights to disk.
     * \param in In file stream
     */
    void load(std::ifstream& in);

private:
    friend class Layer;

    std::string _name;                           ///< Model name;
    std::vector<std::unique_ptr<Layer>> _layers; ///< List of layers pointers;
};

} // namespace Ariadne

#endif // ARIADNE_DNN_MODEL_HPP
