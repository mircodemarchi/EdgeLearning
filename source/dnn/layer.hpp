/***************************************************************************
 *            layer.hpp
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

/*! \file layer.hpp
 *  \brief High level layer of a deep neural network.
 */

#ifndef EDGE_LEARNING_DNN_LAYER_HPP
#define EDGE_LEARNING_DNN_LAYER_HPP

#include "type.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>


namespace EdgeLearning {

class Model;

/**
 * \brief Base class of computational layers in a model.
 */
class Layer 
{
public:
    using SharedPtr = std::shared_ptr<Layer>;

    /**
     * @brief Construct a new Layer object.
     * @param model 
     * @param name 
     */
    Layer(Model& model, std::string name = std::string());

    /**
     * @brief Copy constructor of a new Layer object.
     * @param obj 
     */
    Layer(const Layer& obj);

    /**
     * @brief Destroy the Layer object.
     */
    virtual ~Layer() {};

    /**
     * @brief Assignment operator of a Layer object.
     * @param obj 
     * @return Layer& 
     */
    Layer& operator=(const Layer& obj);

    /**
     * \brief Virtual method used to describe how a layer should be 
     * initialized.
     * \param rne RneType
     */
    virtual void init(RneType& rne) = 0;

    /**
     * \brief Virtual method used to perform forward propagations. During 
     * forward propagation nodes transform input data and feed results to all 
     * subsequent nodes.
     * \param inputs NumType ptr
     */
    virtual void forward(NumType* inputs) = 0;

    /**
     * \brief Virtual method used to perform reverse propagations. During 
     * reverse propagation nodes receive loss gradients to its previous outputs
     * and compute gradients with respect to each tunable parameter.
     * Compute dJ/dz = dJ/dg(z) * dg(z)/dz.
     * \param gradients NumType ptr dJ/dg(z)
     */
    virtual void reverse(NumType* gradients) = 0;

    /**
     * \brief Virtual method that return the number of tunable parameters. 
     * This methos should be overridden to reflect the quantity of tunable 
     * parameters.
     * \return SizeType The amount of tunable parameters. 
     */
    virtual SizeType param_count() const noexcept { return 0; }

    /**
     * \brief Virtual method accessor for parameter by index.
     * \param index SizeType Parameter index.
     * \return NumType* Pointer to parameter.
     */
    virtual NumType* param(SizeType index) { (void) index; return nullptr; }

    /**
     * \brief Virtual method accessor for loss-gradient with respect to a 
     * parameter specified by index.
     * \param index SizeType Parameter index.
     * \return NumType* Pointer to gradient value of parameter.
     */
    virtual NumType* gradient(SizeType index) { (void) index; return nullptr; }

    /**
     * \brief Print.
     */
    virtual void print() const = 0;

    /**
     * \brief Virtual method information dump for debugging purposes.
     * \return std::string const& The layer name.
     */
    [[nodiscard]] std::string const& name() const noexcept 
    { 
        return _name; 
    }

protected:
    friend class Model;

    Model& _model;                      ///< Model reference.
    std::string _name;                  ///< Layer naem (for debug).
    std::vector<SharedPtr> _antecedents;   ///< List of previous layers.
    std::vector<SharedPtr> _subsequents;   ///< List of followers layers.
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_DNN_LAYER_HPP
