/***************************************************************************
 *            parser/parser.hpp
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

/*! \file  parser/parser.hpp
 *  \brief Generic parser implementation.
 */

#ifndef EDGE_LEARNING_PARSER_PARSER_HPP
#define EDGE_LEARNING_PARSER_PARSER_HPP

#include "type_checker.hpp"

namespace EdgeLearning {

/**
 * \brief Generic class for parser the shares the TypeChecker entity.
 */
class Parser 
{
public:
    /**
     * \brief Construct a new Parser object.
     */
    Parser() : _tc() {}

    /**
     * \brief Destroy the Parser object.
     */
    virtual ~Parser() {};

protected:
    TypeChecker _tc; ///< TypeChecker entity to perform parsing and convertion.
};

} // namespace EdgeLearning

#endif // EDGE_LEARNING_PARSER_PARSER_HPP
