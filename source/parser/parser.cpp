/***************************************************************************
 *            parser.cpp
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


#include "parser.hpp"

namespace Ariadne {

const std::regex Parser::float_regex 
    { "^[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$" };
const std::regex Parser::integer_regex { "^(0|[1-9][0-9]*)$" };
const std::regex Parser::boolean_regex { "^(true|false)$"    };
const std::regex Parser::string_regex  { "^\".*\"$"          };

ParserType Parser::parse(const std::string &field)
{
    if (std::regex_match(field, string_regex))
    {
        return ParserType::STRING;
    }
    else if (std::regex_match(field, boolean_regex))
    {
        return ParserType::BOOL;
    }
    else if (std::regex_match(field, integer_regex))
    {
        return ParserType::INT;
    }
    else if (std::regex_match(field, float_regex))
    {
        return ParserType::FLOAT;
    }
    else 
    {
        return ParserType::STRING;
    }
}

std::vector<ParserType> Parser::parse(
    const std::vector<std::string> &fields)
{
    std::vector<ParserType> ret;
    for (const std::string &field: fields)
    {
        ret.push_back(parse(field));
    }
    return ret;
}

std::ostream& operator<<(std::ostream& os, const ParserType& obj)
{
   os << static_cast<std::underlying_type<ParserType>::type>(obj);
   return os;
}

} // namespace Ariadne
