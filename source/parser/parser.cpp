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

const std::regex Parser::_float_regex 
    { "^[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$" };
const std::regex Parser::_integer_regex { "^[-+]?(0|[1-9][0-9]*)$" };
const std::regex Parser::_boolean_regex { "^(true|false)$"    };
const std::regex Parser::_string_regex  { "^\".*\"$"          };

ParserType Parser::parse(const std::string &field)
{
    if (field.empty())
    {
        return ParserType::NONE;
    }

    if (std::regex_match(field, _string_regex))
    {
        return ParserType::STRING;
    }
    else if (std::regex_match(field, _boolean_regex))
    {
        return ParserType::BOOL;
    }
    else if (std::regex_match(field, _integer_regex))
    {
        return ParserType::INT;
    }
    else if (std::regex_match(field, _float_regex))
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
   switch(obj)
   {
        case ParserType::NONE:   os << "NONE"; break;
        case ParserType::AUTO:   os << "AUTO"; break;
        case ParserType::FLOAT:  os << "FLOAT"; break; 
        case ParserType::INT:    os << "INT"; break;
        case ParserType::BOOL:   os << "BOOL"; break;
        case ParserType::STRING: os << "STRING"; break;
        case ParserType::OBJECT: os << "OBJECT"; break;
        default: break;
   }
   os << "(" << static_cast<std::underlying_type<ParserType>::type>(obj) << ")";
   return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<ParserType>& obj)
{
    os << "{";
    for (auto &pt: obj)
    {
        os << pt << ",";
    }
    os << "}";
    return os;
}

bool operator==(const std::vector<ParserType> &lhs, 
    const std::vector<ParserType> &rhs)
{
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        if (lhs.at(i) != rhs.at(i))
        {
            return false;
        }
    }
    return true;
}

bool operator!=(const std::vector<ParserType> &lhs, 
    const std::vector<ParserType> &rhs)
{
    return !operator==(lhs, rhs);
}

} // namespace Ariadne
