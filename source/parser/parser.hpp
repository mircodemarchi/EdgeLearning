/***************************************************************************
 *            parser.hpp
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

/*! \file parser.hpp
 *  \brief Simply replace me.
 */

#ifndef ARIADNE_REPLACEME_HPP
#define ARIADNE_REPLACEME_HPP


#include <string>
#include <sstream>
#include <vector>
#include <regex>

namespace Ariadne {

enum class ParserType : int
{
    NONE    = -1, 
    AUTO    =  0,
    FLOAT   =  1,
    INT     =  2,
    BOOL    =  3,
    STRING  =  4,
    OBJECT  =  5,
};

class Parser
{
public:
    Parser() {};
    virtual ~Parser() = default;
    
    static bool is_float(std::string in)  
        { return parse(in) == ParserType::FLOAT;  };
    static bool is_bool(std::string in)   
        { return parse(in) == ParserType::BOOL;   };
    static bool is_int(std::string in)    
        { return parse(in) == ParserType::INT;    };
    static bool is_string(std::string in) 
        { return parse(in) == ParserType::STRING; };
    
    virtual ParserType operator()(const std::string &field) const 
    {
        return parse(field);
    }

    virtual std::vector<ParserType> operator()(
        const std::vector<std::string> &fields) const 
    {
        return parse(fields);
    }

    friend std::ostream& operator<<(std::ostream& os, const Parser& obj) 
    {
        (void) obj;
        return os;
    }

protected:
    static ParserType parse(const std::string &field);
    static std::vector<ParserType> parse(
        const std::vector<std::string> &fields);

private:
    static const std::regex float_regex;
    static const std::regex boolean_regex;
    static const std::regex integer_regex;
    static const std::regex string_regex;
};

std::ostream& operator<<(std::ostream& os, const ParserType& obj);

template<typename T>
bool convert(const std::string &s, T *ptr)
{
    if constexpr (std::is_same_v<T, bool>)
    {
        if (s == "true" || s == "1")
        {
            *ptr = true;
        }
        else 
        {
            *ptr = false;
        }

        return true;
    }

    std::stringstream ss{s};
    ss >> *ptr;

    return !ss.fail() && ss.eof();
}

} // namespace Ariadne

#endif // ARIADNE_REPLACEME_HPP
