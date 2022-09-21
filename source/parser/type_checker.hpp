/***************************************************************************
 *            parser/type_checker.hpp
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

/*! \file  parser/type_checker.hpp
 *  \brief TypeChecker::Type checking and conversion type of strings.
 */

#ifndef EDGE_LEARNING_PARSER_TYPE_CHECKER_HPP
#define EDGE_LEARNING_PARSER_TYPE_CHECKER_HPP

#include <string>
#include <sstream>
#include <vector>
#include <regex>

namespace EdgeLearning {

/**
 * \brief Regex to detect float values in string.
 */
const std::regex _float_regex   { "^[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$" };

/**
 * \brief Regex to detect int values in string.
 */
const std::regex _integer_regex { "^[-+]?(0|[1-9][0-9]*)$" };

/**
 * \brief Regex to detect bool values in string.
 */
const std::regex _boolean_regex { "^(true|false)$" };

/**
 * \brief Regex to detect strings
 */
const std::regex _string_regex  { "^\".*\"$" };

/**
 * \brief Convert a string in the templated type T and put the result in the
 * reference.
 * \tparam T  The type to convert to.
 * \param s   The string to convert.
 * \param ref The reference to the memory in which put the result.
 * \return true  If conversion succeeds.
 * \return false If conversion fails.
 */
template<typename T>
inline bool convert(const std::string &s, T& ref)
{
    std::stringstream ss{s};
    ss >> ref;
    return !ss.fail() && ss.eof();
}

/**
 * \brief Template specialization for string of convert<T>
 * \tparam std::string
 * \param s   The string to convert.
 * \param ref The reference to the memory in which put the result.
 * \return true  If conversion succeeds.
 * \return false If conversion fails.
 */
template<>
inline bool convert<std::string>(const std::string &s, std::string& ref)
{
    ref = s;
    return true;
}

/**
 * \brief Template specialization for boolean of convert<T>
 * \tparam bool
 * \param s   The string to convert.
 * \param ref The reference to the memory in which put the result.
 * \return true  If conversion succeeds.
 * \return false If conversion fails.
 */
template<>
inline bool convert<bool>(const std::string &s, bool& ref)
{
    if (s == "true" || s == "1")
    {
        ref = true;
    }
    else 
    {
        ref = false;
    }
    return true;
}

/**
 * \brief Convert a value in string.
 * \tparam T The type of the value.
 * \param v T The value to convert.
 * \return std::string The string converted.
 */
template<typename T>
inline std::string convert(T v)
{
    return std::to_string(v);
}

/**
 * \brief Template specialization for boolean of convert<T>.
 * \param v The boolean to convert.
 * \return std::string "true" or "false"
 */
template<>
inline std::string convert<bool>(bool v)
{
    return v ? "true" : "false";
}

/**
 * \brief TypeChecker class that manages the types of parsed strings in input.
 */
class TypeChecker
{
public:

    /**
     * \brief Enumeration class with the list of parserizable types.
     */
    enum class Type : int
    {
        NONE    = -1,
        AUTO    =  0,
        FLOAT   =  1,
        INT     =  2,
        BOOL    =  3,
        STRING  =  4,
        OBJECT  =  5,
    };

    /**
     * \brief Construct a new TypeChecker object.
     */
    TypeChecker() {};

    /**
     * \brief Destroy the TypeChecker object.
     */
    virtual ~TypeChecker() {};
    
    /**
     * \brief Check if the input string is of type float.
     * \param in     The input string.
     * \return true  It is of type float.
     * \return false It is not of type float.
     */
    static bool is_float(const std::string& in)  
        { return parse(in) == TypeChecker::Type::FLOAT;  };

    /**
     * \brief Check if the input string is of type bool.
     * \param in     The input string.
     * \return true  It is of type bool.
     * \return false It is not of type bool.
     */
    static bool is_bool(const std::string& in)   
        { return parse(in) == TypeChecker::Type::BOOL;   };

    /**
     * \brief Check if the input string is of type int.
     * \param in     The input string.
     * \return true  It is of type int.
     * \return false It is not of type int.
     */
    static bool is_int(const std::string& in)    
        { return parse(in) == TypeChecker::Type::INT;    };
    
    /**
     * \brief Check if the input string is of type string.
     * \param in     The input string.
     * \return true  It is of type string.
     * \return false It is not of type string.
     */
    static bool is_string(const std::string& in) 
        { return parse(in) == TypeChecker::Type::STRING; };
    
    /**
     * \brief Operator overloading for string conversion (see convert).
     * \tparam T
     * \param s   The string to convert.
     * \param ref The reference to the memory in which put the result.
     * \return true  If conversion succeeds.
     * \return false If conversion fails.
     */
    template<typename T> 
    bool operator()(const std::string &s, T& ref) const
    {
        return parse<T>(s, ref);
    }

    /**
     * \brief Operator overloading that parse a string (see parse).
     * \param field The string field.
     * \return TypeChecker::Type The type of the parsed field.
     */
    TypeChecker::Type operator()(const std::string& field) const
    {
        return parse(field);
    }

    /**
     * \brief Operator overloading that convert a value in string.
     * \tparam T The type of the value.
     * \param v T The value to convert.
     * \return std::string The string converted.
     */
    template<typename T>
    std::string operator()(T v) const
    {
        return convert<T>(v);
    }

    /**
     * \brief Operator () overloading that parse a vector of string.
     * \param field The vector of string field.
     * \return TypeChecker::Type The vector types of the parsed fields.
     */
    std::vector<TypeChecker::Type> operator()(
        const std::vector<std::string>& fields) const
    {
        return parse(fields);
    }

    /**
     * \brief Convert a string in the specified template type (see convert).
     * \tparam T  The specified type.
     * \param s   The string to convert.
     * \param ref The reference to the memory in which put the result.
     * \return true  If conversion succeeds.
     * \return false If conversion fails.
     */
    template<typename T>
    static bool parse(const std::string &s, T& ref)
    {
        return convert<T>(s, ref);
    }
    
    /**
     * \brief Parse the type of the given string field.
     * \param field The string to parse.
     * \return TypeChecker::Type The type of the converted string.
     */
    static TypeChecker::Type parse(const std::string &field)
    {
        if (field.empty())
        {
            return TypeChecker::Type::NONE;
        }

        if (std::regex_match(field, _string_regex))
        {
            return TypeChecker::Type::STRING;
        }
        else if (std::regex_match(field, _boolean_regex))
        {
            return TypeChecker::Type::BOOL;
        }
        else if (std::regex_match(field, _integer_regex))
        {
            return TypeChecker::Type::INT;
        }
        else if (std::regex_match(field, _float_regex))
        {
            return TypeChecker::Type::FLOAT;
        }
        else 
        {
            return TypeChecker::Type::STRING;
        }
    }

    /**
     * \brief Parse a vector of string fields.
     * \param fields The string fields to parse.
     * \return std::vector<TypeChecker::Type> The resulting types of string fields.
     */
    static std::vector<TypeChecker::Type> parse(
        const std::vector<std::string>& fields)
    {
        std::vector<TypeChecker::Type> ret;
        ret.resize(fields.size());
        for (std::size_t i = 0; i < fields.size(); ++i)
        {
            ret[i] = parse(fields[i]);
        }
        return ret;
    }
};

/**
 * \brief Operator overloading to cout the type.
 * \param os  Input stream.
 * \param obj TypeChecker::Type to print.
 * \return std::ostream& Output stream.
 */
inline std::ostream& operator<<(std::ostream& os, const TypeChecker::Type& obj)
{
   switch(obj)
   {
        case TypeChecker::Type::NONE:   os << "NONE"; break;
        case TypeChecker::Type::AUTO:   os << "AUTO"; break;
        case TypeChecker::Type::FLOAT:  os << "FLOAT"; break; 
        case TypeChecker::Type::INT:    os << "INT"; break;
        case TypeChecker::Type::BOOL:   os << "BOOL"; break;
        case TypeChecker::Type::STRING: os << "STRING"; break;
        case TypeChecker::Type::OBJECT: os << "OBJECT"; break;
        default: break;
   }
   os << "("
      << static_cast<std::underlying_type<TypeChecker::Type>::type>(obj)
      << ")";
   return os;
}

/**
 * \brief Operator overloading to cout a vector of types.
 * \param os  Input stream.
 * \param obj Types to print.
 * \return std::ostream& Output stream.
 */
inline std::ostream& operator<<(
    std::ostream& os,
    const std::vector<TypeChecker::Type>& obj)
{
    os << "{";
    for (auto &pt: obj)
    {
        os << pt << ",";
    }
    os << "}";
    return os;
}

/**
 * \brief Check if two vectors of types are equals.
 * \param lhs Vector of types first operand.
 * \param rhs Vector of types second operand.
 * \return true  Equals.
 * \return false Not equals.
 */
inline bool operator==(
    const std::vector<TypeChecker::Type> &lhs,
    const std::vector<TypeChecker::Type> &rhs)
{
    if (lhs.size() != rhs.size()) return false;
    for (std::size_t i = 0; i < lhs.size(); ++i)
    {
        if (lhs.at(i) != rhs.at(i))
        {
            return false;
        }
    }
    return true;
}

/**
 * \brief Check if two vectors of types are different.
 * \param lhs Vector of types first operand.
 * \param rhs Vector of types second operand.
 * \return true  Different.
 * \return false Equals.
 */
inline bool operator!=(
    const std::vector<TypeChecker::Type> &lhs,
    const std::vector<TypeChecker::Type> &rhs)
{
    return !operator==(lhs, rhs);
}

} // namespace EdgeLearning

#endif // EDGE_LEARNING_PARSER_TYPE_CHECKER_HPP
