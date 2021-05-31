/***************************************************************************
 *            csv.hpp
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

/*! \file csv.hpp
 *  \brief CSV Parser header.
 */

#ifndef ARIADNE_PARSER_CSV_HPP
#define ARIADNE_PARSER_CSV_HPP

#include "parser.hpp"

#include <fstream>
#include <sstream>
#include <cstddef>


namespace Ariadne {

class CSVField
{
public:
    CSVField(std::string field, const ParserType &type, size_t col_index);
    ~CSVField() = default;

    template<typename T>
    void as(T *ptr)
    {
        convert(this->_field, ptr);
    }

    template<typename T>
    T as()
    {
        T ret;
        convert(this->_field, &ret);
        return ret;
    }

    const ParserType &type()  const    { return _type; }
    size_t col_index() const    { return _col_index; }

private:
    std::string _field;
    const ParserType &_type;
    size_t _col_index;
};


class CSVRow 
{
public:
    CSVRow(std::string line, size_t size, std::vector<ParserType> &types, 
        char separator = ',');
    CSVRow(std::string line, std::vector<ParserType> &types, 
        char separator = ',');
    CSVRow(std::vector<ParserType> &types);
    CSVRow(const CSVRow &obj);
    ~CSVRow() = default;

    // CSVField& operator[](size_t idx);
    CSVField operator[](size_t idx) const;
    CSVRow& operator=(const CSVRow &obj);

    size_t size() const { return _size; }
    bool empty() const { return _size == 0; }
    const std::vector<ParserType> &types() const { return _types; } 

private:
    std::string _line;
    size_t _size;
    std::vector<ParserType> &_types;
    char _separator;
};


class CSV : public Parser
{
public:
    CSV(std::string fn, std::vector<ParserType> types = { ParserType::AUTO }, 
        char separator = ',');
    ~CSV() = default;

    size_t row_size() const { return _row_size; }
    const CSVRow &header() const { return _row_header; }
    const std::vector<ParserType> &types() const { return _types; }

    const CSVRow &operator[](size_t idx);

private:
    std::string _fn;
    std::vector<ParserType> _types;
    CSVRow _row_header;
    CSVRow _row_cache;
    size_t _row_size;
    char _separator;
};


} // namespace Ariadne

#endif // ARIADNE_REPLACEME_HPP
