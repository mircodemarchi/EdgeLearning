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
    friend class CSVRow;

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
    size_t idx() const    { return _col_index; }

private:
    std::string _field;
    const ParserType &_type;
    size_t _col_index;
};


class CSVRow 
{
    friend class CSV;
    friend class CSVIterator;

public:
    CSVRow(std::string line, size_t size, size_t row_idx, 
        std::vector<ParserType> &types, char separator = ',');
    CSVRow(std::string line, size_t row_idx, std::vector<ParserType> &types, 
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
    size_t idx() const { return _row_idx; }

private:
    std::string _line;
    size_t _size;
    size_t _row_idx;
    std::vector<ParserType> &_types;
    char _separator;
};


class CSVIterator 
{
    using iterator_category = std::forward_iterator_tag;

public:
    CSVIterator(CSVRow row, std::string fn);
    CSVIterator(const CSVIterator &obj);
    ~CSVIterator();

    CSVRow &operator*() { return _row; }
    CSVRow *operator->() { return &_row; }

    bool operator==(const CSVIterator& rhs) const;
    bool operator!=(const CSVIterator& rhs) const;
    CSVIterator &operator++();
    CSVIterator operator++(int);

private:
    std::string _fn;
    std::ifstream _file;
    CSVRow _row;
};


class CSV : public Parser
{
public:
    CSV(std::string fn, std::vector<ParserType> types = { ParserType::AUTO }, 
        char separator = ',');
    ~CSV() = default;

    size_t col_size() const { return _cols_amount; }
    size_t row_size() const { return _rows_amount; }
    const CSVRow &header() const { return _row_header; }
    const std::vector<ParserType> &types() const { return _types; }

    const CSVRow &operator[](size_t idx);

    CSVIterator begin() { return CSVIterator{operator[](1), _fn}; }
    CSVIterator end()   
        { return CSVIterator{operator[](_rows_amount - 1), _fn}; }

private:
    std::string _fn;
    std::vector<ParserType> _types;
    CSVRow _row_header;
    CSVRow _row_cache;
    size_t _cols_amount;
    size_t _rows_amount;
    char _separator;
};


} // namespace Ariadne

#endif // ARIADNE_REPLACEME_HPP
