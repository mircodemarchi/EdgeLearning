/***************************************************************************
 *            csv.cpp
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


#include "csv.hpp"

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <limits>


namespace Ariadne {

CSVField::CSVField(std::string field, ParserType &type, size_t col_index)
    : _field{field}
    , _type{type}
    , _col_index{col_index}
{
    if (_type == ParserType::AUTO)
    {
        auto p = Parser();
        _type = p(_field);
    }
}

CSVRow::CSVRow(std::string line, size_t row_idx, size_t cols_amount, 
    std::vector<ParserType> &types, char separator)
    : _line{line}
    , _idx{row_idx}
    , _cols_amount{cols_amount}
    , _types{types}
    , _separator{separator}
{
    if((std::find(types.begin(), types.end(), ParserType::AUTO) != types.end())
        || types.size() == 0
        || types.size() != cols_amount) 
    {
        std::stringstream ss{line};
        auto p = Parser();
        _types = std::vector<ParserType>{};
        for (size_t i = 0; i < cols_amount; ++i)
        {
            std::string s;
            std::getline(ss, s, separator);
            _types.push_back(p(s));
        }
    }
}

CSVRow::CSVRow(std::string line, size_t row_idx, std::vector<ParserType> &types, 
    char separator) 
    : CSVRow{line, row_idx,
        static_cast<size_t>(std::count(line.begin(), line.end(), separator)+1), 
        types, separator}
{

}

CSVRow::CSVRow(std::vector<ParserType> &types, char separator)
    : CSVRow{std::string{}, size_t{}, size_t{0}, types, separator}
{

}

CSVRow::CSVRow(const CSVRow &obj)
    : _line{obj._line}
    , _idx{obj._idx}
    , _cols_amount{obj._cols_amount}
    , _types{obj._types}
    , _separator{obj._separator}
{

}

CSVField CSVRow::operator[](size_t idx) const
{
    if (idx >= this->_cols_amount)
    {
        throw std::runtime_error(
            "operator[] failed: idx >= this->_cols_amount");
    }

    std::string field;
    size_t i = 0;
    std::stringstream ss{_line};
    while(std::getline(ss, field, _separator)) 
    {
        if(i == idx)
        {
            return CSVField{field, _types.at(idx), idx};
        }

        i++;
    }

    throw std::runtime_error("CSV bad format: fields missing");
}

CSVRow& CSVRow::operator=(const CSVRow &obj)
{
    _line = obj._line;
    _idx = obj._idx;
    _cols_amount = obj._cols_amount;
    _types = obj._types;
    _separator = obj._separator;
    return *this;
}

std::ostream& operator<<(std::ostream& stream, const CSVRow& obj)
{ 
    stream << obj._line;
    return stream;
}

CSVRow::operator std::vector<std::string>()
{
    std::vector<std::string> ret{};
    std::stringstream ss{_line};
    for (size_t i = 0; i < _cols_amount; ++i)
    {
        std::string s;
        std::getline(ss, s, _separator);
        ret.push_back(s);
    }
    return ret;
}

CSVRow::operator std::vector<CSVField>()
{
    std::vector<CSVField> ret{};
    std::stringstream ss{_line};
    for (size_t i = 0; i < _cols_amount; ++i)
    {
        std::string s;
        std::getline(ss, s, _separator);
        ret.push_back(CSVField{s, _types.at(i), i});
    }
    return ret;
}

CSVIterator::CSVIterator(std::string fn, size_t idx, size_t cols_amount,
    std::vector<ParserType> &types, char separator)
    : _fn{fn}
    , _row{types, separator}
    , _is_stream_updated{false}
    , _stream{fn}
{
    _row._idx = idx;
    _row._cols_amount = cols_amount;
}

CSVIterator::CSVIterator(const CSVIterator &obj)
    : _fn{obj._fn}
    , _row{obj._row}
    , _is_stream_updated{false}
    , _stream{obj._fn}
{

}

CSVIterator::~CSVIterator()
{
    _stream.close();
}

CSVRow &CSVIterator::operator*()
{
    update_stream();
    update_row();
    return _row;
}

CSVRow *CSVIterator::operator->()
{
    update_stream();
    update_row();
    return &_row;
}

bool CSVIterator::operator==(const CSVIterator &rhs) const
{
    return _row._idx == rhs._row._idx;
}

bool CSVIterator::operator!=(const CSVIterator &rhs) const
{
    return _row._idx != rhs._row._idx;
}

CSVIterator &CSVIterator::operator++()
{
    _row._idx++;
    return *this;
}

CSVIterator CSVIterator::operator++(int)
{
    CSVIterator tmp(*this);
    operator++();
    return tmp;
}

void CSVIterator::update_stream()
{
    if (!_is_stream_updated)
    {
        for (size_t i = 0; i < _row._idx; ++i)
        {
            _stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        _is_stream_updated = true;
    }
}

void CSVIterator::update_row()
{
    std::getline(_stream, _row._line);
}

CSV::CSV(std::string fn, std::vector<ParserType> types, char separator) 
    : Parser()
    , _fn{fn}
    , _types{types}
    , _row_header{_types}
    , _row_cache{_types}
    , _separator{separator}
{
    auto file = std::ifstream{fn};
    if(!file.is_open() || !file.good()) 
    {
        throw std::runtime_error("Could not open file");
    }

    // Get number of rows.
    file.unsetf(std::ios_base::skipws);
    _rows_amount = static_cast<size_t>(
        std::count(std::istream_iterator<char>(file), 
            std::istream_iterator<char>(), '\n'));
    file.clear();
    file.seekg(0);

    // Get first two lines.
    std::string header, first_line;
    std::getline(file, header);
    std::getline(file, first_line);

    // Get number of columns.
    _cols_amount = static_cast<size_t>(
        std::count(header.begin(), header.end(), separator)+1);

    if((std::find(types.begin(), types.end(), ParserType::AUTO) != types.end())
        || types.size() == 0
        || types.size() != _cols_amount) 
    {
        std::stringstream ss{first_line};
        for (size_t i = 0; i < _cols_amount; ++i)
        {
            std::string s;
            std::getline(ss, s, separator);
            _types.push_back(parse(s));
        }
    }
    else 
    {
        _types = types;
    }

    _row_header = CSVRow{header,     0, _cols_amount, _types, separator};
    _row_cache  = CSVRow{first_line, 1, _cols_amount, _types, separator};
    file.close();
}

const CSVRow &CSV::operator[](size_t idx)
{
    // Manage row saved in cache.
    if (_row_cache.idx() == idx) return _row_cache;
    // Handle overflow with a circular indexing.
    idx = idx % _rows_amount; 

    auto file = std::ifstream{_fn};
    // file.seekg(_row_header._line.size() + 1);
    for (size_t i = 0; i < idx; ++i)
    {
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    std::string line;
    std::getline(file, line);
    _row_cache = CSVRow{line, idx, _cols_amount, _types, _separator};
    file.close();
    return _row_cache;
}

} // namespace Ariadne
