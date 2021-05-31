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


namespace Ariadne {

CSVField::CSVField(std::string field, const ParserType &type, size_t col_index)
    : _field{field}
    , _type{type}
    , _col_index{col_index}
{

}

CSVRow::CSVRow(std::string line, size_t size, size_t row_idx, 
    std::vector<ParserType> &types, char separator)
    : _line{line}
    , _size{size}
    , _row_idx{row_idx}
    , _types{types}
    , _separator{separator}
{

}

CSVRow::CSVRow(std::string line, size_t row_idx, std::vector<ParserType> &types, 
    char separator)
    : CSVRow{line, 
        static_cast<size_t>(std::count(line.begin(), line.end(), separator)), 
        row_idx, types, separator}
{

}

CSVRow::CSVRow(std::vector<ParserType> &types)
    : _line{}
    , _size{0}
    , _row_idx{}
    , _types{types}
    , _separator{}
{

}

CSVRow::CSVRow(const CSVRow &obj)
    : _line{obj._line}
    , _size{obj._size}
    , _types{obj._types}
    , _separator{obj._separator}
{

}

CSVField CSVRow::operator[](size_t idx) const
{
    if (idx >= this->_size)
    {
        throw std::runtime_error("operator[] failed: idx >= this->size");
    }

    std::string field;
    size_t i = 0;
    std::stringstream ss{_line};
    while(std::getline(ss, field, _separator)) {
        if(i == idx)
        {
            return CSVField{field, _types.at(idx), idx};
        }
    }

    throw std::runtime_error("CSV bad format: fields missing");
}

CSVRow& CSVRow::operator=(const CSVRow &obj)
{
    _line = obj._line;
    _size = obj._size;
    _types = obj._types;
    _separator = obj._separator;
    return *this;
}

CSVIterator::CSVIterator(CSVRow row, std::string fn)
    : _fn{fn}
    , _file{fn}
    , _row{row}
{
    size_t idx = _row.idx();
    for (size_t i = 0; i < idx; ++i)
    {
        _file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

CSVIterator::CSVIterator(const CSVIterator &obj)
    : _fn{obj._fn}
    , _file{obj._fn}
    , _row{obj._row}
{

}

CSVIterator::~CSVIterator()
{
    _file.close();
}

bool CSVIterator::operator==(const CSVIterator& rhs) const
{
    return this->_row.idx() == rhs._row.idx()
        && this->_row.size() == rhs._row.size();
}

bool CSVIterator::operator!=(const CSVIterator& rhs) const
{
    return this->_row.idx() != rhs._row.idx()
        || this->_row.size() != rhs._row.size();
}

CSVIterator &CSVIterator::operator++() 
{
    std::string line;
    std::getline(_file, line);
    this->_row._line = line;
    this->_row._row_idx++;
    return *this;
}

CSVIterator CSVIterator::operator++(int)
{
    CSVIterator tmp(*this); 
    operator++(); 
    return tmp;
}

CSV::CSV(std::string fn, std::vector<ParserType> types, char separator) 
    : Parser()
    , _fn{fn}
    , _row_header{types}
    , _row_cache{types}
    , _separator{separator}
{
    auto file = std::ifstream{fn};
    if(!file.is_open() || !file.good()) 
    {
        throw std::runtime_error("Could not open file");
    }

    // Get number of rows.
    _rows_amount = static_cast<size_t>(
        std::count(std::istream_iterator<char>(file), 
            std::istream_iterator<char>(), '\n'));
    file.seekg(0);

    // Get first two lines.
    std::string header, first_line;
    std::getline(file, header);
    std::getline(file, first_line);

    // Get number of columns.
    _cols_amount = static_cast<size_t>(
        std::count(header.begin(), header.end(), separator));

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

    _row_header = CSVRow{header,     _cols_amount, 0, _types, separator};
    _row_cache  = CSVRow{first_line, _cols_amount, 1, _types, separator};
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
    _row_cache = CSVRow{line, _cols_amount, idx, _types, _separator};
    file.close();
    return _row_cache;
}

} // namespace Ariadne
