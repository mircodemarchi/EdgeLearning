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

CSVRow::CSVRow(std::string line, size_t size, 
    std::vector<ParserType> &types, char separator)
    : _line{line}
    , _size{size}
    , _types{types}
    , _separator{separator}
{

}

CSVRow::CSVRow(std::string line, std::vector<ParserType> &types, 
    char separator)
    : CSVRow{line, 
        static_cast<size_t>(std::count(line.begin(), line.end(), separator)), 
        types, separator}
{

}

CSVRow::CSVRow(std::vector<ParserType> &types)
    : _line{}
    , _size{0}
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

    std::string header, first_line;
    std::getline(file, header);
    std::getline(file, first_line);

    _row_size = static_cast<size_t>(
        std::count(header.begin(), header.end(), separator));

    if((std::find(types.begin(), types.end(), ParserType::AUTO) != types.end())
        || types.size() == 0
        || types.size() != _row_size) 
    {
        std::stringstream ss{first_line};
        for (size_t i = 0; i < _row_size; ++i)
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

    _row_header = CSVRow{header,     _row_size, _types, separator};
    _row_cache  = CSVRow{first_line, _row_size, _types, separator};
    file.close();
}

const CSVRow &CSV::operator[](size_t idx)
{
    auto file = std::ifstream{_fn};

    for (size_t i = 0; i < idx - 1; ++i)
    {
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    std::string line;
    std::getline(file, line);
    _row_cache = CSVRow{line, _row_size, _types, _separator};
    return _row_cache;
}

} // namespace Ariadne
