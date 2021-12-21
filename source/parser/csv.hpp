/***************************************************************************
 *            csv.hpp
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

/*! \file csv.hpp
 *  \brief CSV Parser header.
 */

#ifndef EDGE_LEARNING_PARSER_CSV_HPP
#define EDGE_LEARNING_PARSER_CSV_HPP

#include "parser.hpp"

#include <cstddef>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <limits>
#include <stdexcept>


namespace EdgeLearning {

class CSVField : public Parser
{
    friend class CSVRow;

public:
    CSVField(std::string field, Type &type, size_t col_index)
        : Parser()
        , _field{field}
        , _type{type}
        , _col_index{col_index}
    {
        if (_type == Type::AUTO)
        {
            _type = _tc(_field);
        }
    }

    ~CSVField() {};

    template<typename T>
    void as(T *ptr) const
    {
        _tc(_field, ptr);
    }

    template<typename T>
    T as() const
    {
        T ret;
        _tc(_field, &ret);
        return ret;
    }

    const Type &type() const { return _type; }
    size_t idx() const { return _col_index; }

private:
    std::string _field;
    Type &_type;
    size_t _col_index;
};


class CSVRow : public Parser
{
    friend class CSV;
    friend class CSVIterator;

public:
    CSVRow(std::string line, size_t row_idx, size_t cols_amount, 
        std::vector<Type> &types, char separator = ',')
        : Parser()
        , _line{line}
        , _idx{row_idx}
        , _cols_amount{cols_amount}
        , _types{types}
        , _separator{separator}
    {
        if((std::find(types.begin(), types.end(), Type::AUTO) != types.end())
            || types.size() == 0
            || types.size() != cols_amount) 
        {
            std::stringstream ss{line};
            _types = std::vector<Type>{};
            for (size_t i = 0; i < cols_amount; ++i)
            {
                std::string s;
                std::getline(ss, s, separator);
                _types.push_back(_tc(s));
            }
        }
    }

    CSVRow(std::string line, size_t row_idx, std::vector<Type> &types, 
        char separator = ',') 
        : CSVRow{line, row_idx,
            static_cast<size_t>(std::count(line.begin(), line.end(), separator)+1), 
            types, separator}
    {

    }
    
    CSVRow(std::vector<Type> &types, char separator = ',')
        : CSVRow{std::string{}, size_t{}, size_t{0}, types, separator}
    {

    }

    CSVRow(const CSVRow &obj)
        : _line{obj._line}
        , _idx{obj._idx}
        , _cols_amount{obj._cols_amount}
        , _types{obj._types}
        , _separator{obj._separator}
    {

    }

    ~CSVRow() {};

    bool operator==(const CSVRow& rhs) const
    {
        return _line == rhs._line;
    }

    bool operator!=(const CSVRow& rhs) const
    {
        return _line != rhs._line;
    }

    CSVField operator[](size_t idx) const
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

    CSVRow& operator=(const CSVRow &obj)
    {
        _line = obj._line;
        _idx = obj._idx;
        _cols_amount = obj._cols_amount;
        _types = obj._types;
        _separator = obj._separator;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& stream, const CSVRow& obj)
    { 
        stream << obj._line;
        return stream;
    }

    operator std::vector<std::string>() const
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

    operator std::vector<CSVField>() const
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

    operator std::string() const
    {
        return _line;
    }

    template<typename T>
    operator std::vector<T>() const
    {
        std::vector<T> ret{};
        std::stringstream ss{_line};
        for (size_t i = 0; i < _cols_amount; ++i)
        {
            std::string s;
            std::getline(ss, s, _separator);
            T t;
            _tc(s, &t);
            ret.push_back(t);
        }
        return ret;
    }

    template<typename T>
    std::vector<T> to_vec() const
    {
        return operator std::vector<T>();
    }

    size_t size() const { return _cols_amount; }
    bool empty() const { return _cols_amount == 0; }
    const std::vector<Type> &types() const { return _types; } 
    size_t idx() const { return _idx; }
    std::string line() const { return _line; }

private:
    std::string _line;
    size_t _idx;
    size_t _cols_amount;
    std::vector<Type> &_types;
    char _separator;
};


class CSVIterator 
{
    using iterator_category = std::forward_iterator_tag;

public:
    
    CSVIterator(std::string fn, size_t idx, size_t cols_amount,
        std::vector<Type> &types, char separator = ',')
        : _fn{fn}
        , _row{types, separator}
        , _is_stream_updated{false}
        , _stream{fn}
    {
        _row._idx = idx;
        _row._cols_amount = cols_amount;
    }

    CSVIterator(const CSVIterator &obj)
        : _fn{obj._fn}
        , _row{obj._row}
        , _is_stream_updated{false}
        , _stream{obj._fn}
    {

    }

    ~CSVIterator()
    {
        _stream.close();
    }

    CSVRow &operator*()
    {
        update_stream();
        update_row();
        return _row;
    }

    CSVRow *operator->()
    {
        update_stream();
        update_row();
        return &_row;
    }

    bool operator==(const CSVIterator &rhs) const
    {
        return _row._idx == rhs._row._idx;
    }

    bool operator!=(const CSVIterator &rhs) const
    {
        return _row._idx != rhs._row._idx;
    }

    CSVIterator &operator++()
    {
        _row._idx++;
        return *this;
    }

    CSVIterator operator++(int)
    {
        CSVIterator tmp(*this);
        operator++();
        return tmp;
    }

private:
    void update_stream()
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

    void update_row()
    {
        std::getline(_stream, _row._line);
    }

    std::string _fn;
    CSVRow _row;
    bool _is_stream_updated;
    std::ifstream _stream;
};


class CSV : public Parser
{
public:
    CSV(std::string fn, std::vector<Type> types = { Type::AUTO }, 
        char separator = ',') 
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

        if((std::find(types.begin(), types.end(), Type::AUTO) != types.end())
            || types.size() == 0
            || types.size() != _cols_amount) 
        {
            std::stringstream ss{first_line};
            for (size_t i = 0; i < _cols_amount; ++i)
            {
                std::string s;
                std::getline(ss, s, separator);
                _types.push_back(_tc(s));
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
    
    ~CSV() {};

    size_t cols_size() const { return _cols_amount; }
    size_t rows_size() const { return _rows_amount; }
    const CSVRow &header() const { return _row_header; }
    const std::vector<Type> &types() const { return _types; }
    
    CSVRow operator[](size_t idx)
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
        return CSVRow{_row_cache};
    }

    CSVIterator begin()
    { 
        return CSVIterator{_fn, 1, _cols_amount, _types, _separator}; 
    }
    
    CSVIterator end()
    { 
        return CSVIterator{_fn, _rows_amount - 1, _cols_amount, _types, 
                           _separator}; 
    }

    operator std::vector<std::string>() const
    {
        std::vector<std::string> ret;

        auto file = std::ifstream{_fn};
        std::string line;
        for (size_t i = 0; i < _rows_amount; ++i)
        {
            if (!std::getline(file, line)) break;
            ret.push_back(line);
        }

        file.close();
        return ret;
    }

    operator std::vector<CSVRow>()
    {
        std::vector<CSVRow> ret;

        auto file = std::ifstream{_fn};
        std::string line;
        for (size_t i = 0; i < _rows_amount; ++i)
        {
            if (!std::getline(file, line)) break;
            ret.push_back(CSVRow{line, i, _cols_amount, _types, _separator});
        }

        file.close();
        return ret;
    }

    template<typename T>
    operator std::vector<std::vector<T>>()
    {
        std::vector<std::vector<T>> ret;
        ret.resize(_rows_amount);

        auto file = std::ifstream{_fn};
        std::string line;
        for (size_t i = 0; i < _rows_amount; ++i)
        {
            if (!std::getline(file, line)) break;
            ret[i] = std::vector<T>(
                CSVRow{line, i, _cols_amount, _types, _separator}
            );
        }

        file.close();
        return ret;
    }

    template<typename T>
    std::vector<std::vector<T>> to_vec()
    {
        return operator std::vector<std::vector<T>>();
    }

private:
    std::string _fn;
    std::vector<Type> _types;
    CSVRow _row_header;
    CSVRow _row_cache;
    size_t _cols_amount;
    size_t _rows_amount;
    char _separator;
};


} // namespace EdgeLearning

#endif // EDGE_LEARNING_REPLACEME_HPP
