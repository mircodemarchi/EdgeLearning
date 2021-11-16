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

#include <fstream>
#include <sstream>
#include <cstddef>


namespace EdgeLearning {

class CSVField
{
    friend class CSVRow;

public:
    CSVField(std::string field, ParserType &type, size_t col_index);
    ~CSVField() = default;

    template<typename T>
    void as(T *ptr) const
    {
        convert(this->_field, ptr);
    }

    template<typename T>
    T as() const
    {
        T ret;
        convert(this->_field, &ret);
        return ret;
    }

    const ParserType &type() const { return _type; }
    size_t idx() const { return _col_index; }

private:
    std::string _field;
    ParserType &_type;
    size_t _col_index;
};


class CSVRow 
{
    friend class CSV;
    friend class CSVIterator;

public:
    CSVRow(std::string line, size_t row_idx, size_t cols_amount,  
        std::vector<ParserType> &types, char separator = ',');
    CSVRow(std::string line, size_t row_idx, std::vector<ParserType> &types, 
        char separator = ',');
    CSVRow(std::vector<ParserType> &types, char separator = ',');
    CSVRow(const CSVRow &obj);
    ~CSVRow() = default;

    bool operator==(const CSVRow& rhs) const;
    bool operator!=(const CSVRow& rhs) const;

    CSVField operator[](size_t idx) const;
    CSVRow& operator=(const CSVRow &obj);
    friend std::ostream& operator<<(std::ostream& stream, const CSVRow& obj);
    operator std::vector<std::string>() const;
    operator std::vector<CSVField>() const;
    operator std::string() const;

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
            convert(s, &t);
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
    const std::vector<ParserType> &types() const { return _types; } 
    size_t idx() const { return _idx; }
    std::string line() const { return _line; }

private:
    std::string _line;
    size_t _idx;
    size_t _cols_amount;
    std::vector<ParserType> &_types;
    char _separator;
};


class CSVIterator 
{
    using iterator_category = std::forward_iterator_tag;

public:
    CSVIterator(std::string fn, size_t idx, size_t cols_amount,
        std::vector<ParserType> &types, char separator = ',');
    CSVIterator(const CSVIterator &obj);
    ~CSVIterator();

    CSVRow &operator*();
    CSVRow *operator->();

    bool operator==(const CSVIterator& rhs) const;
    bool operator!=(const CSVIterator& rhs) const;
    CSVIterator &operator++();
    CSVIterator operator++(int);

private:
    void update_stream();
    void update_row();

    std::string _fn;
    CSVRow _row;
    bool _is_stream_updated;
    std::ifstream _stream;
};


class CSV : public Parser
{
public:
    CSV(std::string fn, std::vector<ParserType> types = { ParserType::AUTO }, 
        char separator = ',');
    ~CSV() = default;

    size_t cols_size() const { return _cols_amount; }
    size_t rows_size() const { return _rows_amount; }
    const CSVRow &header() const { return _row_header; }
    const std::vector<ParserType> &types() const { return _types; }

    CSVRow operator[](size_t idx);

    CSVIterator begin()
    { 
        return CSVIterator{_fn, 1, _cols_amount, _types, _separator}; 
    }
    
    CSVIterator end()
    { 
        return CSVIterator{_fn, _rows_amount - 1, _cols_amount, _types, 
                           _separator}; 
    }

    operator std::vector<std::string>() const;
    operator std::vector<CSVRow>();

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
    std::vector<ParserType> _types;
    CSVRow _row_header;
    CSVRow _row_cache;
    size_t _cols_amount;
    size_t _rows_amount;
    char _separator;
};


} // namespace EdgeLearning

#endif // EDGE_LEARNING_REPLACEME_HPP
