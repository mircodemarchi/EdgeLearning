/***************************************************************************
 *            parser/csv.hpp
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

/*! \file  parser/csv.hpp
 *  \brief CSV Parser implementation.
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
#include <iterator>

namespace EdgeLearning {

/**
 * \brief Single field of a CSV file.
 * Composed by its field string, the corresponding type and the column number 
 * in the CSV file.
 */
class CSVField : public Parser
{
    friend class CSVRow;

public:
    /**
     * \brief Construct a new CSVField object
     * \param field     The CSV field string.
     * \param type      The CSV field type.
     * \param col_index The corresponding column index in CSV file.
     * The type field is changed if it contains the TypeChecker::Type::AUTO.
     */ 
    CSVField(std::string field, TypeChecker::Type& type, std::size_t col_index)
        : Parser()
        , _field{field}
        , _type{type}
        , _col_index{col_index}
    {
        if (_type == TypeChecker::Type::AUTO)
        {
            _type = _tc(_field);
        }
    }

    /**
     * \brief Construct a new CSVField object from another
     * \param other     Another CSVField object.
    */
    CSVField(CSVField const& other)
            : Parser()
            , _field{other._field}
            , _type{other._type}
            , _col_index{other._col_index}
    { }

    /**
     * \brief Assignment operator.
     * \param obj The object to assign.
     * \return CSVField& The updated object.
     */
    CSVField& operator=(const CSVField& obj)
    {
        _field = obj._field;
        _type = obj._type;
        _col_index = obj._col_index;
        return *this;
    }

    /**
     * \brief Destroy the CSVField object.
     */
    ~CSVField() = default;

    /**
     * \brief Convert the field in the templated type and put in the ptr.
     * \tparam T  Field type requested.
     * \param ref Reference in which put the result.
     */
    template<typename T>
    void as(T& ref) const
    {
        _tc(_field, ref);
    }

    /**
     * \brief Return the converted field as specified by the template type.
     * \tparam T Field type requested.
     * \return T The converted field.
     */
    template<typename T>
    T as() const
    {
        T ret;
        as(ret);
        return ret;
    }

    /**
     * \brief Field type getter.
     * \return const TypeChecker::Type&
     */
    const TypeChecker::Type& type() const { return _type; }

    /**
     * \brief CSV column index gettere.
     * \return std::size_t
     */
    std::size_t idx() const { return _col_index; }

private:
    std::string _field;     ///< CSV field string.
    TypeChecker::Type& _type;            ///< CSV field type.
    std::size_t _col_index; ///> CSV field column index.
};

/**
 * \brief The CSV row of a CSV file.
 * Composed by the CSV line string, the row index of the main CSV file, 
 * the number of columns, the type list of each column and the separator of each 
 * CSV field.
 * Composed by CSVField.
 */
class CSVRow : public Parser
{
    friend class CSV;
    friend class CSVIterator;

public:
    /**
     * \brief Construct a new CSVRow object.
     * \param line        The CSV line string.
     * \param row_idx     The CSV row index in the main CSV file.
     * \param cols_amount The number of columns.
     * \param types       The list of types of each field.
     * \param separator   The separator of each field.
     * The vector of types is changed if it's not compliant to the 
     * column amount or if it contains the TypeChecker::Type::AUTO.
     */
    CSVRow(std::string line, std::size_t row_idx, std::size_t cols_amount, 
        std::vector<TypeChecker::Type>& types, char separator = ',')
        : Parser()
        , _line{line}
        , _idx{row_idx}
        , _cols_amount{cols_amount}
        , _types{types}
        , _separator{separator}
    {
        if((std::find(types.begin(), types.end(), TypeChecker::Type::AUTO) != types.end())
            || types.size() == 0
            || types.size() != cols_amount) 
        {
            std::stringstream ss{line};
            _types = std::vector<TypeChecker::Type>{};
            for (std::size_t i = 0; i < cols_amount; ++i)
            {
                std::string s;
                std::getline(ss, s, separator);
                _types.push_back(_tc(s));
            }
        }
    }

    /**
     * \brief Construct a new CSVRow object.
     * \param line      The CSV line string.
     * \param row_idx   The CSV row index in the main CSV file.
     * \param types     The list of types of each field.
     * \param separator The separator of each field.
     * Automatically calculates the number of columns.
     */
    CSVRow(std::string line, std::size_t row_idx, std::vector<TypeChecker::Type>& types, 
        char separator = ',') 
        : CSVRow{line, row_idx,
            static_cast<std::size_t>(
                std::count(line.begin(), line.end(), separator) + 1), 
            types, separator}
    {

    }
    
    /**
     * \brief Construct a new CSVRow object.
     * \param types     The list of types of each field.
     * \param separator The separator of each field.
     * Construct an empty CSV Row.
     */
    CSVRow(std::vector<TypeChecker::Type>& types, char separator = ',')
        : CSVRow{std::string{}, std::size_t{}, std::size_t{0}, types, separator}
    {

    }

    /**
     * \brief Construct a new CSVRow object by copy.
     * \param obj
     */
    CSVRow(const CSVRow& obj)
        : _line{obj._line}
        , _idx{obj._idx}
        , _cols_amount{obj._cols_amount}
        , _types{obj._types}
        , _separator{obj._separator}
    {

    }

    /**
     * \brief Destroy the CSVRow object.
     */
    ~CSVRow() {};

    /**
     * \brief Check if two CSVRow are equal.
     * Two CSVRow are equal if the line strings are equal.
     * \param rhs The CSVRow to compare with this.
     * \return true  Equals.
     * \return false Different.
     */
    bool operator==(const CSVRow& rhs) const
    {
        return _line == rhs._line;
    }

    /**
     * \brief Check if two CSVRow are differne.
     * Two CSVRow are different if the line strings are different.
     * \param rhs The CSVRow to compare with this.
     * \return true  Different.
     * \return false Equals.
     */
    bool operator!=(const CSVRow& rhs) const
    {
        return _line != rhs._line;
    }

    /**
     * \brief Get a CSVField from the CSVRow.
     * \param idx The index of the CSVField.
     * \return CSVField The field of the CSVRow in the corresponding index.
     * Could throw a std::runtime_error if index is out of range. 
     */
    CSVField operator[](std::size_t idx) const
    {
        if (idx >= this->_cols_amount)
        {
            throw std::runtime_error(
                "operator[] failed: idx >= this->_cols_amount");
        }

        std::string field;
        std::size_t i = 0;
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

    /**
     * \brief Assignment operator.
     * \param obj The object to assign.
     * \return CSVRow& The updated object.
     */
    CSVRow& operator=(const CSVRow& obj)
    {
        _line = obj._line;
        _idx = obj._idx;
        _cols_amount = obj._cols_amount;
        _types = obj._types;
        _separator = obj._separator;
        return *this;
    }

    /**
     * \brief Operator overloading to print a CSVRow.
     * \param stream The input stream.
     * \param obj    The object to print.
     * \return std::ostream& The output stream.
     */
    friend std::ostream& operator<<(std::ostream& stream, const CSVRow& obj)
    { 
        stream << obj._line;
        return stream;
    }

    /**
     * \brief Divide the CSVRow in a vector of string fields.
     * \return std::vector<std::string>
     */
    operator std::vector<std::string>() const
    {
        std::vector<std::string> ret;
        ret.resize(_cols_amount);
        std::stringstream ss{_line};
        for (std::size_t i = 0; i < _cols_amount; ++i)
        {
            std::string s;
            std::getline(ss, s, _separator);
            ret[i] = s;
        }
        return ret;
    }

    /**
     * \brief Convert the CSVRow in a vector of CSVField objects.
     * \return std::vector<CSVField>
     */
    operator std::vector<CSVField>() const
    {
        std::vector<CSVField> ret{};
        std::stringstream ss{_line};
        for (std::size_t i = 0; i < _cols_amount; ++i)
        {
            std::string s;
            std::getline(ss, s, _separator);
            ret.push_back(CSVField{s, _types.at(i), i});
        }
        return ret;
    }

    /**
     * \brief Return the row in string.
     * \return std::string The CSV line.
     */
    operator std::string() const
    {
        return _line;
    }

    /**
     * \brief Convert each field of the CSVRow in the templated type and return
     * in a vector.
     * \tparam T The specified type of each row field.
     * \return std::vector<T> The converted vector of fields.
     */
    template<typename T>
    operator std::vector<T>() const
    {
        std::vector<T> ret;
        ret.resize(_cols_amount);
        std::stringstream ss{_line};
        for (std::size_t i = 0; i < _cols_amount; ++i)
        {
            std::string s;
            std::getline(ss, s, _separator);
            T t;
            _tc.template operator()<T>(s, t);
            ret[i] = t;
        }
        return ret;
    }

    /**
     * \brief Convert each field in the templated type
     * (see operator std::vector<T>()).
     * \tparam T The specified type of each row field.
     * \return std::vector<T> The converted vector of fields.
     */
    template<typename T>
    std::vector<T> to_vec() const
    {
        return std::vector<T>(*this);
    }

    /**
     * \brief Column amount getter.
     * \return std::size_t
     */
    std::size_t size() const { return _cols_amount; }
    
    /**
     * \brief Check if the CSVRow is empty. It is empty if the column amount is
     * zero.
     * \return true  Column amount greater than 0.
     * \return false Column amount equals than 0.
     */
    bool empty() const { return _cols_amount == 0; }
    
    /**
     * \brief Column types getter.
     * \return const std::vector<TypeChecker::Type>& The vector of column types.
     */
    const std::vector<TypeChecker::Type>& types() const { return _types; } 
    
    /**
     * \brief Getter of CSVRow index.
     * \return std::size_t Index of the line in CSV file.
     */
    std::size_t idx() const { return _idx; }
    
    /**
     * \brief Getter of CSVRow line.
     * \return std::string
     */
    std::string line() const { return _line; }

private:
    std::string _line;         ///< The line in string. 
    std::size_t _idx;          ///< The index of the CSV row.
    std::size_t _cols_amount;  ///< The column amount of the row.
    std::vector<TypeChecker::Type>& _types; ///< The types of each field in the row.
    char _separator;           ///< The separator of each field.
};

/**
 * \brief Iterator pattern for CSV class.
 */
class CSVIterator 
{
    using iterator_category = std::forward_iterator_tag;

public:
    
    /**
     * \brief Construct a new CSVIterator object.
     * \param fn          Filename of CSV file.
     * \param idx         Index of CSV row.
     * \param cols_amount Number of columns.
     * \param types       The list of types of each field.
     * \param separator   The separator of each field.
     * Initialize the internal CSVRow with the types list, the separator, the 
     * column amount and the requested row index. 
     * Open a file stream to input path fn. 
     */
    CSVIterator(std::string fn, std::size_t idx, std::size_t cols_amount,
        std::vector<TypeChecker::Type>& types, char separator = ',')
        : _fn{fn}
        , _row{std::string(), 0, cols_amount, types, separator}
        , _req_row_idx{idx}
        , _stream{fn}
    {
        std::getline(_stream, _row._line);
    }

    /**
     * \brief Construct a new CSVIterator object by copy.
     * \param obj Object to copy.
     */
    CSVIterator(const CSVIterator& obj)
        : _fn{obj._fn}
        , _row{std::string(), 0, obj._row._cols_amount, 
            obj._row._types, obj._row._separator}
        , _req_row_idx{obj._req_row_idx}
        , _stream{obj._fn}
    {
        std::getline(_stream, _row._line);
    }

    /**
     * \brief Destroy the CSVIterator object
     * Clone file stream.
     */
    ~CSVIterator()
    {
        _stream.close();
    }

    /**
     * \brief Access the requested CSVRow reference in CSV file and index.
     * \return CSVRow& The CSVRow reference.
     */
    CSVRow& operator*()
    {
        update_row();
        return _row;
    }

    /**
     * \brief Access the requested CSVRow pointer in CSV file and index.
     * \return CSVRow* The CSVRow pointer.
     */
    CSVRow* operator->()
    {
        update_row();
        return& _row;
    }

    /**
     * \brief Check if iterators are equal. Two iterators are equals if they
     * consider the same request row index. 
     * \param rhs The object to compare with this.
     * \return true  Equals.
     * \return false Different.
     */
    bool operator==(const CSVIterator& rhs) const
    {
        return _req_row_idx == rhs._req_row_idx;
    }

    /**
     * \brief Check if iterators are different. Two iterators are different if
     * they consider different request row index. 
     * \param rhs The object to compare with this.
     * \return true  Different.
     * \return false Equals.
     */
    bool operator!=(const CSVIterator& rhs) const
    {
        return _req_row_idx != rhs._req_row_idx;
    }

    /**
     * \brief Increment the iterator by 1 row.
     * \return CSVIterator& The reference to the updated iterator.
     */
    CSVIterator& operator++()
    {
        _req_row_idx++;
        return *this;
    }

    /**
     * \brief Increment the iterator by 1 row.
     * \return CSVIterator& The copy to the updated iterator.
     */
    CSVIterator operator++(int)
    {
        CSVIterator tmp(*this);
        operator++();
        return tmp;
    }

    /**
     * \brief Decrement the iterator by 1 row.
     * \return CSVIterator& The reference to the updated iterator.
     */
    CSVIterator& operator--()
    {
        _req_row_idx--;
        return *this;
    }

    /**
     * \brief Decrement the iterator by 1 row.
     * \return CSVIterator& The copy to the updated iterator.
     */
    CSVIterator operator--(int)
    {
        CSVIterator tmp(*this);
        operator--();
        return tmp;
    }

private:
    /**
     * \brief Read the right row line according to the requested row.
     */
    void update_row()
    {
        if (_req_row_idx < _row._idx) 
        {
            _row._idx = 0;
            _stream.seekg(0);
            std::getline(_stream, _row._line);
        }

        for (std::size_t i = _row._idx + 1; i < _req_row_idx; ++i)
        {
            _stream.ignore(
                std::numeric_limits<std::streamsize>::max(), '\n');
        }

        if (_row._idx != _req_row_idx)
        {
            std::getline(_stream, _row._line);
            _row._idx = _req_row_idx;
        }
    }

    /**
     * \brief Path of the CSV file.
     */
    std::string _fn; 

    /**
     * \brief CSVRow object updated and returned on iterator request.
     */
    CSVRow _row;
    
    /**
     * \brief
     */
    std::size_t _req_row_idx;
    
    /**
     * \brief Input stream used to read the CSV file.
     */
    std::ifstream _stream;
};


class CSV : public DatasetParser
{
public:
    /**
     * \brief Construct a new CSV object.
     * \param fn        The path of the CSV file.
     * \param types     The list of types of each field.
     * \param separator The separator of each field.
     * If the list of types is not compliant to the colums amount of the CSV 
     * file or it contains the TypeChecker::Type::AUTO, then the types list is automatically
     * computed using the second line of the CSV file.
     * The constructor throws a std::runtime_error if the file fails on open.
     */
    CSV(std::string fn, std::vector<TypeChecker::Type> types = { TypeChecker::Type::AUTO }, 
        char separator = ',', std::set<SizeType> labels_idx = {})
        : DatasetParser()
        , _fn{fn}
        , _types{types}
        , _row_header{_types}
        , _row_cache{_types}
        , _separator{separator}
        , _labels_idx{labels_idx}
    {
        auto file = std::ifstream{fn};
        if(!file.is_open() || !file.good()) 
        {
            throw std::runtime_error("Could not open file");
        }

        // Get number of rows.
        file.unsetf(std::ios_base::skipws);
        _rows_amount = static_cast<std::size_t>(
            std::count(std::istream_iterator<char>(file), 
                std::istream_iterator<char>(), '\n'));
        file.clear();
        file.seekg(0);

        // Get first two lines.
        std::string header, first_line;
        std::getline(file, header);
        std::getline(file, first_line);

        // Get number of columns.
        _cols_amount = static_cast<std::size_t>(
            std::count(header.begin(), header.end(), separator)+1);

        if((std::find(types.begin(), types.end(), TypeChecker::Type::AUTO) != types.end())
            || types.size() == 0
            || types.size() != _cols_amount) 
        {
            std::stringstream ss{first_line};
            for (std::size_t i = 0; i < _cols_amount; ++i)
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

    /**
     * \brief Destroy the CSV object.
     */
    ~CSV() {};

    /**
     * \brief Getter of the column size of the CSV file.
     * \return std::size_t The columns amount.
     */
    std::size_t cols_size() const { return _cols_amount; }

    /**
     * \brief Getter of the row size of the CSV file.
     * \return std::size_t The rows amount.
     */
    std::size_t rows_size() const { return _rows_amount; }

    /**
     * \brief Getter of the header row: the first line of the CSV file.
     * \return const CSVRow&
     */
    const CSVRow& header() const { return _row_header; }

    /**
     * \brief Getter of the types list of each field.
     * \return const std::vector<TypeChecker::Type>& Vector of types.
     */
    const std::vector<TypeChecker::Type>& types() const { return _types; }
    
    /**
     * \brief Get a CSVRow at the index row specified.
     * \param idx The index row.
     * \return CSVRow The requested row.
     */
    CSVRow operator[](std::size_t idx)
    {
        // Manage row saved in cache.
        if (_row_cache.idx() == idx) return _row_cache;
        // Handle overflow with a circular indexing.
        idx = idx % _rows_amount; 

        auto file = std::ifstream{_fn};
        // file.seekg(_row_header._line.size() + 1);
        for (std::size_t i = 0; i < idx; ++i)
        {
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

        std::string line;
        std::getline(file, line);
        _row_cache = CSVRow{line, idx, _cols_amount, _types, _separator};
        file.close();
        return CSVRow{_row_cache};
    }

    /**
     * \brief Get the first CSVIterator that is the second line of the CSV
     * file (it skips the header row).
     * \return CSVIterator
     */
    CSVIterator begin()
    { 
        return CSVIterator{_fn, 1, _cols_amount, _types, _separator}; 
    }
    
    /**
     * \brief Get the last CSVIterator.
     * \return CSVIterator
     */
    CSVIterator end()
    { 
        return CSVIterator{_fn, _rows_amount - 1, _cols_amount, _types, 
                           _separator}; 
    }

    /**
     * \brief Convert the CSV file in a vector of string for each row.
     * \return std::vector<std::string> Vector of CSV lines.
     */
    operator std::vector<std::string>()
    {
        std::vector<std::string> ret;
        ret.resize(_rows_amount);

        auto file = std::ifstream{_fn};
        std::string line;
        for (std::size_t i = 0; i < _rows_amount; ++i)
        {
            if (!std::getline(file, line)) break;
            ret[i] = line;
        }

        file.close();
        return ret;
    }

    /**
     * \brief Convert the CSV file in a vector of CSVRow for each row.
     * \return std::vector<CSVRow>
     */
    operator std::vector<CSVRow>()
    {
        std::vector<CSVRow> ret{};

        auto file = std::ifstream{_fn};
        std::string line;
        for (std::size_t i = 0; i < _rows_amount; ++i)
        {
            if (!std::getline(file, line)) break;
            ret.push_back(CSVRow{line, i, _cols_amount, _types, _separator});
        }

        file.close();
        return ret;
    }

    /**
     * \brief Convert the CSV file in a vector of vector with element converted
     * in the type specified by the template.
     * \tparam T The requested type of each CSV field.
     * \return std::vector<std::vector<T>> A vector of vector for each field.
     */
    template<typename T>
    operator std::vector<std::vector<T>>()
    {
        std::vector<std::vector<T>> ret;
        ret.resize(_rows_amount);

        auto file = std::ifstream{_fn};
        std::string line;
        for (std::size_t i = 0; i < _rows_amount; ++i)
        {
            if (!std::getline(file, line)) break;
            ret[i] = std::vector<T>(CSVRow{line, i, _cols_amount, _types, _separator});
        }

        file.close();
        return ret;
    }

    /**
     * \brief Convert the CSV file in a vector with each element converted
     * in the type specified by the template.
     * \tparam T The requested type of each CSV field.
     * \return std::vector<std::vector<T>> A vector of vector for each field.
     */
    template<typename T>
    operator std::vector<T>()
    {
        std::vector<T> ret{};

        auto file = std::ifstream{_fn};
        std::string line;
        for (std::size_t i = 0; i < _rows_amount; ++i)
        {
            if (!std::getline(file, line)) break;
            auto line_vec = std::vector<T>(CSVRow{line, i, _cols_amount, _types, _separator});
            ret.insert(ret.end(), line_vec.begin(), line_vec.end());
        }

        file.close();
        return ret;
    }

    /**
     * \brief Convert the CSV file in a vector for each field.
     * Wrapper of operator std::vector<std::vector<T>>().
     * \tparam T The requested type of each CSV field.
     * \return std::vector<T> A vector for each field.
     */
    template<typename T>
    std::vector<T> to_vec()
    {
        return std::vector<T>(*this);
    }

    /**
     * \brief Convert the CSV file in a vector of vector for each line.
     * Wrapper of operator std::vector<std::vector<T>>().
     * \tparam T The requested type of each CSV field.
     * \return std::vector<std::vector<T>> A vector of vector for each line.
     */
    template<typename T>
    std::vector<std::vector<T>> to_mat()
    {
        return std::vector<std::vector<T>>(*this);
    }

    std::vector<NumType> entry(SizeType i) override
    {
        return std::vector<NumType>(operator[](i + 1));
    }

    SizeType entries_amount() const override
    {
        return rows_size() - 1;
    }

    SizeType feature_size() const override
    {
        return cols_size();
    }

    std::set<SizeType> labels_idx() const override
    {
        return _labels_idx;
    }

private:
    std::string _fn;          ///< The CSV file path.
    std::vector<TypeChecker::Type> _types; ///< The types vector of each field.
    CSVRow _row_header;       ///< The header of the CSV file.
    CSVRow _row_cache;        ///< A Row cache used for optimization.
    std::size_t _cols_amount; ///< Number of columns in CSV file.
    std::size_t _rows_amount; ///< Number of rows in CSV file.
    char _separator;          ///< Separator character of each field.
    std::set<SizeType> _labels_idx;
};


} // namespace EdgeLearning

#endif // EDGE_LEARNING_PARSER_CSV_HPP
