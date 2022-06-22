/***************************************************************************
 *            parser/json.hpp
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

/*! \file  parser/json.hpp
 *  \brief JSON format Parser implementation.
 */

#ifndef EDGE_LEARNING_PARSER_JSON_HPP
#define EDGE_LEARNING_PARSER_JSON_HPP

#include "parser.hpp"

#include <cstddef>
#include <string>
#include <sstream>
#include <fstream>
#include <utility>
#include <vector>
#include <map>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <filesystem>
#include <tuple>
#include <memory>
#include <algorithm>

namespace EdgeLearning {

namespace fs = std::filesystem;

class JsonLeaf;
class JsonList;
class JsonDict;
class JsonItem;

/**
 * \brief Abstract JSON class for generalization.
 */
class JsonObject : public Parser
{
public:

    /**
     * \brief JSON Types that identify an implemented JSON class.
     */
    enum class JsonType : int
    {
        LEAF,   ///< \brief Enum for JsonLeaf type.
        LIST,   ///< \brief Enum for JsonList type.
        DICT,   ///< \brief Enum for JsonDict type.
        NONE    ///< \brief Enum for JsonItem or Json type.
    };

    using Shared = std::shared_ptr<JsonObject>;
    using Reference = JsonObject&;
    using Pointer = JsonObject*;

    /**
     * \brief Default constructor with NONE json_type.
     */
    JsonObject()
        : Parser()
        , _json_type(JsonType::NONE)
    { }

    /**
     * \brief Constructor with specific json_type.
     * \param json_type JsonType The JSON Type of the implemented class.
     */
    JsonObject(JsonType json_type)
        : Parser()
        , _json_type(json_type)
    { }

    /**
     * \brief Default deconstruct.
     */
    virtual ~JsonObject() {};

    /**
     * \brief Output stream of a generic JsonObject.
     * It converts the JsonObject in its implemented class through the json_type
     * attribute.
     * \param os std::ostream& The output stream to use.
     * \param obj const JsonObject& The object to stream out.
     * \return std::ostream& The updated output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const JsonObject& obj);

    /**
     * \brief Input stream of a generic JsonObject.
     * It converts the JsonObject in its implemented class through the json_type
     * attribute.
     * \param os std::istream& The input stream to use.
     * \param obj JsonObject& The object to stream in.
     * \return std::istream& The updated input stream.
     */
    friend std::istream& operator>>(std::istream& os, JsonObject& obj);

    /**
     * \brief Check if two JsonObject are equals converting them in the
     * implemented class through the json_type attribute.
     * \param rhs const JsonObject& Object to compare with this.
     * \return True if the objects are same types and if the converted type of
     * the objects are equals, otherwise false.
     */
    bool operator==(const JsonObject& rhs) const;
    bool operator!=(const JsonObject& rhs) const
    {
        return !(operator==(rhs));
    }

    /**
     * \brief Getter of json_type attribute.
     * \return JsonType The type of the instantiated object.
     */
    [[nodiscard]] virtual JsonType json_type() const { return _json_type; }

protected:
    JsonType _json_type; ///< \brief Type of the instantiated object.
};

/**
 * \brief Conversion function from generic JsonObject to implemented Json Type
 * based on json_type field of JsonObject.
 * If the json_type field does not correspond to the pointer of the object
 * passed, the function throws an exception.
 * \tparam T     The type of the Json class to convert the JsonObject.
 * \param jo_ptr The pointer to a JsonObject.
 * \return T The type cast of the JsonObject.
 */
template <typename T>
T convert_json_object(JsonObject::Shared jo_ptr);

/**
 * \brief An item of a JSON that contains a generic shared ptr of a JsonObject
 * that can be a JsonDict, a JsonList or a JsonLeaf.
 * The json type of this class acquired the json type of the contained
 * JsonObject.
 */
class JsonItem : public JsonObject
{
public:
    /**
     * \brief Empty constructor of a JsonItem.
     */
    JsonItem()
        : JsonObject(JsonType::NONE)
        , _value{}
    { }

    /**
     * \brief Constructor of a JsonItem with a generic reference of JsonObject.
     * \param value const JsonObject& The JsonObject reference object.
     */
    JsonItem(const JsonObject& value)
        : JsonObject(value.json_type())
        , _value(std::make_shared<JsonObject>(value))
    { }

    /**
     * \brief Constructor of a JsonItem with a JsonLeaf reference object.
     * \param jl const JsonLeaf& The leaf reference to use as JsonItem.
     */
    JsonItem(const JsonLeaf& jl);

    /**
     * \brief Constructor of a JsonItem with a std::string value.
     * \param val std::string The string value.
     */
    JsonItem(std::string val);

    /**
     * \brief Constructor of a JsonItem with a inline string value.
     * \param val const char* The string value.
     */
    JsonItem(const char* val);

    /**
     * \brief Constructor of a JsonItem with an integer value.
     * \param val The integer value: int, unsigned int, long, unsigned long,
     * long long, unsigned long long.
     */
    JsonItem(int val);
    JsonItem(unsigned int val);
    JsonItem(long val);
    JsonItem(unsigned long val);
    JsonItem(long long val);
    JsonItem(unsigned long long val);

    /**
     * \brief Constructor of a JsonItem with a double value.
     * \param val double The floating point value.
     */
    JsonItem(double val);

    /**
     * \brief Constructor of a JsonItem with a bool value.
     * \param val bool The boolean value.
     */
    JsonItem(bool val);

    /**
     * \brief Constructor of a JsonItem with a JsonList object.
     * \param list JsonList The list of JsonItem to use as JsonItem.
     */
    JsonItem(const JsonList& list);

    /**
     * \brief Constructor of a JsonItem with a vector of JsonItem.
     * \param list std::vector<JsonItem> The JsonItem list to use as JsonItem.
     */
    JsonItem(std::vector<JsonItem> list);

    /**
     * \brief Constructor of a JsonItem with a JsonDict object.
     * \param dict JsonDict The dictionary of JsonItem to use as JsonItem.
     */
    JsonItem(const JsonDict& dict);

    /**
     * \brief Constructor of a JsonItem with a string-JsonItem map.
     * \param dict std::map<std::string, JsonItem> The JsonItem string-JsonItem
     * map to use as JsonItem.
     */
    JsonItem(std::map<std::string, JsonItem> dict);

    /**
     * \brief Default deconstruct.
     */
    ~JsonItem() = default;

    /**
     * \brief Copy constructor.
     * \param obj const JsonItem& The object to copy.
     */
    JsonItem(const JsonItem& obj);

    /**
     * \brief Copy assignment overloading.
     * \param obj JsonItem The object to copy.
     * \return JsonItem& The assigned object.
     */
    JsonItem& operator=(const JsonItem& obj);

    /**
     * \brief Subscript operator overloading for contained list json.
     * It can be used to write the object and if the JsonItem is empty or there
     * are no json list, then it throws a runtime_error exception.
     * \param idx unsigned long The index of the list.
     * \return JsonItem& The JsonItem at the index.
     */
    JsonItem& operator[](std::size_t idx);
    JsonItem& operator[](int idx)
    {
        return operator[](static_cast<std::size_t>(idx));
    }

    /**
     * \brief Subscript operator overloading for contained dict json.
     * It can be used to write the object and if the JsonItem is empty then
     * it creates a JsonItem of type DICT.
     * \param key std::string The key of the dict.
     * \return JsonItem& The JsonItem value at the key.
     */
    JsonItem& operator[](std::string key);
    JsonItem& operator[](const char* key)
    {
        return operator[](std::string(key));
    }

    /**
     * \brief Constant read-only subscript method for contained list json.
     * If the JsonItem is empty or there are no json list, then it throws a
     * runtime_error exception.
     * \param idx unsigned long The index of the list.
     * \return JsonItem& The JsonItem at the index.
     */
    [[nodiscard]] const JsonItem& at(unsigned long idx) const;
    [[nodiscard]] const JsonItem& at(int idx) const
    {
        return at(static_cast<unsigned long>(idx));
    }

    /**
     * \brief Constant read-only subscript method for contained dict json.
     * If the JsonItem is empty or there are no json dict, then it throws a
     * runtime_error exception.
     * \param key std::string The key of the dict.
     * \return const JsonItem& The JsonItem value at the key.
     */
    [[nodiscard]] const JsonItem& at(std::string key) const;
    [[nodiscard]] const JsonItem& at(const char* key) const
    {
        return at(std::string(key));
    }

    /**
     * \brief Append a new JsonItem element in the contained list json.
     * If the JsonItem is empty then it creates a JsonItem of type LIST.
     * \param ji JsonItem The item to add in the list.
     */
    void append(JsonItem ji);

    /**
     * \brief Getter of the JsonObject reference.
     * \return The JsonObject reference.
     */
    [[nodiscard]] const JsonObject& value() const
    {
        if (!_value) throw std::runtime_error("value failed: empty object");
        return *_value;
    }

    /**
     * \brief Getter of the size of the JsonItem.
     * \return unsigned long Size of the item.
     */
    [[nodiscard]] unsigned long size() const;

    /**
     * \brief Convert the json object in the templated type and put in the ptr.
     * \tparam T  Field type requested.
     * \param ptr T& Reference in which put the result.
     */
    template<typename T>
    void as(T& ptr) const;

    /**
     * \brief Convert the json object in the templated vector type and put in
     * the ptr.
     * \tparam T  Field type requested for vector elements.
     * \param ptr std::vector<T>& Reference in which put the result.
     */
    template<typename T>
    void as_vec(std::vector<T>& ptr) const;

    /**
     * \brief Convert the json object in the templated map values type and put
     * in the ptr.
     * \tparam T  Field type requested for map values.
     * \param ptr std::map<std::string, T>& Reference in which put the result.
     */
    template<typename T>
    void as_map(std::map<std::string, T>& ptr) const;

    /**
     * \brief Return the converted json object as specified by the
     * template type.
     * \tparam T Field type requested.
     * \return T The converted field.
     */
    template<typename T>
    T as() const;

    /**
     * \brief Return the converted json object as a vector of the specified
     * template type.
     * \tparam T Field type requested for vector elements.
     * \return std::vector<T> The converted vector.
     */
    template<typename T>
    std::vector<T> as_vec() const;

    /**
     * \brief Return the converted json object as a map with values of the
     * specified template type.
     * \tparam T Field type requested for map values.
     * \return std::map<std::string, T> The converted map.
     */
    template<typename T>
    std::map<std::string, T> as_map() const;

    /**
     * \brief Overloading of equals operator.
     * \param rhs const JsonItem& The object to compare with.
     * \return bool True if the contained objects of the JsonItems are equals,
     * otherwise false.
     */
    bool operator==(const JsonItem& rhs) const;
    bool operator!=(const JsonItem& rhs) const
    {
        return !(operator==(rhs));
    }

    /**
     * \brief Output stream of a JsonItem.
     * \param os  std::ostream& The output stream to use.
     * \param obj const JsonItem& The object to stream out.
     * \return std::ostream& The updated output stream with the stream of the
     * JsonObject.
     */
    friend std::ostream& operator<<(std::ostream& os, const JsonItem& obj);

    /**
     * \brief Input stream of a JsonItem.
     * \param os  std::istream& The input stream to use.
     * \param obj JsonItem& The object to stream in the JsonObject.
     * \return std::istream& The updated input stream.
     */
    friend std::istream& operator>>(std::istream& os, JsonItem& obj);

    /**
     * \brief Overloading of integer conversion cast.
     * Integer types allowed: int, unsigned int, long, unsigned long,
     * long long, unsigned long long.
     * \return The converted integer value of this object.
     */
    operator int() const { return as<int>(); }
    operator unsigned int() const { return as<unsigned int>(); }
    operator long() const { return as<long>(); }
    operator unsigned long() const { return as<unsigned long>(); }
    operator long long() const { return as<long long>(); }
    operator unsigned long long() const { return as<unsigned long long>(); }

    /**
     * \brief Overloading of double conversion cast.
     * \return double The converted double of this object.
     */
    operator double() const { return as<double>(); }

    /**
     * \brief Overloading of float conversion cast.
     * \return float The converted float of this object.
     */
    operator float() const { return as<float>(); }

    /**
     * \brief Overloading of booloolean conversion cast.
     * \return bool The converted bool of this object.
     */
    operator bool() const { return as<bool>(); }

    /**
     * \brief Overloading of string conversion cast.
     * \return std::string The converted string of this object.
     */
    operator std::string() const;

private:

    Shared _value; ///< \brief The shared ptr of the JsonObject.
};

/**
 * \brief The minimum element class of a JSON.
 */
class JsonLeaf : public JsonObject
{
public:

    /**
     * \brief Default construct of a JsonLeaf.
     */
    JsonLeaf()
        : JsonObject(JsonType::LEAF)
        , _val{}
        , _type{Type::NONE}
    { }

    /**
     * \brief Constructor of a JsonLeaf.
     * \param val  const std::string& String of the value.
     * \param type Type The type of the value: int, float, string, boolean.
     * If type is AUTO, then the type is inferred through the parser.
     */
    JsonLeaf(const std::string& val, Type type = Type::AUTO)
        : JsonObject(JsonType::LEAF)
        , _val{val}
        , _type{type}
    {
        if (_type == Type::AUTO)
        {
            _type = _tc(_val);
        }
    }

    /**
     * \brief Construct a JsonLeaf with an array of char.
     * \param val  const char* An array of char as a value.
     * \param type Type The type of the value passed: int, float, string,
     * boolean.
     * If type is AUTO, then the type is inferred through the parser.
     */
    JsonLeaf(const char* val, Type type = Type::AUTO)
        : JsonLeaf(std::string(val), type)
    { }

    /**
     * \brief Construct a JsonLeaf with an integer.
     * \param val An integer value: int, unsigned int, long,
     * unsigned long, long long, unsigned long long.
     */
    JsonLeaf(int val)
        : JsonLeaf(std::to_string(val), Type::INT)
    { }
    JsonLeaf(unsigned int val)
        : JsonLeaf(std::to_string(val), Type::INT)
    { }
    JsonLeaf(long val)
        : JsonLeaf(std::to_string(val), Type::INT)
    { }
    JsonLeaf(unsigned long val)
        : JsonLeaf(std::to_string(val), Type::INT)
    { }
    JsonLeaf(long long val)
        : JsonLeaf(std::to_string(val), Type::INT)
    { }
    JsonLeaf(unsigned long long val)
        : JsonLeaf(std::to_string(val), Type::INT)
    { }

    /**
     * \brief Construct a JsonLeaf with a double.
     * \param val double A double as a value.
     */
    JsonLeaf(double val)
        : JsonLeaf(std::to_string(val), Type::FLOAT)
    { }

    /**
     * \brief Construct a JsonLeaf with a bool.
     * \param val bool A boolean as a value.
     */
    JsonLeaf(bool val)
        : JsonLeaf(val ? "true" : "false", Type::BOOL)
    { }

    /**
     * \brief Default deconstruct.
     */
    ~JsonLeaf() = default;

    /**
     * \brief Setter of the value of a JsonLeaf object.
     * \tparam T The type of the value passed as argument.
     * \param val T The value to overwrite.
     */
    template<typename T>
    void value(T val)
    {
        _val = _tc(val);
        _type = _tc(_val);
    }

    /**
     * \brief Getter of the value of a JsonLeaf object.
     * \return const std::string& The value field in string.
     */
    [[nodiscard]] const std::string& value() const { return _val; }

    /**
     * \brief Convert the field in the templated type and put in the ptr.
     * \tparam T  Field type requested.
     * \param ref T& Reference in which put the result.
     */
    template<typename T>
    void as(T& ptr) const
    {
        _tc(_val, ptr);
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
     * \brief Getter of the value type (that is different from the json_type).
     * \return const Type& The type of the value.
     */
    [[nodiscard]] const Type& type() const { return _type; }

    /**
     * \brief Overloading of equals operator.
     * \param rhs const JsonLeaf& The object to compare with.
     * \return bool True if the values are equals, otherwise false.
     */
    bool operator==(const JsonLeaf& rhs) const;
    bool operator!=(const JsonLeaf& rhs) const
    {
        return !(operator==(rhs));
    }

    /**
     * \brief Output stream of a JsonLeaf.
     * \param os  std::ostream& The output stream to use.
     * \param obj const JsonLeaf& The object to stream out.
     * \return std::ostream& The updated output stream with the value of
     * the JsonLeaf.
     */
    friend std::ostream& operator<<(std::ostream& os, const JsonLeaf& obj);

    /**
     * \brief Input stream of a JsonLeaf.
     * \param os  std::istream& The input stream to use.
     * \param obj JsonLeaf& The object to stream in the value of the JsonLeaf.
     * \return std::istream& The updated input stream.
     */
    friend std::istream& operator>>(std::istream& os, JsonLeaf& obj);

    /**
     * \brief Convert the value of the JsonLeaf in integer.
     * Integer types allowed: int, unsigned int, long, unsigned long,
     * long long, unsigned long long.
     * \return The converted integer value.
     */
    operator int() const { return as<int>(); }
    operator unsigned int() const { return as<unsigned int>(); }
    operator long() const { return as<long>(); }
    operator unsigned long() const { return as<unsigned long>(); }
    operator long long() const { return as<long long>(); }
    operator unsigned long long() const { return as<unsigned long long>(); }

    /**
     * \brief Convert the value of the JsonLeaf in floating point.
     * Floating point types allowed: double, float.
     * \return The converted floating point value.
     */
    operator double() const { return as<double>(); }
    operator float() const { return as<float>(); }

    /**
     * \brief Convert the value of the JsonLeaf in boolean.
     * \return bool The converted value.
     */
    operator bool() const { return as<bool>(); }

    /**
     * \brief Convert the value of the JsonLeaf in string.
     * \return std::string The converted value.
     */
    operator std::string() const;

private:
    std::string _val;   ///< \brief Value of the JsonLeaf.
    Type _type;         ///< \brief Type of the value field.
};

/**
 * \brief The list class of a JSON, identified by two squared bracket: [ ... ].
 */
class JsonList : public JsonObject
{
public:
    /**
     * \brief Empty constructor.
     */
    JsonList()
        : JsonObject(JsonType::LIST)
        , _list()
    { }

    /**
     * \brief Constructor of a JsonList with a vector of JsonItem.
     * \param list std::vector<JsonItem> The vector of JsonItem.
     */
    JsonList(std::vector<JsonItem> list)
        : JsonObject(JsonType::LIST)
        , _list(std::move(list))
    { }

    /**
     * \brief Constructor of a JsonList with a vector of integer.
     * \param val A vector of integers: int, unsigned int, long,
     * unsigned long, long long, unsigned long long.
     */
    JsonList(std::vector<int> val)
        : JsonObject(JsonType::LIST)
        , _list(_convert_vec<int>(val))
    { }
    JsonList(std::vector<unsigned int> val)
        : JsonObject(JsonType::LIST)
        , _list(_convert_vec<unsigned int>(val))
    { }
    JsonList(std::vector<long> val)
        : JsonObject(JsonType::LIST)
        , _list(_convert_vec<long>(val))
    { }
    JsonList(std::vector<unsigned long> val)
        : JsonObject(JsonType::LIST)
        , _list(_convert_vec<unsigned long>(val))
    { }
    JsonList(std::vector<long long> val)
        : JsonObject(JsonType::LIST)
        , _list(_convert_vec<long long>(val))
    { }
    JsonList(std::vector<unsigned long long> val)
        : JsonObject(JsonType::LIST)
        , _list(_convert_vec<unsigned long long>(val))
    { }

    /**
     * \brief Constructor of a JsonList with a vector of floating point.
     * \param val std::vector<double> A vector of double.
     */
    JsonList(std::vector<double> val)
        : JsonObject(JsonType::LIST)
        , _list(_convert_vec<double>(val))
    { }

    /**
     * \brief Constructor of a JsonList with a vector of boolean.
     * \param val std::vector<bool> A vector of boolean.
     */
    JsonList(std::vector<bool> val)
        : JsonObject(JsonType::LIST)
        , _list(_convert_vec<bool>(val))
    { }

    /**
     * \brief Constructor of a JsonList with a vector of string.
     * \param val std::vector<std::string> A vector of string.
     */
    JsonList(std::vector<std::string> val)
        : JsonObject(JsonType::LIST)
        , _list(_convert_vec<std::string>(val))
    { }

    /**
     * \brief Default deconstruct.
     */
    ~JsonList() = default;

    /**
     * \brief Subscript operator overloading to access and overwrite a JsonItem
     * contained in the list.
     * \param idx unsigned long The index to access.
     * \return JsonItem& The item accessed at the index.
     */
    JsonItem& operator[](std::size_t idx)
    {
        if (idx >= _list.size())
        {
            throw std::runtime_error(
                "operator[] failed: idx >= list size");
        }
        return _list[idx];
    }

    /**
     * \brief Constant read-only subscript method to access a JsonItem contained
     * in the list.
     * \param idx unsigned long The index to access.
     * \return const JsonItem& The item accessed at the index.
     */
    [[nodiscard]] const JsonItem& at(unsigned long idx) const
    {
        if (idx >= _list.size())
        {
            throw std::runtime_error(
                "method at() failed: idx >= list size");
        }
        return _list.at(idx);
    }

    /**
     * \brief Insert a JsonItem in the list.
     * \param ji const JsonItem& The JsonItem object to append.
     */
    void append(JsonItem ji)
    {
        _list.push_back(ji);
    }

    /**
     * \brief Getter of the vector of JsonItem contained in the object.
     * \return const std::vector<JsonItem>& The contained vector of JsonItem.
     */
    [[nodiscard]] const std::vector<JsonItem>& value() const { return _list; }

    /**
     * \brief Convert the json object in the templated vector type and put in
     * the ptr.
     * \tparam T  Field type requested for vector elements.
     * \param ptr std::vector<T>& Reference in which put the result.
     */
    template<typename T>
    void as(std::vector<T>& ptr) const
    {
        ptr = static_cast<std::vector<T>>(*this);
    }

    /**
     * \brief Return the converted json object as a vector of the specified
     * template type.
     * \tparam T Field type requested for vector elements.
     * \return std::vector<T> The converted vector.
     */
    template<typename T>
    std::vector<T> as() const
    {
        return static_cast<std::vector<T>>(*this);
    }

    /**
     * \brief Return the length of the list.
     * \return unsigned long The length of the list.
     */
    [[nodiscard]] unsigned long size() const { return _list.size(); }

    /**
     * \brief Check if the list is empty.
     * \return bool True if the list is empty, otherwise false.
     */
    [[nodiscard]] bool empty() const { return _list.empty(); }

    /**
     * \brief Overloading of equals operator.
     * \param rhs const JsonList& The object to compare with.
     * \return bool True if equals in size and the items in the lists are
     * equals in element-wise order, otherwise false.
     */
    bool operator==(const JsonList& rhs) const;
    bool operator!=(const JsonList& rhs) const
    {
        return !(operator==(rhs));
    }

    /**
     * \brief Output stream of a JsonList.
     * \param os  std::ostream& The output stream to use.
     * \param obj const JsonList& The object to stream out.
     * \return std::ostream& The updated output stream with the items of the
     * JsonList with two squared bracket at the beginning and at the end.
     */
    friend std::ostream& operator<<(std::ostream& os, const JsonList& obj);

    /**
     * \brief Input stream of a JsonList.
     * \param os  std::istream& The input stream to use.
     * \param obj JsonList& The object to stream in the items of the JsonList.
     * \return std::istream& The updated input stream.
     */
    friend std::istream& operator>>(std::istream& os, JsonList& obj);

    /**
     * \brief Convert to a vector of any type.
     * \tparam T The type of each element of the vector.
     * \return std::vector<T> The vector of the specified type.
     */
    template<typename T>
    operator std::vector<T>() const
    {
        std::vector<T> ret(_list.size());
        for (unsigned long i = 0; i < _list.size(); ++i)
        {
            ret[i] = static_cast<T>(_list.at(i));
        }
        return ret;
    }

    /**
     * \brief Overloading of string conversion cast.
     * \return std::string The converted string of this object.
     */
    operator std::string() const;

private:
    /**
     * \brief Convert a vector of any type elements in a vector of JsonItem.
     * \tparam T   The type of the input elements of the vector.
     * \param list const std::vector<T>& Input vector.
     * \return std::vector<JsonItem> Output JsonItem vector.
     */
    template<typename T>
    std::vector<JsonItem> _convert_vec(const std::vector<T>& list);

    std::vector<JsonItem> _list; ///< \brief The list of items.
};

/**
 * \brief The dictionary class of a JSON, identified by two bracket: { ... }.
 */
class JsonDict : public JsonObject
{
public:
    /**
     * \brief Empty constructor.
     */
    JsonDict()
        : JsonObject(JsonType::DICT)
        , _map()
    { }

    /**
     * \brief Constructor of a JsonDict with a map of string keys and JsonItem
     * values.
     * \param map std::map<std::string, JsonItem> The map string to JsonItem.
     */
    JsonDict(std::map<std::string, JsonItem> map)
        : JsonObject(JsonType::DICT)
        , _map(std::move(map))
    { }

    /**
     * \brief Constructor of a JsonDict with a map of string-integer pairs.
     * \param val A map of integer values: int, unsigned int, long,
     * unsigned long, long long, unsigned long long.
     */
    JsonDict(std::map<std::string, int> val)
        : JsonObject(JsonType::DICT)
        , _map(_convert_map<int>(val))
    { }
    JsonDict(std::map<std::string, unsigned int> val)
        : JsonObject(JsonType::DICT)
        , _map(_convert_map<unsigned int>(val))
    { }
    JsonDict(std::map<std::string, long> val)
        : JsonObject(JsonType::DICT)
        , _map(_convert_map<long>(val))
    { }
    JsonDict(std::map<std::string, unsigned long> val)
        : JsonObject(JsonType::DICT)
        , _map(_convert_map<unsigned long>(val))
    { }
    JsonDict(std::map<std::string, long long> val)
        : JsonObject(JsonType::DICT)
        , _map(_convert_map<long long>(val))
    { }
    JsonDict(std::map<std::string, unsigned long long> val)
        : JsonObject(JsonType::DICT)
        , _map(_convert_map<unsigned long long>(val))
    { }

    /**
     * \brief Constructor of a JsonDict with a map of string-double pairs.
     * \param val std::map<std::string, double> A map of floating point values.
     */
    JsonDict(std::map<std::string, double> val)
        : JsonObject(JsonType::DICT)
        , _map(_convert_map<double>(val))
    { }

    /**
     * \brief Constructor of a JsonDict with a map of string-boolean pairs.
     * \param val std::map<std::string, bool> A map of boolean values.
     */
    JsonDict(std::map<std::string, bool> val)
        : JsonObject(JsonType::DICT)
        , _map(_convert_map<bool>(val))
    { }

    /**
     * \brief Constructor of a JsonDict with a map of string-string pairs.
     * \param val std::map<std::string, std::string> A map of string values.
     */
    JsonDict(std::map<std::string, std::string> val)
        : JsonObject(JsonType::DICT)
        , _map(_convert_map<std::string>(val))
    { }

    /**
     * \brief Default deconstruct.
     */
    ~JsonDict() = default;

    /**
     * \brief Read-only subscript operator for a JsonItem in the dictionary map.
     * If the item is not contained in the map, it throws a runtime_error.
     * \param key std::string The key item to access.
     * \return const JsonItem& A read-only JsonItem in key.
     */
    [[nodiscard]] const JsonItem& at(std::string key) const
    {
        if (!_map.contains(key))
        {
            throw std::runtime_error(
                "operator[] failed: idx not contained in dict");
        }
        return _map.at(key);
    }

    /**
     * \brief Subscript operator overloading for read-write JsonItem in the
     * dictionary map.
     * \param key std::string The key item to read or write.
     * \return JsonItem& The reference of a JsonItem in the map.
     */
    JsonItem& operator[](std::string key)
    {
        return _map[key];
    }

    /**
     * \brief Getter of the map of JsonItem values contained in the object.
     * \return const std::map<std::string, JsonItem>& The contained map of
     * JsonItem values.
     */
    [[nodiscard]] const std::map<std::string, JsonItem>& value() const
    { return _map; }

    /**
     * \brief Convert the json object in the templated map values type and put
     * in the ptr.
     * \tparam T  Field type requested for map values.
     * \param ptr std::map<std::string, T>& Reference in which put the result.
     */
    template<typename T>
    void as(std::map<std::string, T>& ptr) const
    {
        ptr = static_cast<std::map<std::string, T>>(*this);
    }

    /**
     * \brief Return the converted json object as a map with values of the
     * specified template type.
     * \tparam T Field type requested for map values.
     * \return std::map<std::string, T> The converted map.
     */
    template<typename T>
    std::map<std::string, T> as() const
    {
        return static_cast<std::map<std::string, T>>(*this);
    }

    /**
     * \brief Return the size of the dictionary.
     * \return unsigned long The size of the map dictionary.
     */
    [[nodiscard]] unsigned long size() const { return _map.size(); }

    /**
     * \brief Check if the map is empty.
     * \return bool True if empty, otherwise false.
     */
    [[nodiscard]] bool empty() const { return _map.empty(); }

    /**
     * \brief Overloading of equals operator.
     * \param rhs const JsonDict& The object to compare with.
     * \return bool True if equals in size the items in the maps contain the
     * same keys and if items accessed through the keys are equals,
     * otherwise false.
     */
    bool operator==(const JsonDict& rhs) const;
    bool operator!=(const JsonDict& rhs) const
    {
        return !(operator==(rhs));
    }

    /**
     * \brief Output stream of a JsonDict.
     * \param os  std::ostream& The output stream to use.
     * \param obj const JsonDict& The object to stream out.
     * \return std::ostream& The updated output stream with the keys and the
     * items of the JsonDict separated by a comma and two brackets one at the
     * beginning and one at the end.
     */
    friend std::ostream& operator<<(std::ostream& os, const JsonDict& obj);

    /**
     * \brief Input stream of a JsonDict.
     * \param os  std::istream& The input stream to use.
     * \param obj JsonDict& The object to stream in the keys and the items of
     * the JsonDict.
     * \return std::istream& The updated input stream.
     */
    friend std::istream& operator>>(std::istream& os, JsonDict& obj);

    /**
     * \brief Convert to a map of string and any type pairs.
     * \tparam T The type of the values of the map.
     * \return std::map<std::string, T> The map of string and the specified
     * type pairs.
     */
    template<typename T>
    operator std::map<std::string, T>() const
    {
        std::map<std::string, T> ret;
        for (const auto& e: _map)
        {
            ret[e.first] = static_cast<T>(e.second);
        }
        return ret;
    }

    /**
     * \brief Overloading of string conversion cast.
     * \return std::string The converted string of this object.
     */
    operator std::string() const;

private:
    /**
     * \brief Convert a map of any type values in a map of string-JsonItem.
     * \tparam T   The type of the input values of the map.
     * \param list const std::map<std::string, T>& Input map.
     * \return std::map<std::string, JsonItem> Output string-JsonItem map.
     */
    template<typename T>
    std::map<std::string, JsonItem> _convert_map(
        const std::map<std::string, T>& map);

    /**
     * \brief The map with the pairs string, JsonItem.
     */
    std::map<std::string, JsonItem> _map;
};

/**
 * \brief High interface of a JSON.
 */
using Json = JsonItem;

inline JsonItem::JsonItem(const JsonLeaf& value)
    : JsonObject(value.json_type())
    , _value(std::make_shared<JsonLeaf>(value))
{ }

inline JsonItem::JsonItem(std::string val)
    : JsonItem(JsonLeaf(val))
{ }
inline JsonItem::JsonItem(const char* val)
    : JsonItem(JsonLeaf(val))
{ }

inline JsonItem::JsonItem(int val)
    : JsonItem(JsonLeaf(val))
{ }
inline JsonItem::JsonItem(unsigned int val)
    : JsonItem(JsonLeaf(val))
{ }
inline JsonItem::JsonItem(long val)
    : JsonItem(JsonLeaf(val))
{ }
inline JsonItem::JsonItem(unsigned long val)
    : JsonItem(JsonLeaf(val))
{ }
inline JsonItem::JsonItem(long long val)
    : JsonItem(JsonLeaf(val))
{ }
inline JsonItem::JsonItem(unsigned long long val)
    : JsonItem(JsonLeaf(val))
{ }

inline JsonItem::JsonItem(double val)
    : JsonItem(JsonLeaf(val))
{ }
inline JsonItem::JsonItem(bool val)
    : JsonItem(JsonLeaf(val))
{ }

inline JsonItem::JsonItem(const JsonList& value)
    : JsonObject(value.json_type())
    , _value(std::make_shared<JsonList>(value))
{ }

inline JsonItem::JsonItem(std::vector<JsonItem> list)
    : JsonItem(JsonList(list))
{ }

inline JsonItem::JsonItem(const JsonDict& value)
    : JsonObject(value.json_type())
    , _value(std::make_shared<JsonDict>(value))
{ }

inline JsonItem::JsonItem(std::map<std::string, JsonItem> dict)
    : JsonItem(JsonDict(dict))
{ }

inline JsonItem::JsonItem(const JsonItem& obj)
    : JsonObject(obj)
    , _value{}
{
    if (!obj._value) return;
    switch(obj._value->json_type())
    {
        case JsonObject::JsonType::LEAF:
        {
            auto obj_jl = std::dynamic_pointer_cast<JsonLeaf>(obj._value);
            if (obj_jl) _value = std::make_shared<JsonLeaf>(*obj_jl);
            break;
        }
        case JsonObject::JsonType::LIST:
        {
            auto obj_jl = std::dynamic_pointer_cast<JsonList>(obj._value);
            if (obj_jl) _value = std::make_shared<JsonList>(*obj_jl);
            break;
        }
        case JsonObject::JsonType::DICT:
        {
            auto obj_jd = std::dynamic_pointer_cast<JsonDict>(obj._value);
            if (obj_jd) _value = std::make_shared<JsonDict>(*obj_jd);
            break;
        }
        case JsonObject::JsonType::NONE:
        default:
        {
            break;
        }
    };
}

inline JsonItem& JsonItem::operator=(const JsonItem& obj)
{
    JsonItem tmp(obj);
    _value = tmp._value;
    _json_type = tmp.json_type();
    return *this;
}

inline JsonItem& JsonItem::operator[](std::size_t idx)
{
    if (!_value)
    {
        throw std::runtime_error("Try to call a subscript operator in empty "
                                 "JsonItem");
    }
    switch(_value->json_type())
    {
        case JsonObject::JsonType::LIST:
        {
            auto jl = std::dynamic_pointer_cast<JsonList>(_value);
            if (jl) return (*jl)[idx];
            break;
        }
        case JsonObject::JsonType::DICT:
        case JsonObject::JsonType::LEAF:
        case JsonObject::JsonType::NONE:
        default:
        {
            break;
        }
    };
    throw std::runtime_error("Try to subscript index operator in a non-list "
                             "json object");
}

inline JsonItem& JsonItem::operator[](std::string key)
{
    if (!_value || _json_type == JsonObject::JsonType::NONE)
    {
        _json_type = JsonObject::JsonType::DICT;
        _value = std::make_shared<JsonDict>();
    }
    switch(_value->json_type())
    {
        case JsonObject::JsonType::DICT:
        {
            auto jd = std::dynamic_pointer_cast<JsonDict>(_value);
            if (jd) return (*jd)[key];
            break;
        }
        case JsonObject::JsonType::LIST:
        case JsonObject::JsonType::LEAF:
        case JsonObject::JsonType::NONE:
        default:
        {
            break;
        }
    };
    throw std::runtime_error("Try to subscript key operator in a non-dict "
                             "json object");
}

inline const JsonItem& JsonItem::at(unsigned long idx) const
{
    if (!_value)
    {
        throw std::runtime_error("Try to call method at() in empty "
                                 "JsonItem");
    }
    switch(_value->json_type())
    {
        case JsonObject::JsonType::LIST:
        {
            auto jl = std::dynamic_pointer_cast<JsonList>(_value);
            if (jl) return (*jl).at(idx);
            break;
        }
        case JsonObject::JsonType::DICT:
        case JsonObject::JsonType::LEAF:
        case JsonObject::JsonType::NONE:
        default:
        {
            break;
        }
    };
    throw std::runtime_error("Try to call method at() in a non-list "
                             "json object");
}

inline const JsonItem& JsonItem::at(std::string key) const
{
    if (!_value)
    {
        throw std::runtime_error("Try to call method at() in empty "
                                 "JsonItem");
    }
    switch(_value->json_type())
    {
        case JsonObject::JsonType::DICT:
        {
            auto jd = std::dynamic_pointer_cast<JsonDict>(_value);
            if (jd) return (*jd).at(key);
            break;
        }
        case JsonObject::JsonType::LIST:
        case JsonObject::JsonType::LEAF:
        case JsonObject::JsonType::NONE:
        default:
        {
            break;
        }
    };
    throw std::runtime_error("Try to call method at() in a non-dict "
                             "json object");
}

inline void JsonItem::append(JsonItem ji)
{
    if (!_value || _json_type == JsonObject::JsonType::NONE)
    {
        _json_type = JsonObject::JsonType::LIST;
        _value = std::make_shared<JsonList>();
    }
    switch(_value->json_type())
    {
        case JsonObject::JsonType::LIST:
        {
            auto jl = std::dynamic_pointer_cast<JsonList>(_value);
            if (jl) return (*jl).append(ji);
            break;
        }
        case JsonObject::JsonType::DICT:
        case JsonObject::JsonType::LEAF:
        case JsonObject::JsonType::NONE:
        default:
        {
            break;
        }
    };
    throw std::runtime_error("Try to call method append() in a non-list "
                             "json object");
}

inline unsigned long JsonItem::size() const
{
    if (!_value)
    {
        return 0;
    }
    switch(_value->json_type())
    {
        case JsonObject::JsonType::DICT:
        {
            auto jd = std::dynamic_pointer_cast<JsonDict>(_value);
            if (jd) return (*jd).size();
            break;
        }
        case JsonObject::JsonType::LIST:
        {
            auto jl = std::dynamic_pointer_cast<JsonList>(_value);
            if (jl) return (*jl).size();
            break;
        }
        case JsonObject::JsonType::LEAF:
        case JsonObject::JsonType::NONE:
        default:
        {
            break;
        }
    };
    return 0;
}

template<typename T>
inline void JsonItem::as(T& ptr) const
{
    ptr = convert_json_object<T>(_value);
}

template<typename T>
inline void JsonItem::as_vec(std::vector<T>& ptr) const
{
    ptr = static_cast<std::vector<T>>(
        convert_json_object<JsonList>(_value));
}

template<typename T>
inline void JsonItem::as_map(std::map<std::string, T>& ptr) const
{
    ptr = static_cast<std::map<std::string, T>>(
        convert_json_object<JsonDict>(_value));
}

template<typename T>
inline T JsonItem::as() const
{
    return convert_json_object<T>(_value);
}

template<typename T>
inline std::vector<T> JsonItem::as_vec() const
{
    return static_cast<std::vector<T>>(
        convert_json_object<JsonList>(_value));
}

template<typename T>
inline std::map<std::string, T> JsonItem::as_map() const
{
    return static_cast<std::map<std::string, T>>(
        convert_json_object<JsonDict>(_value));
}

inline std::ostream& operator<<(std::ostream& os, const JsonObject& obj)
{
    switch(obj._json_type)
    {
        case JsonObject::JsonType::LEAF:
        {
            auto jl = dynamic_cast<const JsonLeaf*>(&obj);
            if (jl) os << *jl;
            break;
        }
        case JsonObject::JsonType::LIST:
        {
            auto jl = dynamic_cast<const JsonList*>(&obj);
            if (jl) os << *jl;
            break;
        }
        case JsonObject::JsonType::DICT:
        {
            auto jd = dynamic_cast<const JsonDict*>(&obj);
            if (jd) os << *jd;
            break;
        }
        case JsonObject::JsonType::NONE:
        default:
        {
            os << "";
            break;
        }
    };
    return os;
}

inline std::istream& operator>>(std::istream& os, JsonObject& obj)
{
    (void) obj;
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const JsonLeaf& obj)
{
    if (obj._type == Type::STRING)
    {
        os << "\"" << obj._val << "\"";
    }
    else
    {
        os << obj._val;
    }
    return os;
}

inline std::istream& operator>>(std::istream &os, JsonLeaf& obj)
{
    char c;
    os.read(&c, 1);
    if (c == '\"')
    {
        obj._type = Type::STRING;
        std::getline(os, obj._val, '\"');
        if (!os.eof())
        {
            os.read(&c, 1);
            if (c != ',') os.seekg(-1, os.cur);
        }
    }
    else
    {
        os.seekg(-1, os.cur);
        std::getline(os, obj._val, ',');
        auto obj_val_len = obj._val.size();
        auto pos_square = obj._val.find(']');
        auto pos_bracket = obj._val.find('}');
        auto pos = std::min(pos_square, pos_bracket);
        if (pos != std::string::npos)
        {
            obj._val = obj._val.substr(0, pos);
            if (!os.eof()) os.seekg(-1, os.cur); // ','
            os.seekg(-long(obj_val_len - pos), os.cur);
        }
        obj._type = obj._tc(obj._val);
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const JsonList& obj)
{
    os << "[";
    if (obj._list.empty())
    {
        os << "]";
        return os;
    }
    for (unsigned long i = 0; i < obj._list.size() - 1; ++i)
    {
        os << obj._list[i] << ",";
    }
    os << obj._list[obj._list.size() - 1] << "]";
    return os;
}

inline std::istream& operator>>(std::istream &os, JsonList& obj)
{
    char c;
    os.read(&c, 1);

    std::string more_items;
    while (true) {
        std::getline(os, more_items, ']');
        os.seekg(-long(more_items.size() + 1), os.cur);
        JsonItem ji;
        os >> ji;
        obj._list.push_back(ji);

        os.read(&c, 1);
        if (c == ']')
        {
            break;
        }
        else
        {
            os.seekg(-1, os.cur);
        }
    }

    if (!os.eof())
    {
        os.read(&c, 1);
        if (c != ',') os.seekg(-1, os.cur);
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const JsonDict& obj)
{
    unsigned long i = 0;
    os << "{";
    for (const auto& e: obj._map)
    {
        os << "\"" << e.first << "\"" << ":" << e.second;
        if (i++ != (obj._map.size() - 1))
        {
            os << ",";
        }
    }
    os << "}";
    return os;
}

inline std::istream& operator>>(std::istream& os, JsonDict& obj)
{
    char c;
    os.read(&c, 1);

    std::string more_items;
    while (true)
    {
        std::getline(os, more_items, '}');
        os.seekg(-long(more_items.size() + 1), os.cur);
        os.read(&c, 1);
        std::string key;
        std::getline(os, key, '\"');
        os.read(&c, 1);

        JsonItem ji;
        os >> ji;
        obj._map[key] = ji;

        os.read(&c, 1);
        if (c == '}')
        {
            break;
        }
        else
        {
            os.seekg(-1, os.cur);
        }
    }

    if (!os.eof())
    {
        os.read(&c, 1);
        if (c != ',') os.seekg(-1, os.cur);
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const JsonItem& obj)
{
    if (obj._value) os << *obj._value;
    return os;
}

inline std::istream& operator>>(std::istream& os, JsonItem& obj)
{
    char c;
    os.read(&c, 1);
    os.seekg(-1, os.cur);
    switch(c)
    {
        case '[':
        {
            auto jl = std::make_shared<JsonList>();
            os >> *jl;
            obj._value = jl;
            obj._json_type = JsonObject::JsonType::LIST;
            break;
        }
        case '{':
        {
            auto jd = std::make_shared<JsonDict>();
            os >> *jd;
            obj._value = jd;
            obj._json_type = JsonObject::JsonType::DICT;
            break;
        }
        default:
        {
            auto jd = std::make_shared<JsonLeaf>();
            os >> *jd;
            obj._value = jd;
            obj._json_type = JsonObject::JsonType::LEAF;
            break;
        }
    }
    return os;
}

inline bool JsonObject::operator==(const JsonObject& rhs) const
{
    if (_json_type != rhs._json_type) return false;
    switch(_json_type)
    {
        case JsonObject::JsonType::LEAF:
        {
            auto this_jl = dynamic_cast<const JsonLeaf*>(this);
            auto rhs_jl = dynamic_cast<const JsonLeaf*>(&rhs);
            if (this_jl && rhs_jl) return *this_jl == *rhs_jl;
            return false;
        }
        case JsonObject::JsonType::LIST:
        {
            auto this_jl = dynamic_cast<const JsonList*>(this);
            auto rhs_jl = dynamic_cast<const JsonList*>(&rhs);
            if (this_jl && rhs_jl) return *this_jl == *rhs_jl;
            return false;
        }
        case JsonObject::JsonType::DICT:
        {
            auto this_jd = dynamic_cast<const JsonDict*>(this);
            auto rhs_jd = dynamic_cast<const JsonDict*>(&rhs);
            if (this_jd && rhs_jd) return *this_jd == *rhs_jd;
            return false;
        }
        case JsonObject::JsonType::NONE:
        default:
        {
            return false;
        }
    };
}

inline bool JsonLeaf::operator==(const JsonLeaf& rhs) const
{
    return _val == rhs._val;
}

inline bool JsonList::operator==(const JsonList& rhs) const
{
    if (_list.size() != rhs._list.size()) return false;
    for (unsigned long i = 0; i < _list.size(); ++i)
    {
        if (_list[i] != rhs._list[i]) return false;
    }
    return true;
}

inline bool JsonDict::operator==(const JsonDict& rhs) const
{
    if (_map.size() != rhs._map.size()) return false;
    std::vector<std::string> this_keys;
    for (const auto& e: _map) this_keys.push_back(e.first);
    for (const auto& e: rhs._map)
    {
        if (std::find(this_keys.begin(), this_keys.end(), e.first)
            == this_keys.end()) return false;
    }
    for (const auto& k: this_keys)
    {
        if (_map.at(k) != rhs._map.at(k)) return false;
    }
    return true;
}

inline bool JsonItem::operator==(const JsonItem& rhs) const
{
    return *_value == *rhs._value;
}

inline JsonLeaf::operator std::string() const
{
    return _val;
}

inline JsonList::operator std::string() const
{
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

inline JsonDict::operator std::string() const
{
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

inline JsonItem::operator std::string() const
{
    if (!_value) return "";
    switch(_value->json_type())
    {
        case JsonObject::JsonType::LEAF:
        {
            auto jl = std::dynamic_pointer_cast<JsonLeaf>(_value);
            if (jl) return (*jl).value();
            break;
        }
        case JsonObject::JsonType::LIST:
        {
            auto jl = std::dynamic_pointer_cast<JsonList>(_value);
            if (jl) return static_cast<std::string>(*jl);
            break;
        }
        case JsonObject::JsonType::DICT:
        {
            auto jd = std::dynamic_pointer_cast<JsonDict>(_value);
            if (jd) return static_cast<std::string>(*jd);
            break;
        }
        case JsonObject::JsonType::NONE:
        default:
        {
            break;
        }
    };
    return "";
}

template<typename T>
inline std::vector<JsonItem> JsonList::_convert_vec(const std::vector<T>& list)
{
    std::vector<JsonItem> tmp(list.size());
    for (unsigned long i = 0; i < list.size(); ++i)
    {
        tmp[i] = JsonItem(list[i]);
    }
    return tmp;
}

template<typename T>
inline std::map<std::string, JsonItem> JsonDict::_convert_map(
    const std::map<std::string, T>& map)
{
    std::map<std::string, JsonItem> tmp;
    for (const auto& e: map)
    {
        tmp[e.first] = JsonItem(e.second);
    }
    return tmp;
}

template<typename T>
inline T convert_json_object(JsonObject::Shared jo_ptr)
{
    if (!jo_ptr)
    {
        throw std::runtime_error("Try to convert an empty JsonItem");
    }
    switch(jo_ptr->json_type())
    {
        case JsonObject::JsonType::LEAF:
        {
            auto jl = std::dynamic_pointer_cast<JsonLeaf>(jo_ptr);
            if (jl) return static_cast<T>(*jl);
            break;
        }
        case JsonObject::JsonType::DICT:
        case JsonObject::JsonType::LIST:
        case JsonObject::JsonType::NONE:
        default:
        {
            break;
        }
    };
    throw std::runtime_error("Try to convert a non-leaf json object");
}

template<>
inline JsonList convert_json_object<JsonList>(JsonObject::Shared jo_ptr)
{
    if (!jo_ptr)
    {
        throw std::runtime_error("Try to convert an empty JsonItem");
    }
    switch(jo_ptr->json_type())
    {
        case JsonObject::JsonType::LIST:
        {
            auto jl = std::dynamic_pointer_cast<JsonList>(jo_ptr);
            if (jl) return *jl;
            break;
        }
        case JsonObject::JsonType::DICT:
        case JsonObject::JsonType::LEAF:
        case JsonObject::JsonType::NONE:
        default:
        {
            break;
        }
    };
    throw std::runtime_error("Try to convert a non-list json object in a "
                             "vector");
}

template<>
inline JsonDict convert_json_object<JsonDict>(JsonObject::Shared jo_ptr)
{
    if (!jo_ptr)
    {
        throw std::runtime_error("Try to convert an empty JsonItem");
    }
    switch(jo_ptr->json_type())
    {
        case JsonObject::JsonType::DICT:
        {
            auto jd = std::dynamic_pointer_cast<JsonDict>(jo_ptr);
            if (jd) return *jd;
            break;
        }
        case JsonObject::JsonType::LIST:
        case JsonObject::JsonType::LEAF:
        case JsonObject::JsonType::NONE:
        default:
        {
            break;
        }
    };
    throw std::runtime_error("Try to convert a non-dict json object in a "
                             "map");
}

} // namespace EdgeLearning

#endif // EDGE_LEARNING_PARSER_JSON_HPP