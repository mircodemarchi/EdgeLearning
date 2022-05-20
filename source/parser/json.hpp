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

/*! \file  parser/mnist.hpp
 *  \brief MNIST Dataset Parser implementation.
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
class Json;

/**
 * \brief Abstract JSON class for generalization.
 */
class JsonObject : public Parser
{
public:

    /**
     * \brief JSON Types that identify an implemented JSON class.
     */
    enum JsonType
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

    /**
     * \brief Getter of json_type attribute.
     * \return JsonType The type of the instantiated object.
     */
    [[nodiscard]] JsonType json_type() const { return _json_type; }

protected:
    JsonType _json_type; ///< \brief Type of the instantiated object.
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
     * \param type Type The type of the value passed: int, float, string, boolean.
     * If type is AUTO, then the type is inferred through the parser.
     */
    JsonLeaf(const char* val, Type type = Type::AUTO)
        : JsonLeaf(std::string(val), type)
    { }

    /**
     * \brief Construct a JsonLeaf with an integer.
     * \param val int An integer as a value.
     */
    JsonLeaf(int val)
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
        _val = std::to_string(val);
        _type = _tc(_val);
    }
    template<>
    void value<bool>(bool val)
    {
        _val = val ? "true" : "false";
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
     * \param ptr T* Pointer in which put the result.
     */
    template<typename T>
    void as(T* ptr) const
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
        _tc(_val, &ret);
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
     * \brief Constructor of a JsonList with a vector of JsonItem.
     * \param list std::vector<JsonItem> The vector of JsonItem.
     */
    JsonList(std::vector<JsonItem> list = {})
        : JsonObject(JsonType::LIST)
        , _list{std::move(list)}
    { }

    /**
     * \brief Default deconstruct.
     */
    ~JsonList() = default;

    /**
     * \brief Subscript operator overloading to access a JsonItem contained in
     * the list.
     * \param idx std::size_t The index to access.
     * \return const JsonItem& The item accessed at the index.
     */
    const JsonItem& operator[](std::size_t idx) const
    {
        if (idx >= _list.size())
        {
            throw std::runtime_error(
                "operator[] failed: idx >= list size");
        }
        return _list[idx];
    }

    /**
     * \brief Insert a JsonItem in the list.
     * \param ji JsonItem The JsonItem object to append.
     */
    void append(JsonItem ji);

    /**
     * \brief Return the length of the list.
     * \return std::size_t The length of the list.
     */
    [[nodiscard]] std::size_t size() const { return _list.size(); }

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

private:
    std::vector<JsonItem> _list; ///< \brief The list of items.
};

/**
 * \brief The dictionary class of a JSON, identified by two bracket: { ... }.
 */
class JsonDict : public JsonObject
{
public:

    /**
     * \brief Constructor of a JsonDict with a map of string keys and JsonItem
     * values.
     * \param map std::map<std::string, JsonItem> The map string to JsonItem.
     */
    JsonDict(std::map<std::string, JsonItem> map = {})
        : JsonObject(JsonType::DICT)
        , _map{map}
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
    const JsonItem& at(std::string key) const
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
     * \brief Return the size of the dictionary.
     * \return std::size_t The size of the map dictionary.
     */
    [[nodiscard]] std::size_t size() const { return _map.size(); }

    /**
     * \brief Check if the map is empty.
     * \return
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

private:
    /**
     * \brief The map with the pairs string, JsonItem.
     */
    std::map<std::string, JsonItem> _map;
};

/**
 * \brief An item of a JSON that contains a generic shared ptr of a JsonObject
 * that can be a JsonDict, a JsonList or a JsonLeaf.
 * The json type of this class is always NONE.
 */
class JsonItem : public JsonObject
{
    friend class Json;

public:
    /**
     * \brief Empty constructor of a JsonItem.
     */
    JsonItem()
        : JsonObject()
        , _value{}
    { }

    /**
     * \brief Constructor of a JsonItem with a JsonLeaf.
     * \param value JsonLeaf The JsonLeaf object.
     */
    JsonItem(JsonLeaf value)
        : JsonObject()
        , _value(std::make_shared<JsonLeaf>(value))
    { }

    /**
     * \brief Constructor of a JsonItem with a JsonLeaf.
     * \param value int The JsonLeaf initialized with an integer.
     */
    JsonItem(int value)
        : JsonItem(JsonLeaf(value))
    { }

    /**
     * \brief Constructor of a JsonItem with a JsonLeaf.
     * \param value double The JsonLeaf initialized with a double.
     */
    JsonItem(double value)
        : JsonItem(JsonLeaf(value))
    { }

    /**
     * \brief Constructor of a JsonItem with a JsonLeaf.
     * \param value bool The JsonLeaf initialized with a boolean.
     */
    JsonItem(bool value)
        : JsonItem(JsonLeaf(value))
    { }

    /**
     * \brief Constructor of a JsonItem with a JsonLeaf.
     * \param value std::string The JsonLeaf initialized with a string.
     */
    JsonItem(std::string value)
        : JsonItem(JsonLeaf(value))
    { }

    /**
     * \brief Constructor of a JsonItem with a JsonLeaf.
     * \param value const char* The JsonLeaf initialized with a char array.
     */
    JsonItem(const char* value)
        : JsonItem(JsonLeaf(value))
    { }

    /**
     * \brief Constructor of a JsonItem with a JsonList.
     * \param value JsonList The JsonList object.
     */
    JsonItem(JsonList value)
        : JsonObject()
        , _value(std::make_shared<JsonList>(value))
    { }

    /**
     * \brief Constructor of a JsonItem with a JsonList.
     * \param value JsonList The JsonList initialized with a vector of JsonItem.
     */
    JsonItem(std::vector<JsonItem> value)
        : JsonItem(JsonList(value))
    { }

    /**
     * \brief Constructor of a JsonItem with a JsonDict.
     * \param value JsonDict The JsonDict object.
     */
    JsonItem(JsonDict value)
        : JsonObject()
        , _value(std::make_shared<JsonDict>(value))
    { }

    /**
     * \brief Constructor of a JsonItem with a JsonDict.
     * \param value JsonDict The JsonDict initialized with a (string,JsonItem)
     * pairs map.
     */
    JsonItem(std::map<std::string, JsonItem> value)
        : JsonItem(JsonDict(value))
    { }

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
    JsonItem& operator=(JsonItem obj);

    /**
     * \brief Copy assignment overloading with a JsonLeaf object.
     * \param obj const JsonLeaf& The object to copy.
     * \return JsonItem& The assigned object.
     */
    JsonItem& operator=(const JsonLeaf& obj);

    /**
     * \brief Copy assignment overloading with an integer value.
     * \param obj const int& The JsonLeaf integer value to copy.
     * \return JsonItem& The assigned object.
     */
    JsonItem& operator=(const int& obj);

    /**
     * \brief Copy assignment overloading with a double value.
     * \param obj const double& The JsonLeaf double value to copy.
     * \return JsonItem& The assigned object.
     */
    JsonItem& operator=(const double& obj);

    /**
     * \brief Copy assignment overloading with a boolean value.
     * \param obj const bool& The JsonLeaf bool value to copy.
     * \return JsonItem& The assigned object.
     */
    JsonItem& operator=(const bool& obj);

    /**
     * \brief Copy assignment overloading with a string object.
     * \param obj const std::string& The JsonLeaf string object to copy.
     * \return JsonItem& The assigned object.
     */
    JsonItem& operator=(const std::string& obj);

    /**
     * \brief Copy assignment overloading with a char array object.
     * \param obj const char* The JsonLeaf char array object to copy.
     * \return JsonItem& The assigned object.
     */
    JsonItem& operator=(const char* obj);

    /**
     * \brief Copy assignment overloading with JsonList object.
     * \param obj const JsonList& The JsonList object to copy.
     * \return JsonItem& The assigned object.
     */
    JsonItem& operator=(const JsonList& obj);

    /**
     * \brief Copy assignment overloading with a vector of JsonItem.
     * \param obj const std::vector<JsonItem>& The JsonList vector of JsonItem
     * to copy.
     * \return JsonItem& The assigned object.
     */
    JsonItem& operator=(const std::vector<JsonItem>& obj);

    /**
     * \brief Copy assignment overloading with JsonDict object.
     * \param obj const JsonDict& The JsonDict object to copy.
     * \return JsonItem& The assigned object.
     */
    JsonItem& operator=(const JsonDict& obj);

    /**
     * \brief Copy assignment overloading with a (string,JsonItem) pair map.
     * \param obj const std::map<std::string, JsonItem>& The JsonDict
     * initialized with a (string,JsonItem) pair map to copy.
     * \return JsonItem& The assigned object.
     */
    JsonItem& operator=(const std::map<std::string, JsonItem>& obj);

    /**
     * \brief Copy assignment overloading with Json object.
     * \param obj const Json& The Json object to copy.
     * \return JsonItem& The assigned object.
     */
    JsonItem& operator=(Json obj);

    /**
     * \brief Getter of the JsonObject reference.
     * \return The JsonObject reference.
     */
    const JsonObject& value() const
    {
        if (!_value) throw std::runtime_error("value failed: empty object");
        return *_value;
    }

    /**
     * \brief Overloading of equals operator.
     * \param rhs const JsonItem& The object to compare with.
     * \return bool True if the contained objects of the JsonItems are equals,
     * otherwise false.
     */
    bool operator==(const JsonItem& rhs) const;

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

private:
    Shared _value; ///< \brief The shared ptr of the JsonObject.
};

/**
 * \brief High interface of a JSON.
 * The json type of this class acquired the json type of the contained
 * JsonObject.
 */
class Json : public JsonObject
{
    friend JsonItem;

public:
    /**
     * \brief Empty constructor. The json type is JsonType::NONE.
     */
    Json()
        : JsonObject(JsonType::NONE)
        , _obj{}
    { }

    /**
     * \brief Constructor of a Json object with a JsonList.
     * The json type is JsonType::LIST.
     * \param jl JsonList The JsonList object.
     */
    Json(JsonList jl)
        : JsonObject(JsonType::LIST)
        , _obj{jl}
    { }

    /**
     * \brief Constructor of a Json object with a JsonList.
     * The json type is JsonType::LIST.
     * \param value std::vector<JsonItem> The vector of JsonItems to wrap in a
     * JsonList.
     */
    Json(std::vector<JsonItem> value)
        : Json(JsonList(value))
    { }

    /**
     * \brief Constructor of a Json object with a JsonDict.
     * The json type is JsonType::DICT.
     * \param jl JsonDict The JsonDict object.
     */
    Json(JsonDict jd)
        : JsonObject(JsonType::DICT)
        , _obj{jd}
    { }

    /**
     * \brief Constructor of a Json object with a JsonDict.
     * The json type is JsonType::DICT.
     * \param value std::map<std::string, JsonItem> The (string,JsonItem) pair
     * map to wrap in a JsonDict.
     */
    Json(std::map<std::string, JsonItem> value)
        : Json(JsonDict(value))
    { }

    /**
     * \brief Insert a JsonItem in the Json.
     * If the json type is NONE, then the Json is initialized as a JsonList,
     * otherwise it throws a runtime_error because the append method cannot be
     * called in a DICT json type.
     * \param obj JsonItem The JsonItem to append.
     */
    void append(JsonItem obj)
    {
        if (_json_type == JsonObject::JsonType::NONE)
        {
            _json_type = JsonObject::JsonType::LIST;
            _obj = JsonItem(JsonList());
        }
        auto jl = std::dynamic_pointer_cast<JsonList>(_obj._value);
        if (!jl) throw std::runtime_error("Json is not a list");
        jl->append(obj);
    }

    /**
     * \brief Insert a Json object.
     * \param obj Json The Json object to append.
     */
    void append(Json obj)
    {
        append(obj._obj);
    }

    /**
     * \brief Subscript operator overloading to add a new (string, JsonItem)
     * pair in the Json. If the json type is NONE, then the Json is initialized
     * as a JsonDict, otherwise it throws a runtime_error because the subscript
     * operator cannot be called in a LIST json type.
     * \param key std::string The new key to insert.
     * \return JsonItem& The JsonItem to read or write in key.
     */
    JsonItem& operator[](std::string key)
    {
        if (_json_type == JsonObject::JsonType::NONE)
        {
            _json_type = JsonObject::JsonType::DICT;
            _obj = JsonItem(JsonDict());
        }
        auto jd = std::dynamic_pointer_cast<JsonDict>(_obj._value);
        if (!jd) throw std::runtime_error("Json is not a dictionary");
        return (*jd)[key];
    }

    /**
     * \brief Operator overloading of equals.
     * \param rhs The object to compare with.
     * \return bool True if the contained objects are equals, otherwise false.
     */
    bool operator==(const Json& rhs) const
    {
        return _obj == rhs._obj;
    }

    /**
     * \brief Output stream of a Json.
     * \param os  std::ostream& The output stream to use.
     * \param obj const JsonItem& The object to stream out.
     * \return std::ostream& The updated output stream with the stream of the
     * JsonObject.
     */
    friend std::ostream& operator<<(std::ostream& os, const Json& obj)
    {
        os << obj._obj;
        return os;
    }

    /**
     * \brief Input stream of a Json.
     * \param os  std::istream& The input stream to use.
     * \param obj Json& The object to stream in the JsonObject.
     * \return std::istream& The updated input stream.
     */
    friend std::istream& operator>>(std::istream& os, Json& obj)
    {
        os >> obj._obj;
        return os;
    }

private:
    JsonItem _obj; ///< \brief The JsonItem that contains the whole JSON.
};

void JsonList::append(JsonItem ji)
{
    _list.push_back(ji);
}

JsonItem::JsonItem(const JsonItem& obj)
    : _value{}
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

JsonItem& JsonItem::operator=(JsonItem obj)
{
    JsonItem tmp(obj);
    _value = tmp._value;
    return *this;
}

JsonItem& JsonItem::operator=(const JsonLeaf& obj)
{
    _value = std::make_shared<JsonLeaf>(obj);
    return *this;
}
JsonItem& JsonItem::operator=(const int& obj)
{
    return operator=(JsonLeaf(obj));
}
JsonItem& JsonItem::operator=(const double& obj)
{
    return operator=(JsonLeaf(obj));
}
JsonItem& JsonItem::operator=(const bool& obj)
{
    return operator=(JsonLeaf(obj));
}
JsonItem& JsonItem::operator=(const std::string& obj)
{
    return operator=(JsonLeaf(obj));
}
JsonItem& JsonItem::operator=(const char* obj)
{
    return operator=(JsonLeaf(obj));
}
JsonItem& JsonItem::operator=(const JsonList& obj)
{
    _value = std::make_shared<JsonList>(obj);
    return *this;
}
JsonItem& JsonItem::operator=(const std::vector<JsonItem>& obj)
{
    return operator=(JsonList(obj));
}
JsonItem& JsonItem::operator=(const JsonDict& obj)
{
    _value = std::make_shared<JsonDict>(obj);
    return *this;
}
JsonItem& JsonItem::operator=(const std::map<std::string, JsonItem>& obj)
{
    return operator=(JsonDict(obj));
}
JsonItem& JsonItem::operator=(Json obj)
{
    return operator=(obj._obj);
}

std::ostream& operator<<(std::ostream& os, const JsonObject& obj)
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

std::istream& operator>>(std::istream& os, JsonObject& obj)
{
    (void) obj;
    return os;
}

std::ostream& operator<<(std::ostream& os, const JsonLeaf& obj)
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

std::istream& operator>>(std::istream &os, JsonLeaf& obj)
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
            os.seekg(-std::int64_t(obj_val_len - pos), os.cur);
        }
        obj._type = obj._tc(obj._val);
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const JsonList& obj)
{
    os << "[";
    for (std::size_t i = 0; i < obj._list.size() - 1; ++i)
    {
        os << obj[i] << ",";
    }
    os << obj[obj._list.size() - 1] << "]";
    return os;
}

std::istream& operator>>(std::istream &os, JsonList& obj)
{
    char c;
    os.read(&c, 1);

    std::string more_items;
    while (true) {
        std::getline(os, more_items, ']');
        os.seekg(-std::int64_t(more_items.size() + 1), os.cur);
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

std::ostream& operator<<(std::ostream& os, const JsonDict& obj)
{
    std::size_t i = 0;
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

std::istream& operator>>(std::istream& os, JsonDict& obj)
{
    char c;
    os.read(&c, 1);

    std::string more_items;
    while (true)
    {
        std::getline(os, more_items, '}');
        os.seekg(-std::int64_t(more_items.size() + 1), os.cur);
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

std::ostream& operator<<(std::ostream& os, const JsonItem& obj)
{
    if (obj._value) os << *obj._value;
    return os;
}

std::istream& operator>>(std::istream& os, JsonItem& obj)
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
            break;
        }
        case '{':
        {
            auto jd = std::make_shared<JsonDict>();
            os >> *jd;
            obj._value = jd;
            break;
        }
        default:
        {
            auto jd = std::make_shared<JsonLeaf>();
            os >> *jd;
            obj._value = jd;
            break;
        }
    }
    return os;
}

bool JsonObject::operator==(const JsonObject& rhs) const
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

bool JsonLeaf::operator==(const JsonLeaf& rhs) const
{
    return _val == rhs._val;
}

bool JsonList::operator==(const JsonList& rhs) const
{
    if (_list.size() != rhs._list.size()) return false;
    for (std::size_t i = 0; i < _list.size(); ++i)
    {
        if (_list[i] != rhs._list[i]) return false;
    }
    return true;
}

bool JsonDict::operator==(const JsonDict& rhs) const
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

bool JsonItem::operator==(const JsonItem& rhs) const
{
    return *_value == *rhs._value;
}

} // namespace EdgeLearning

#endif // EDGE_LEARNING_PARSER_JSON_HPP
