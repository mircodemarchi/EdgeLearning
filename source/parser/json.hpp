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

class JsonObject : public Parser
{
public:
    enum JsonType
    {
        LEAF,
        LIST,
        DICT,
        NONE
    };

    using Shared = std::shared_ptr<JsonObject>;
    using Reference = JsonObject&;
    using Pointer = JsonObject*;

    JsonObject()
        : Parser()
        , _json_type(JsonType::NONE)
    { }

    JsonObject(JsonType json_type)
        : Parser()
        , _json_type(json_type)
    { }

    virtual ~JsonObject() {};

    friend std::ostream& operator<<(std::ostream& os, const JsonObject& obj);
    friend std::istream& operator>>(std::istream& os, JsonObject& obj);
    bool operator==(const JsonObject& rhs) const;

    JsonType json_type() const { return _json_type; }

protected:
    JsonType _json_type;
};

class JsonLeaf : public JsonObject
{
public:
    JsonLeaf()
        : JsonObject(JsonType::LEAF)
        , _val{}
        , _type{Type::NONE}
    { }

    JsonLeaf(const std::string& val, Type type = Type::AUTO)
        : JsonObject(JsonType::LEAF)
        , _val{std::move(val)}
        , _type{type}
    {
        if (_type == Type::AUTO)
        {
            _type = _tc(_val);
        }
    }

    JsonLeaf(const char* val)
        : JsonLeaf(std::string(val), Type::AUTO)
    { }

    JsonLeaf(int val)
        : JsonLeaf(std::to_string(val), Type::INT)
    { }

    JsonLeaf(double val)
        : JsonLeaf(std::to_string(val), Type::FLOAT)
    { }

    JsonLeaf(bool val)
        : JsonLeaf(val ? "true" : "false", Type::BOOL)
    { }

    ~JsonLeaf() = default;

    template<typename T>
    void value(T val)
    {
        _val = std::to_string(val);
        _type = _tc(_val);
    }
    [[nodiscard]] const std::string& value() const { return _val; }

    /**
     * \brief Convert the field in the templated type and put in the ptr.
     * \tparam T  Field type requested.
     * \param ptr Pointer in which put the result.
     */
    template<typename T>
    void as(T *ptr) const
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
     * \brief Field type getter.
     * \return const Type&
     */
    [[nodiscard]] const Type& type() const { return _type; }

    bool operator==(const JsonLeaf& rhs) const;

    friend std::ostream& operator<<(std::ostream& os, const JsonLeaf& obj);
    friend std::istream& operator>>(std::istream& os, JsonLeaf& obj);

private:
    std::string _val;
    Type _type;
};

class JsonList : public JsonObject
{
public:
    JsonList(std::vector<JsonItem> list = {})
        : JsonObject(JsonType::LIST)
        , _list{std::move(list)}
    { }

    ~JsonList() = default;

    const JsonItem& operator[](std::size_t idx) const
    {
        if (idx >= _list.size())
        {
            throw std::runtime_error(
                "operator[] failed: idx >= list size");
        }
        return _list[idx];
    }

    void append(JsonItem jd);

    std::size_t size() const { return _list.size(); }
    bool empty() const { return _list.empty(); }

    bool operator==(const JsonList& rhs) const;

    friend std::ostream& operator<<(std::ostream& os, const JsonList& obj);
    friend std::istream& operator>>(std::istream& os, JsonList& obj);

private:
    std::vector<JsonItem> _list;
};

class JsonDict : public JsonObject
{
public:
    JsonDict(std::map<std::string, JsonItem> map = {})
        : JsonObject(JsonType::DICT)
        , _map{map}
    { }

    ~JsonDict() = default;

    const JsonItem& at(std::string key) const
    {
        if (!_map.contains(key))
        {
            throw std::runtime_error(
                "operator[] failed: idx not contained in dict");
        }
        return _map.at(key);
    }
    JsonItem& operator[](std::string key)
    {
        return _map[key];
    }

    std::size_t size() const { return _map.size(); }
    bool empty() const { return _map.empty(); }

    bool operator==(const JsonDict& rhs) const;

    friend std::ostream& operator<<(std::ostream& os, const JsonDict& obj);
    friend std::istream& operator>>(std::istream& os, JsonDict& obj);

private:
    std::map<std::string, JsonItem> _map;
};

class JsonItem : public JsonObject
{
    friend class Json;

public:
    JsonItem()
        : JsonObject()
        , _value{}
    { }

    JsonItem(JsonLeaf value)
        : JsonObject()
        , _value(std::make_shared<JsonLeaf>(value))
    { }
    JsonItem(int value)
        : JsonItem(JsonLeaf(value))
    { }
    JsonItem(double value)
        : JsonItem(JsonLeaf(value))
    { }
    JsonItem(bool value)
        : JsonItem(JsonLeaf(value))
    { }
    JsonItem(std::string value)
        : JsonItem(JsonLeaf(value))
    { }
    JsonItem(const char* value)
        : JsonItem(JsonLeaf(value))
    { }

    JsonItem(JsonList value)
        : JsonObject()
        , _value(std::make_shared<JsonList>(value))
    { }
    JsonItem(std::vector<JsonItem> value)
        : JsonItem(JsonList(value))
    { }

    JsonItem(JsonDict value)
        : JsonObject()
        , _value(std::make_shared<JsonDict>(value))
    { }
    JsonItem(std::map<std::string, JsonItem> value)
        : JsonItem(JsonDict(value))
    { }

    ~JsonItem() = default;

    JsonItem(const JsonItem& obj);
    JsonItem& operator=(JsonItem obj);

    JsonItem& operator=(const JsonLeaf& ji);
    JsonItem& operator=(const int& val);
    JsonItem& operator=(const double& val);
    JsonItem& operator=(const bool& val);
    JsonItem& operator=(const std::string& val);
    JsonItem& operator=(const char* val);
    JsonItem& operator=(const JsonList& jl);
    JsonItem& operator=(const std::vector<JsonItem>& jl);
    JsonItem& operator=(const JsonDict& jd);
    JsonItem& operator=(const std::map<std::string, JsonItem>& jd);
    JsonItem& operator=(Json j);

    [[nodiscard]] const JsonObject& value() const
    {
        if (!_value) throw std::runtime_error("value failed: empty object");
        return *_value;
    }

    bool operator==(const JsonItem& rhs) const;

    friend std::ostream& operator<<(std::ostream& os, const JsonItem& obj);
    friend std::istream& operator>>(std::istream& os, JsonItem& obj);

private:
    Shared _value;
};


class Json : JsonObject
{
    friend JsonItem;

public:
    Json()
        : JsonObject(JsonType::NONE)
        , _obj{}
        , _type{JsonObject::JsonType::NONE}
    { }

    Json(JsonList jl)
        : JsonObject()
        , _obj{jl}
        , _type{JsonObject::JsonType::LIST}
    { }
    Json(std::vector<JsonItem> value)
        : Json(JsonList(value))
    { }

    Json(JsonDict jd)
        : JsonObject()
        , _obj{jd}
        , _type{JsonObject::JsonType::DICT}
    { }
    Json(std::map<std::string, JsonItem> value)
        : Json(JsonDict(value))
    { }

    void append(JsonItem obj)
    {
        if (_type == JsonObject::JsonType::NONE)
        {
            _type = JsonObject::JsonType::LIST;
            _obj = JsonItem(JsonList());
        }
        auto jl = std::dynamic_pointer_cast<JsonList>(_obj._value);
        if (!jl) throw std::runtime_error("Json is not a list");
        jl->append(obj);
    }
    void append(Json obj)
    {
        append(obj._obj);
    }

    JsonItem& operator[](std::string key)
    {
        if (_type == JsonObject::JsonType::NONE)
        {
            _type = JsonObject::JsonType::DICT;
            _obj = JsonItem(JsonDict());
        }
        auto jd = std::dynamic_pointer_cast<JsonDict>(_obj._value);
        if (!jd) throw std::runtime_error("Json is not a dictionary");
        return (*jd)[key];
    }

    bool operator==(const Json& rhs) const
    {
        return _obj == rhs._obj;
    }

    friend std::ostream& operator<<(std::ostream& os, const Json& obj)
    {
        os << obj._obj;
        return os;
    }

    friend std::istream& operator>>(std::istream& os, Json& obj)
    {
        os >> obj._obj;
        return os;
    }

private:
    JsonItem _obj;
    JsonType _type;
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

JsonItem& JsonItem::operator=(const JsonLeaf& ji)
{
    _value = std::make_shared<JsonLeaf>(ji);
    return *this;
}
JsonItem& JsonItem::operator=(const int& val)
{
    return operator=(JsonLeaf(val));
}
JsonItem& JsonItem::operator=(const double& val)
{
    return operator=(JsonLeaf(val));
}
JsonItem& JsonItem::operator=(const bool& val)
{
    return operator=(JsonLeaf(val));
}
JsonItem& JsonItem::operator=(const std::string& val)
{
    return operator=(JsonLeaf(val));
}
JsonItem& JsonItem::operator=(const char* val)
{
    return operator=(JsonLeaf(val));
}
JsonItem& JsonItem::operator=(const JsonList& jl)
{
    _value = std::make_shared<JsonList>(jl);
    return *this;
}
JsonItem& JsonItem::operator=(const std::vector<JsonItem>& jl)
{
    return operator=(JsonList(jl));
}
JsonItem& JsonItem::operator=(const JsonDict& jd)
{
    _value = std::make_shared<JsonDict>(jd);
    return *this;
}
JsonItem& JsonItem::operator=(const std::map<std::string, JsonItem>& jd)
{
    return operator=(JsonDict(jd));
}
JsonItem& JsonItem::operator=(Json j)
{
    return operator=(j._obj);
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
