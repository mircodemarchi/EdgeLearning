/***************************************************************************
 *            parser/test_json.cpp
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

#include "test.hpp"
#include "parser/json.hpp"

#include <filesystem>
#include <stdexcept>

using namespace std;
using namespace EdgeLearning;

class TestJson {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_json_object());
        EDGE_LEARNING_TEST_CALL(test_json_leaf());
        EDGE_LEARNING_TEST_CALL(test_json_list());
        EDGE_LEARNING_TEST_CALL(test_json_dict());
        EDGE_LEARNING_TEST_CALL(test_json_item());
        EDGE_LEARNING_TEST_CALL(test_json());
        EDGE_LEARNING_TEST_CALL(test_stream());
    }

private:
    void test_json_object()
    {
        JsonLeaf j_leaf;
        JsonObject jo_leaf(JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_NOT_EQUAL(
            dynamic_cast<const JsonObject&>(j_leaf), jo_leaf);
        JsonList j_list;
        JsonObject jo_list(JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_NOT_EQUAL(
            dynamic_cast<const JsonObject&>(j_list), jo_list);
        JsonDict j_dict;
        JsonObject jo_dict(JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_NOT_EQUAL(
            dynamic_cast<const JsonObject&>(j_dict), jo_dict);
        JsonObject j_obj;
        EDGE_LEARNING_TEST_NOT_EQUAL(j_obj, j_obj);
        JsonObject j_copy(jo_dict);
        EDGE_LEARNING_TEST_NOT_EQUAL(j_copy, jo_dict);
        EDGE_LEARNING_TEST_EQUAL(j_copy.json_type(), jo_dict.json_type());
    }

    void test_json_leaf() {
        JsonLeaf jl = JsonLeaf("10");
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(), JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(jl.value(), "10");
        EDGE_LEARNING_TEST_EQUAL(jl.type(), Type::INT);
        EDGE_LEARNING_TEST_EQUAL(jl.as<int>(), 10);
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(jl), 10);
        EDGE_LEARNING_TEST_EQUAL(jl.as<float>(), 10.0);
        EDGE_LEARNING_TEST_EQUAL(static_cast<double>(jl), 10);
        EDGE_LEARNING_TEST_EQUAL(jl.as<bool>(), false);
        EDGE_LEARNING_TEST_EQUAL(static_cast<bool>(jl), false);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jl), std::string("10"));
        jl = JsonLeaf("1.0");
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(), JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(jl.value(), "1.0");
        EDGE_LEARNING_TEST_EQUAL(jl.type(), Type::FLOAT);
        EDGE_LEARNING_TEST_EQUAL(jl.as<int>(), 1);
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(jl), 1);
        EDGE_LEARNING_TEST_EQUAL(jl.as<float>(), 1.0);
        EDGE_LEARNING_TEST_EQUAL(static_cast<double>(jl), 1.0);
        EDGE_LEARNING_TEST_EQUAL(jl.as<bool>(), false);
        EDGE_LEARNING_TEST_EQUAL(static_cast<bool>(jl), false);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jl), "1.0");
        jl = JsonLeaf("true");
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(), JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(jl.value(), "true");
        EDGE_LEARNING_TEST_EQUAL(jl.type(), Type::BOOL);
        EDGE_LEARNING_TEST_EQUAL(jl.as<int>(), 0);
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(jl), 0);
        EDGE_LEARNING_TEST_EQUAL(jl.as<float>(), 0.0);
        EDGE_LEARNING_TEST_EQUAL(static_cast<double>(jl), 0.0);
        EDGE_LEARNING_TEST_EQUAL(jl.as<bool>(), true);
        EDGE_LEARNING_TEST_EQUAL(static_cast<bool>(jl), true);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jl), "true");

        JsonLeaf i = JsonLeaf(int(10));
        EDGE_LEARNING_TEST_EQUAL(static_cast<unsigned>(JsonLeaf(10U)), 10U);
        EDGE_LEARNING_TEST_EQUAL(static_cast<long>(JsonLeaf(10L)), 10L);
        EDGE_LEARNING_TEST_EQUAL(
            static_cast<unsigned long>(JsonLeaf(10UL)), 10UL);
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(), JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(i.value(), "10");
        EDGE_LEARNING_TEST_EQUAL(i.type(), Type::INT);
        EDGE_LEARNING_TEST_EQUAL(i.as<int>(), 10);
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(i), 10);
        EDGE_LEARNING_TEST_EQUAL(i.as<float>(), 10.0);
        EDGE_LEARNING_TEST_EQUAL(static_cast<double>(i), 10.0);
        EDGE_LEARNING_TEST_EQUAL(i.as<bool>(), false);
        EDGE_LEARNING_TEST_EQUAL(static_cast<bool>(i), false);
        JsonLeaf f = JsonLeaf(float(1.0));
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(), JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_PRINT(f.value());
        EDGE_LEARNING_TEST_EQUAL(f.value(), "1.000000");
        EDGE_LEARNING_TEST_EQUAL(f.type(), Type::FLOAT);
        EDGE_LEARNING_TEST_EQUAL(f.as<int>(), 1);
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(f), 1);
        EDGE_LEARNING_TEST_EQUAL(f.as<float>(), 1.0);
        EDGE_LEARNING_TEST_EQUAL(static_cast<double>(f), 1.0);
        EDGE_LEARNING_TEST_EQUAL(f.as<bool>(), false);
        EDGE_LEARNING_TEST_EQUAL(static_cast<bool>(f), false);
        JsonLeaf b = JsonLeaf(true);
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(), JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(b.value(), "true");
        EDGE_LEARNING_TEST_EQUAL(b.type(), Type::BOOL);
        EDGE_LEARNING_TEST_EQUAL(b.as<int>(), 0);
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(b), 0);
        EDGE_LEARNING_TEST_EQUAL(b.as<float>(), 0.0);
        EDGE_LEARNING_TEST_EQUAL(static_cast<double>(b), 0.0);
        EDGE_LEARNING_TEST_EQUAL(b.as<bool>(), true);
        EDGE_LEARNING_TEST_EQUAL(static_cast<bool>(b), true);

        JsonLeaf jl1 = JsonLeaf(5);
        JsonLeaf jl2 = JsonLeaf(true);
        EDGE_LEARNING_TEST_PRINT(jl1);
        EDGE_LEARNING_TEST_PRINT(jl2);
        EDGE_LEARNING_TEST_EQUAL(jl1.type(), Type::INT);
        EDGE_LEARNING_TEST_EQUAL(jl2.type(), Type::BOOL);
        EDGE_LEARNING_TEST_ASSERT(jl1 != jl2);
        jl2.value(5.0);
        EDGE_LEARNING_TEST_EQUAL(jl2.type(), Type::FLOAT);
        EDGE_LEARNING_TEST_ASSERT(jl1 != jl2);
        jl2.value(5);
        EDGE_LEARNING_TEST_EQUAL(jl2.type(), Type::INT);
        EDGE_LEARNING_TEST_ASSERT(jl1 == jl2);
    }

    void test_json_list() {
        JsonList jl_empty;
        EDGE_LEARNING_TEST_EQUAL(jl_empty.json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_ASSERT(jl_empty.empty());
        EDGE_LEARNING_TEST_FAIL(jl_empty[0]);
        EDGE_LEARNING_TEST_THROWS(jl_empty[0], std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(jl_empty[10]);
        EDGE_LEARNING_TEST_THROWS(jl_empty[10], std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jl_empty), "[]");

        JsonList jl = JsonList(
            std::vector<JsonItem>({10, 10.0, "string", true}));
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_PRINT(jl);
        EDGE_LEARNING_TEST_PRINT(jl[0]);
        EDGE_LEARNING_TEST_ASSERT(!jl.empty());
        EDGE_LEARNING_TEST_EQUAL(jl.size(), 4);
        EDGE_LEARNING_TEST_FAIL(jl[10]);
        EDGE_LEARNING_TEST_THROWS(jl[10], std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jl),
                                 "[10,10.000000,\"string\",true]");
        EDGE_LEARNING_TEST_EQUAL(
            static_cast<std::vector<std::string>>(jl).size(), jl.size());
        auto jl_vec = static_cast<std::vector<std::string>>(jl);
        for (std::size_t i = 0; i < jl_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(jl_vec[i],
                                     static_cast<std::string>(jl[i]));
        }

        jl = JsonList(std::vector<int>({10, 11, 12, 13}));
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_PRINT(jl);
        EDGE_LEARNING_TEST_PRINT(jl[0]);
        EDGE_LEARNING_TEST_ASSERT(!jl.empty());
        EDGE_LEARNING_TEST_EQUAL(jl.size(), 4);
        EDGE_LEARNING_TEST_FAIL(jl[10]);
        EDGE_LEARNING_TEST_THROWS(jl[10], std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jl),
                                 "[10,11,12,13]");
        EDGE_LEARNING_TEST_EQUAL(
            static_cast<std::vector<std::string>>(jl).size(), jl.size());
        jl_vec = static_cast<std::vector<std::string>>(jl);
        for (std::size_t i = 0; i < jl_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(jl_vec[i],
                                     static_cast<std::string>(jl[i]));
        }
        jl = JsonList(std::vector<unsigned int>({10U, 11U, 12U, 13U}));
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_ASSERT(!jl.empty());
        EDGE_LEARNING_TEST_EQUAL(jl.size(), 4);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jl),
                                 "[10,11,12,13]");
        jl = JsonList(std::vector<long>({10L, 11L, 12L, 13L}));
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_ASSERT(!jl.empty());
        EDGE_LEARNING_TEST_EQUAL(jl.size(), 4);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jl),
                                 "[10,11,12,13]");
        jl = JsonList(std::vector<unsigned long>({10UL, 11UL, 12UL, 13UL}));
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_ASSERT(!jl.empty());
        EDGE_LEARNING_TEST_EQUAL(jl.size(), 4);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jl),
                                 "[10,11,12,13]");

        jl = JsonList(std::vector<double>({10.0, 11.0, 12.5, 13.5}));
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_PRINT(jl);
        EDGE_LEARNING_TEST_PRINT(jl[0]);
        EDGE_LEARNING_TEST_ASSERT(!jl.empty());
        EDGE_LEARNING_TEST_EQUAL(jl.size(), 4);
        EDGE_LEARNING_TEST_FAIL(jl[10]);
        EDGE_LEARNING_TEST_THROWS(jl[10], std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jl),
                                 "[10.000000,11.000000,12.500000,13.500000]");
        EDGE_LEARNING_TEST_EQUAL(
            static_cast<std::vector<std::string>>(jl).size(), jl.size());
        jl_vec = static_cast<std::vector<std::string>>(jl);
        for (std::size_t i = 0; i < jl_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(jl_vec[i],
                                     static_cast<std::string>(jl[i]));
        }

        jl = JsonList(std::vector<bool>({true, false, true, true}));
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_PRINT(jl);
        EDGE_LEARNING_TEST_PRINT(jl[0]);
        EDGE_LEARNING_TEST_ASSERT(!jl.empty());
        EDGE_LEARNING_TEST_EQUAL(jl.size(), 4);
        EDGE_LEARNING_TEST_FAIL(jl[10]);
        EDGE_LEARNING_TEST_THROWS(jl[10], std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jl),
                                 "[true,false,true,true]");
        EDGE_LEARNING_TEST_EQUAL(
            static_cast<std::vector<std::string>>(jl).size(), jl.size());
        jl_vec = static_cast<std::vector<std::string>>(jl);
        for (std::size_t i = 0; i < jl_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(jl_vec[i],
                                     static_cast<std::string>(jl[i]));
        }

        JsonList jl_int = JsonList(
            std::vector<JsonItem>({10, 11, 12, 13}));
        auto jl_vec_int = static_cast<std::vector<int>>(jl_int);
        for (std::size_t i = 0; i < jl_vec_int.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(jl_vec_int[i],
                                     static_cast<int>(jl_int[i]));
        }

        JsonList jl1 = JsonList(
            std::vector<JsonItem>({10, 10.0, "string", true}));
        JsonList jl2 = JsonList(
            std::vector<JsonItem>({10, 10.0, "string", true, "1_more"}));
        EDGE_LEARNING_TEST_ASSERT(jl1 != jl2);
        EDGE_LEARNING_TEST_TRY(jl1.append("1_more"));
        EDGE_LEARNING_TEST_ASSERT(jl1 == jl2);
        EDGE_LEARNING_TEST_ASSERT(jl1[4] == jl2[4]);
    }

    void test_json_dict() {
        JsonDict jd_empty;
        EDGE_LEARNING_TEST_EQUAL(jd_empty.json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_ASSERT(jd_empty.empty());
        EDGE_LEARNING_TEST_FAIL(jd_empty.at("key"));
        EDGE_LEARNING_TEST_THROWS(jd_empty.at("key"), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jd_empty), "{}");

        JsonDict jd = JsonDict(
            std::map<std::string, JsonItem>({{"a", 0}, {"b", true}}));
        EDGE_LEARNING_TEST_EQUAL(jd.json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_PRINT(jd);
        EDGE_LEARNING_TEST_PRINT(jd["a"]);
        EDGE_LEARNING_TEST_EQUAL(jd.at("a"), jd["a"]);
        EDGE_LEARNING_TEST_ASSERT(!jd.empty());
        EDGE_LEARNING_TEST_EQUAL(jd.size(), 2);
        EDGE_LEARNING_TEST_FAIL(jd.at("key"));
        EDGE_LEARNING_TEST_THROWS(jd.at("key"), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jd),
                                 "{\"a\":0,\"b\":true}");
        auto jd_map = static_cast<std::map<std::string, std::string>>(jd);
        EDGE_LEARNING_TEST_EQUAL(jd_map.size(), jd.size());
        for (const auto& e: jd_map)
        {
            EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jd[e.first]),
                                     e.second);
        }

        jd = JsonDict(std::map<std::string, int>({{"a", 1}, {"b", 2}}));
        EDGE_LEARNING_TEST_EQUAL(jd.json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_PRINT(jd);
        EDGE_LEARNING_TEST_PRINT(jd["a"]);
        EDGE_LEARNING_TEST_EQUAL(jd.at("a"), jd["a"]);
        EDGE_LEARNING_TEST_ASSERT(!jd.empty());
        EDGE_LEARNING_TEST_EQUAL(jd.size(), 2);
        EDGE_LEARNING_TEST_FAIL(jd.at("key"));
        EDGE_LEARNING_TEST_THROWS(jd.at("key"), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jd),
                                 "{\"a\":1,\"b\":2}");
        jd_map = static_cast<std::map<std::string, std::string>>(jd);
        EDGE_LEARNING_TEST_EQUAL(jd_map.size(), jd.size());
        for (const auto& e: jd_map)
        {
            EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jd[e.first]),
                                     e.second);
        }
        jd = JsonDict(
            std::map<std::string, unsigned int>({{"a", 1U}, {"b", 2U}}));
        EDGE_LEARNING_TEST_EQUAL(jd.json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_EQUAL(jd.at("a"), jd["a"]);
        EDGE_LEARNING_TEST_ASSERT(!jd.empty());
        EDGE_LEARNING_TEST_EQUAL(jd.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jd),
                                 "{\"a\":1,\"b\":2}");
        jd = JsonDict(
            std::map<std::string, long>({{"a", 1L}, {"b", 2L}}));
        EDGE_LEARNING_TEST_EQUAL(jd.json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_EQUAL(jd.at("a"), jd["a"]);
        EDGE_LEARNING_TEST_ASSERT(!jd.empty());
        EDGE_LEARNING_TEST_EQUAL(jd.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jd),
                                 "{\"a\":1,\"b\":2}");
        jd = JsonDict(
            std::map<std::string, unsigned long>({{"a", 1UL}, {"b", 2UL}}));
        EDGE_LEARNING_TEST_EQUAL(jd.json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_EQUAL(jd.at("a"), jd["a"]);
        EDGE_LEARNING_TEST_ASSERT(!jd.empty());
        EDGE_LEARNING_TEST_EQUAL(jd.size(), 2);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jd),
                                 "{\"a\":1,\"b\":2}");

        jd = JsonDict(
            std::map<std::string, double>({{"a", 1.0}, {"b", 2.5}}));
        EDGE_LEARNING_TEST_EQUAL(jd.json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_PRINT(jd);
        EDGE_LEARNING_TEST_PRINT(jd["a"]);
        EDGE_LEARNING_TEST_EQUAL(jd.at("a"), jd["a"]);
        EDGE_LEARNING_TEST_ASSERT(!jd.empty());
        EDGE_LEARNING_TEST_EQUAL(jd.size(), 2);
        EDGE_LEARNING_TEST_FAIL(jd.at("key"));
        EDGE_LEARNING_TEST_THROWS(jd.at("key"), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jd),
                                 "{\"a\":1.000000,\"b\":2.500000}");
        jd_map = static_cast<std::map<std::string, std::string>>(jd);
        EDGE_LEARNING_TEST_EQUAL(jd_map.size(), jd.size());
        for (const auto& e: jd_map)
        {
            EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jd[e.first]),
                                     e.second);
        }

        jd = JsonDict(
            std::map<std::string, bool>({{"a", true}, {"b", false}}));
        EDGE_LEARNING_TEST_EQUAL(jd.json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_PRINT(jd);
        EDGE_LEARNING_TEST_PRINT(jd["a"]);
        EDGE_LEARNING_TEST_EQUAL(jd.at("a"), jd["a"]);
        EDGE_LEARNING_TEST_ASSERT(!jd.empty());
        EDGE_LEARNING_TEST_EQUAL(jd.size(), 2);
        EDGE_LEARNING_TEST_FAIL(jd.at("key"));
        EDGE_LEARNING_TEST_THROWS(jd.at("key"), std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jd),
                                 "{\"a\":true,\"b\":false}");
        jd_map = static_cast<std::map<std::string, std::string>>(jd);
        EDGE_LEARNING_TEST_EQUAL(jd_map.size(), jd.size());
        for (const auto& e: jd_map)
        {
            EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(jd[e.first]),
                                     e.second);
        }

        JsonDict jd_int = JsonDict(
            std::map<std::string, JsonItem>({{"a", 0}, {"b", 1}}));
        auto jd_map_int = static_cast<std::map<std::string, int>>(jd_int);
        for (const auto& e: jd_map_int)
        {
            EDGE_LEARNING_TEST_EQUAL(static_cast<int>(jd_int[e.first]),
                                     e.second);
        }

        JsonDict jd1 = JsonDict(
            std::map<std::string, JsonItem>({{"a", 0}, {"b", true}}));
        JsonDict jd2 = JsonDict(
            std::map<std::string, JsonItem>({{"a", 0}, {"b", true}, {"c", "1_more"}}));
        EDGE_LEARNING_TEST_ASSERT(jd1 != jd2);
        EDGE_LEARNING_TEST_TRY(jd1["c"] = "1_more");
        EDGE_LEARNING_TEST_ASSERT(jd1 == jd2);
        EDGE_LEARNING_TEST_ASSERT(jd1["c"] == jd2["c"]);
    }

    void test_json_item() {
        using DictStringInt = std::map<std::string, int>;
        using DictStringString = std::map<std::string, std::string>;

        JsonItem ji_empty;
        EDGE_LEARNING_TEST_PRINT(ji_empty);
        EDGE_LEARNING_TEST_EQUAL(ji_empty.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_FAIL(ji_empty.value());
        EDGE_LEARNING_TEST_THROWS(ji_empty.value(), std::runtime_error);
        EDGE_LEARNING_TEST_ASSERT(static_cast<std::string>(ji_empty).empty());
        EDGE_LEARNING_TEST_FAIL(ji_empty[0]);
        EDGE_LEARNING_TEST_THROWS(ji_empty[0], std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(ji_empty[10]);
        EDGE_LEARNING_TEST_THROWS(ji_empty[10], std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(ji_empty["key"]);
        EDGE_LEARNING_TEST_THROWS(ji_empty["key"], std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(ji_empty.size(), 0);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<int>(ji_empty));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<int>(ji_empty),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<double>(ji_empty));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<double>(ji_empty),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<bool>(ji_empty));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<bool>(ji_empty),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<std::vector<int>>(ji_empty));
        EDGE_LEARNING_TEST_THROWS(
            (void) static_cast<std::vector<int>>(ji_empty), std::runtime_error);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<DictStringInt>(ji_empty));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<DictStringInt>(ji_empty),
                                  std::runtime_error);

        EDGE_LEARNING_TEST_TRY(JsonItem ji_empty_copy(ji_empty));
        JsonObject j_none; JsonItem ji_none(j_none);
        EDGE_LEARNING_TEST_TRY(JsonItem ji_empty_copy(ji_none));
        EDGE_LEARNING_TEST_ASSERT(static_cast<std::string>(ji_none).empty());

        JsonItem ji(10);
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(static_cast<int>(ji) == 10);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji), "10");
        EDGE_LEARNING_TEST_FAIL(ji[0]);
        EDGE_LEARNING_TEST_THROWS(ji[0], std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(ji[10]);
        EDGE_LEARNING_TEST_THROWS(ji[10], std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(ji["key"]);
        EDGE_LEARNING_TEST_THROWS(ji["key"], std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(ji_empty.size(), 0);
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ji), 10);
        EDGE_LEARNING_TEST_EQUAL(static_cast<double>(ji), 10.0);
        EDGE_LEARNING_TEST_EQUAL(static_cast<bool>(ji), false);
        EDGE_LEARNING_TEST_FAIL(
            (void) static_cast<std::vector<std::string>>(ji));
        EDGE_LEARNING_TEST_THROWS(
            (void) static_cast<std::vector<std::string>>(ji),
            std::runtime_error);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<DictStringString>(ji));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<DictStringString>(ji),
                                  std::runtime_error);
        ji = JsonItem(10);
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(static_cast<int>(ji) == 10);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji), "10");
        ji = JsonItem(1.0);
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(static_cast<double>(ji) == 1.0);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji), "1.000000");
        ji = JsonItem(false);
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(static_cast<bool>(ji) == false);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji), "false");
        ji = JsonItem("stringtest");
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(static_cast<std::string>(ji) == "stringtest");
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        ji = JsonItem(std::string("stringtest"));
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(
            static_cast<std::string>(ji) == std::string("stringtest"));
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        ji = 10;
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(static_cast<int>(ji) == 10);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji), "10");
        ji = 10U;
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(static_cast<unsigned int>(ji) == 10U);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji), "10");
        ji = 10L;
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(static_cast<long>(ji) == 10L);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji), "10");
        ji = 10UL;
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(static_cast<unsigned long>(ji) == 10UL);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji), "10");
        ji = 1.0;
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(static_cast<double>(ji) == 1.0);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji), "1.000000");
        ji = false;
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(static_cast<bool>(ji) == false);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji), "false");
        ji = "stringtest";
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(
            static_cast<std::string>(ji) == "stringtest");
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        ji = std::string("stringtest");
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(
            static_cast<std::string>(ji) == std::string("stringtest"));
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        ji = JsonItem({10, "test1", false, "test1", 1.0});
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(
            ji == JsonItem({10, "test1", false, "test1", 1.0}));
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji),
                                 "[10,\"test1\",false,\"test1\",1.000000]");
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ji[0]), 10);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji[1]), "test1");
        EDGE_LEARNING_TEST_EQUAL(static_cast<bool>(ji[2]), false);
        EDGE_LEARNING_TEST_EQUAL(static_cast<double>(ji[4]), 1.0);
        EDGE_LEARNING_TEST_FAIL(ji["key"]);
        EDGE_LEARNING_TEST_THROWS(ji["key"], std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(ji.size(), 5);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<int>(ji));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<int>(ji),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<double>(ji));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<double>(ji),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<bool>(ji));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<bool>(ji),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_TRY(
            (void) static_cast<std::vector<std::string>>(ji));
        auto ji_vec = static_cast<std::vector<std::string>>(ji);
        EDGE_LEARNING_TEST_EQUAL(
            static_cast<std::vector<std::string>>(ji).size(), ji.size());
        for (std::size_t i = 0; i < ji_vec.size(); ++i)
        {
            EDGE_LEARNING_TEST_EQUAL(ji_vec[i],
                                     static_cast<std::string>(ji[i]));
        }
        EDGE_LEARNING_TEST_FAIL((void) static_cast<DictStringString>(ji));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<DictStringString>(ji),
                                  std::runtime_error);
        ji = {10, "test2", false, "test2", 1.0};
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(
            ji == JsonItem({10, "test2", false, "test2", 1.0}));
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji),
                                 "[10,\"test2\",false,\"test2\",1.000000]");
        ji = JsonItem({{"test1a", 10}, {"test1b", false}});
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(
            ji == JsonItem({{"test1a", 10}, {"test1b", false}}));
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji),
                                 "{\"test1a\":10,\"test1b\":false}");
        EDGE_LEARNING_TEST_FAIL(ji[0]);
        EDGE_LEARNING_TEST_THROWS(ji[0], std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(ji[10]);
        EDGE_LEARNING_TEST_THROWS(ji[10], std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(static_cast<int>(ji["test1a"]), 10);
        EDGE_LEARNING_TEST_EQUAL(bool(ji["test1b"]), false);
        EDGE_LEARNING_TEST_EQUAL(ji.size(), 2);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<int>(ji));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<int>(ji),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<double>(ji));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<double>(ji),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<bool>(ji));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<bool>(ji),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(
            (void) static_cast<std::vector<std::string>>(ji));
        EDGE_LEARNING_TEST_THROWS(
            (void) static_cast<std::vector<std::string>>(ji),
            std::runtime_error);
        EDGE_LEARNING_TEST_TRY((void) static_cast<DictStringString>(ji));
        auto ji_dict = static_cast<DictStringString>(ji);
        EDGE_LEARNING_TEST_EQUAL(
            static_cast<DictStringString>(ji).size(), ji.size());
        for (const auto& e: ji_dict)
        {
            EDGE_LEARNING_TEST_EQUAL(e.second,
                                     static_cast<std::string>(ji[e.first]));
        }
        ji = {{"test2a", 10}, {"test2b", false}};
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(
            ji == JsonItem({{"test2a", 10}, {"test2b", false}}));
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_EQUAL(static_cast<std::string>(ji),
                                 "{\"test2a\":10,\"test2b\":false}");

        ji_empty = ji;
        EDGE_LEARNING_TEST_EQUAL(ji, ji_empty);

        JsonObject jo_leaf = JsonObject(JsonObject::JsonType::LEAF);
        JsonObject jo_list = JsonObject(JsonObject::JsonType::LIST);
        JsonObject jo_dict = JsonObject(JsonObject::JsonType::DICT);
        JsonItem ji_leaf(jo_leaf);
        EDGE_LEARNING_TEST_ASSERT(static_cast<std::string>(ji_leaf).empty());
        EDGE_LEARNING_TEST_EQUAL(ji_leaf.size(), 0);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<int>(ji_leaf));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<int>(ji_leaf),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<bool>(ji_leaf));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<bool>(ji_leaf),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL((void) static_cast<double>(ji_leaf));
        EDGE_LEARNING_TEST_THROWS((void) static_cast<double>(ji_leaf),
                                  std::runtime_error);
        JsonItem ji_list(jo_list);
        EDGE_LEARNING_TEST_ASSERT(static_cast<std::string>(ji_list).empty());
        EDGE_LEARNING_TEST_FAIL(ji_list[0]);
        EDGE_LEARNING_TEST_THROWS(ji_list[0], std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(ji_list[10]);
        EDGE_LEARNING_TEST_THROWS(ji_list[10], std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(ji_list.size(), 0);
        EDGE_LEARNING_TEST_FAIL(
            (void) static_cast<std::vector<std::string>>(ji_list));
        EDGE_LEARNING_TEST_THROWS(
            (void) static_cast<std::vector<std::string>>(ji_list),
            std::runtime_error);
        JsonItem ji_dict1(jo_dict);
        EDGE_LEARNING_TEST_ASSERT(static_cast<std::string>(ji_dict1).empty());
        EDGE_LEARNING_TEST_FAIL(ji_dict1["key"]);
        EDGE_LEARNING_TEST_THROWS(ji_dict1["key"], std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(ji_dict1.size(), 0);
        EDGE_LEARNING_TEST_FAIL(
            (void) static_cast<DictStringString>(ji_dict1));
        EDGE_LEARNING_TEST_THROWS(
            (void) static_cast<DictStringString>(ji_dict1), std::runtime_error);
    }

    void test_json() {
        Json j_list;
        EDGE_LEARNING_TEST_EQUAL(j_list.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_TRY(j_list.append(10));
        EDGE_LEARNING_TEST_EQUAL(j_list.json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_TRY(j_list.append(true));
        EDGE_LEARNING_TEST_TRY(j_list.append("test"));
        EDGE_LEARNING_TEST_TRY(j_list.append(1.0));
        EDGE_LEARNING_TEST_ASSERT(j_list == Json({10, true, "test", 1.0}));
        EDGE_LEARNING_TEST_TRY(j_list.append(Json({{"test", 1}, {"b", 10}})));
        EDGE_LEARNING_TEST_PRINT(j_list);
        EDGE_LEARNING_TEST_FAIL(j_list["fail_test"] = -1);
        EDGE_LEARNING_TEST_THROWS(j_list["fail_test"] = -1,
                                  std::runtime_error);
        EDGE_LEARNING_TEST_EQUAL(j_list[0].as<int>(), 10);
        EDGE_LEARNING_TEST_EQUAL(j_list[1].as<bool>(), true);
        EDGE_LEARNING_TEST_EQUAL(j_list[2].as<std::string>(), "test");
        EDGE_LEARNING_TEST_EQUAL(j_list[3].as<float>(), 1.0);

        Json j_dict;
        EDGE_LEARNING_TEST_EQUAL(j_dict.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_TRY(j_dict["test"] = 1);
        EDGE_LEARNING_TEST_EQUAL(j_dict.json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_TRY(j_dict["b"] = 10);
        EDGE_LEARNING_TEST_ASSERT(j_dict == Json({{"test", 1}, {"b", 10}}));
        EDGE_LEARNING_TEST_TRY(j_dict["list"] = Json({10, true, "test", 1.0}));
        EDGE_LEARNING_TEST_PRINT(j_dict);
        EDGE_LEARNING_TEST_FAIL(j_dict.append({"fail_test"}));
        EDGE_LEARNING_TEST_THROWS(j_dict.append({"fail_test"}),
                                  std::runtime_error);
        EDGE_LEARNING_TEST_FAIL(j_dict[0] = -1);
        EDGE_LEARNING_TEST_THROWS(j_dict[0] = -1, std::runtime_error);

        Json j_comb1 = j_list;
        EDGE_LEARNING_TEST_EQUAL(j_comb1.json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_TRY(j_comb1.append(j_dict));
        EDGE_LEARNING_TEST_PRINT(j_comb1);
        EDGE_LEARNING_TEST_TRY(j_comb1.append(j_list));
        EDGE_LEARNING_TEST_PRINT(j_comb1);
        EDGE_LEARNING_TEST_TRY(j_comb1.append(j_comb1));
        EDGE_LEARNING_TEST_PRINT(j_comb1);

        Json j_comb2 = j_dict;
        EDGE_LEARNING_TEST_EQUAL(j_comb2.json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_TRY(j_comb2["j_list"] = j_list);
        EDGE_LEARNING_TEST_PRINT(j_comb2);
        EDGE_LEARNING_TEST_TRY(j_comb2["j_dict"] = j_dict);
        EDGE_LEARNING_TEST_PRINT(j_comb2);
        EDGE_LEARNING_TEST_TRY(j_comb2["j_comb"] = j_comb2);
        EDGE_LEARNING_TEST_PRINT(j_comb2);

        Json j_item(j_list[0]);
        EDGE_LEARNING_TEST_EQUAL(j_item.json_type(),
                                 JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_PRINT(j_item);
        EDGE_LEARNING_TEST_EQUAL(j_item, Json(10));
        EDGE_LEARNING_TEST_EQUAL(JsonItem(j_item), j_list[0]);
        EDGE_LEARNING_TEST_EQUAL(std::string(j_item),
                                 std::string(j_list[0]));
    }

    void test_stream()
    {
        JsonObject jo_in, jo_out;
        EDGE_LEARNING_TEST_ASSERT(jo_in != jo_out);
        std::ofstream ofs("tmp.json", std::ofstream::trunc);
        EDGE_LEARNING_TEST_TRY(ofs << jo_out);
        ofs.close();
        std::ifstream ifs("tmp.json");
        EDGE_LEARNING_TEST_TRY(ifs >> jo_in);
        ifs.close();

        JsonLeaf jl_in;

        JsonLeaf jl_out = 10;
        ofs = std::ofstream("tmp.json", std::ofstream::trunc);
        EDGE_LEARNING_TEST_TRY(ofs << jl_out);
        ofs.close();

        ifs = std::ifstream("tmp.json");
        EDGE_LEARNING_TEST_TRY(ifs >> jl_in);
        ifs.close();
        EDGE_LEARNING_TEST_PRINT(jl_in);
        EDGE_LEARNING_TEST_EQUAL(jl_in, jl_out);
        EDGE_LEARNING_TEST_EQUAL(jl_in.type(), Type::INT);
        EDGE_LEARNING_TEST_EQUAL(jl_in.json_type(),
                                 JsonObject::JsonType::LEAF);

        jl_out = "string";
        ofs = std::ofstream("tmp.json", std::ofstream::trunc);
        EDGE_LEARNING_TEST_TRY(ofs << jl_out);
        ofs.close();

        ifs = std::ifstream("tmp.json");
        EDGE_LEARNING_TEST_TRY(ifs >> jl_in);
        ifs.close();
        EDGE_LEARNING_TEST_PRINT(jl_in);
        EDGE_LEARNING_TEST_EQUAL(jl_in, jl_out);
        EDGE_LEARNING_TEST_EQUAL(jl_in.type(), Type::STRING);
        EDGE_LEARNING_TEST_EQUAL(jl_in.json_type(),
                                 JsonObject::JsonType::LEAF);

        JsonList jlist_in;

        JsonList jlist_out(std::vector<JsonItem>({"string", 10, 1.0, true}));
        ofs = std::ofstream("tmp.json", std::ofstream::trunc);
        EDGE_LEARNING_TEST_TRY(ofs << jlist_out);
        ofs.close();

        ifs = std::ifstream("tmp.json");
        EDGE_LEARNING_TEST_TRY(ifs >> jlist_in);
        ifs.close();
        EDGE_LEARNING_TEST_PRINT(jlist_in);
        EDGE_LEARNING_TEST_EQUAL(jlist_in, jlist_out);
        EDGE_LEARNING_TEST_EQUAL(jlist_in.json_type(),
                                 JsonObject::JsonType::LIST);

        JsonDict jdict_in;

        JsonDict jdict_out(
            std::map<std::string, JsonItem>({{"a", 10}, {"b", "string"}, {"c", true}}));
        ofs = std::ofstream("tmp.json", std::ofstream::trunc);
        EDGE_LEARNING_TEST_TRY(ofs << jdict_out);
        ofs.close();

        ifs = std::ifstream("tmp.json");
        EDGE_LEARNING_TEST_TRY(ifs >> jdict_in);
        ifs.close();
        EDGE_LEARNING_TEST_PRINT(jdict_in);
        EDGE_LEARNING_TEST_EQUAL(jdict_in, jdict_out);
        EDGE_LEARNING_TEST_EQUAL(jdict_in.json_type(),
                                 JsonObject::JsonType::DICT);

        Json j_comp1_in;

        Json j_comp1_out(jlist_in);
        j_comp1_out.append(Json(jlist_in));
        j_comp1_out.append(Json(jdict_in));
        ofs = std::ofstream("tmp.json", std::ofstream::trunc);
        EDGE_LEARNING_TEST_TRY(ofs << j_comp1_out);
        ofs.close();

        ifs = std::ifstream("tmp.json");
        EDGE_LEARNING_TEST_TRY(ifs >> j_comp1_in);
        ifs.close();
        EDGE_LEARNING_TEST_PRINT(j_comp1_in);
        EDGE_LEARNING_TEST_EQUAL(j_comp1_in, j_comp1_out);

        Json j_comp2_in;

        Json j_comp2_out(jdict_in);
        j_comp2_out["list"] = Json(jlist_in);
        j_comp2_out["dict"] = Json(jdict_in);
        ofs = std::ofstream("tmp.json", std::ofstream::trunc);
        EDGE_LEARNING_TEST_TRY(ofs << j_comp2_out);
        ofs.close();

        ifs = std::ifstream("tmp.json");
        EDGE_LEARNING_TEST_TRY(ifs >> j_comp2_in);
        ifs.close();
        EDGE_LEARNING_TEST_PRINT(j_comp2_in);
        EDGE_LEARNING_TEST_EQUAL(j_comp2_in, j_comp2_out);

        fs::remove(fs::path("tmp.json"));
    }

};

int main() {
    TestJson().test();
    return EDGE_LEARNING_TEST_FAILURES;
}



