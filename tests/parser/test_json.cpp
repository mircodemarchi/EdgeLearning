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
        EDGE_LEARNING_TEST_CALL(test_json_leaf());
        EDGE_LEARNING_TEST_CALL(test_json_list());
        EDGE_LEARNING_TEST_CALL(test_json_dict());
        EDGE_LEARNING_TEST_CALL(test_json_item());
        EDGE_LEARNING_TEST_CALL(test_json());
        EDGE_LEARNING_TEST_CALL(test_stream());
    }
private:
    const std::string DATA_TRAINING_FN = "execution-time.csv";
    const std::filesystem::path data_training_fp =
        std::filesystem::path(__FILE__).parent_path()
        / ".." / ".." / "data" / DATA_TRAINING_FN;

    void test_json_leaf() {
        JsonLeaf jl = JsonLeaf("10");
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(), JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(jl.value(), "10");
        EDGE_LEARNING_TEST_EQUAL(jl.type(), Type::INT);
        EDGE_LEARNING_TEST_EQUAL(jl.as<int>(), 10);
        EDGE_LEARNING_TEST_EQUAL(jl.as<float>(), 10.0);
        EDGE_LEARNING_TEST_EQUAL(jl.as<bool>(), false);
        jl = JsonLeaf("1.0");
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(), JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(jl.value(), "1.0");
        EDGE_LEARNING_TEST_EQUAL(jl.type(), Type::FLOAT);
        EDGE_LEARNING_TEST_EQUAL(jl.as<int>(), 1);
        EDGE_LEARNING_TEST_EQUAL(jl.as<float>(), 1.0);
        EDGE_LEARNING_TEST_EQUAL(jl.as<bool>(), false);
        jl = JsonLeaf("true");
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(), JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(jl.value(), "true");
        EDGE_LEARNING_TEST_EQUAL(jl.type(), Type::BOOL);
        EDGE_LEARNING_TEST_EQUAL(jl.as<int>(), 0);
        EDGE_LEARNING_TEST_EQUAL(jl.as<float>(), 0.0);
        EDGE_LEARNING_TEST_EQUAL(jl.as<bool>(), true);

        JsonLeaf i = JsonLeaf(int(10));
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(), JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(i.value(), "10");
        EDGE_LEARNING_TEST_EQUAL(i.type(), Type::INT);
        EDGE_LEARNING_TEST_EQUAL(i.as<int>(), 10);
        EDGE_LEARNING_TEST_EQUAL(i.as<float>(), 10.0);
        EDGE_LEARNING_TEST_EQUAL(i.as<bool>(), false);
        JsonLeaf f = JsonLeaf(float(1.0));
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(), JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_PRINT(f.value());
        EDGE_LEARNING_TEST_EQUAL(f.value(), "1.000000");
        EDGE_LEARNING_TEST_EQUAL(f.type(), Type::FLOAT);
        EDGE_LEARNING_TEST_EQUAL(f.as<int>(), 1);
        EDGE_LEARNING_TEST_EQUAL(f.as<float>(), 1.0);
        EDGE_LEARNING_TEST_EQUAL(f.as<bool>(), false);
        JsonLeaf b = JsonLeaf(true);
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(), JsonObject::JsonType::LEAF);
        EDGE_LEARNING_TEST_EQUAL(b.value(), "true");
        EDGE_LEARNING_TEST_EQUAL(b.type(), Type::BOOL);
        EDGE_LEARNING_TEST_EQUAL(b.as<int>(), 0);
        EDGE_LEARNING_TEST_EQUAL(b.as<float>(), 0.0);
        EDGE_LEARNING_TEST_EQUAL(b.as<bool>(), true);

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

        JsonList jl = JsonList({10, 10.0, "string", true});
        EDGE_LEARNING_TEST_EQUAL(jl.json_type(),
                                 JsonObject::JsonType::LIST);
        EDGE_LEARNING_TEST_PRINT(jl);
        EDGE_LEARNING_TEST_PRINT(jl[0]);
        EDGE_LEARNING_TEST_ASSERT(!jl.empty());
        EDGE_LEARNING_TEST_EQUAL(jl.size(), 4);
        EDGE_LEARNING_TEST_FAIL(jl[10]);
        EDGE_LEARNING_TEST_THROWS(jl[10], std::runtime_error);

        JsonList jl1 = JsonList({10, 10.0, "string", true});
        JsonList jl2 = JsonList({10, 10.0, "string", true, "1_more"});
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

        JsonDict jd = JsonDict({{"a", 0}, {"b", true}});
        EDGE_LEARNING_TEST_EQUAL(jd.json_type(),
                                 JsonObject::JsonType::DICT);
        EDGE_LEARNING_TEST_PRINT(jd);
        EDGE_LEARNING_TEST_PRINT(jd["a"]);
        EDGE_LEARNING_TEST_ASSERT(!jd.empty());
        EDGE_LEARNING_TEST_EQUAL(jd.size(), 2);
        EDGE_LEARNING_TEST_FAIL(jd.at("key"));
        EDGE_LEARNING_TEST_THROWS(jd.at("key"), std::runtime_error);

        JsonDict jd1 = JsonDict({{"a", 0}, {"b", true}});
        JsonDict jd2 = JsonDict({{"a", 0}, {"b", true}, {"c", "1_more"}});
        EDGE_LEARNING_TEST_ASSERT(jd1 != jd2);
        EDGE_LEARNING_TEST_TRY(jd1["c"] = "1_more");
        EDGE_LEARNING_TEST_ASSERT(jd1 == jd2);
        EDGE_LEARNING_TEST_ASSERT(jd1["c"] == jd2["c"]);
    }

    void test_json_item() {
        JsonItem ji_empty;
        EDGE_LEARNING_TEST_PRINT(ji_empty);
        EDGE_LEARNING_TEST_EQUAL(ji_empty.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_FAIL(ji_empty.value().json_type());
        EDGE_LEARNING_TEST_THROWS(ji_empty.value().json_type(),
                                  std::runtime_error);

        JsonItem ji(10);
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(ji == 10);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        ji = JsonItem(1.0);
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(ji == 1.0);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        ji = JsonItem(false);
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(ji == false);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        ji = JsonItem("stringtest");
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(ji == "stringtest");
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        ji = JsonItem(std::string("stringtest"));
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(ji == std::string("stringtest"));
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        ji = 10;
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(ji == 10);
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LEAF);
        ji = JsonItem({10, "test1", false, "test1", 1.0});
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(
            ji == JsonItem({10, "test1", false, "test1", 1.0}));
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LIST);
        ji = {10, "test2", false, "test2", 1.0};
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(
            ji == JsonItem({10, "test2", false, "test2", 1.0}));
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::LIST);
        ji = JsonItem({{"test1", 10}, {"test1", false}});
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(
            ji == JsonItem({{"test1", 10}, {"test1", false}}));
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::DICT);
        ji = {{"test2", 10}, {"test2", false}};
        EDGE_LEARNING_TEST_PRINT(ji);
        EDGE_LEARNING_TEST_ASSERT(
            ji == JsonItem({{"test2", 10}, {"test2", false}}));
        EDGE_LEARNING_TEST_EQUAL(ji.json_type(),
                                 JsonObject::JsonType::NONE);
        EDGE_LEARNING_TEST_EQUAL(ji.value().json_type(),
                                 JsonObject::JsonType::DICT);
    }

    void test_json() {
        Json j_list;
        EDGE_LEARNING_TEST_TRY(j_list.append(10));
        EDGE_LEARNING_TEST_TRY(j_list.append(true));
        EDGE_LEARNING_TEST_TRY(j_list.append("test"));
        EDGE_LEARNING_TEST_TRY(j_list.append(1.0));
        EDGE_LEARNING_TEST_ASSERT(j_list == Json({10, true, "test", 1.0}));
        EDGE_LEARNING_TEST_TRY(j_list.append(Json({{"test", 1}, {"b", 10}})));
        EDGE_LEARNING_TEST_PRINT(j_list);
        EDGE_LEARNING_TEST_FAIL(j_list["fail_test"] = -1);
        EDGE_LEARNING_TEST_THROWS(j_list["fail_test"] = -1,
                                  std::runtime_error);

        Json j_dict;
        EDGE_LEARNING_TEST_TRY(j_dict["test"] = 1);
        EDGE_LEARNING_TEST_TRY(j_dict["b"] = 10);
        EDGE_LEARNING_TEST_ASSERT(j_dict == Json({{"test", 1}, {"b", 10}}));
        EDGE_LEARNING_TEST_TRY(j_dict["list"] = Json({10, true, "test", 1.0}));
        EDGE_LEARNING_TEST_PRINT(j_dict);
        EDGE_LEARNING_TEST_FAIL(j_dict.append({"fail_test"}));
        EDGE_LEARNING_TEST_THROWS(j_dict.append({"fail_test"}),
                                  std::runtime_error);

        Json j_comb1 = j_list;
        EDGE_LEARNING_TEST_TRY(j_comb1.append(j_dict));
        EDGE_LEARNING_TEST_PRINT(j_comb1);
        EDGE_LEARNING_TEST_TRY(j_comb1.append(j_list));
        EDGE_LEARNING_TEST_PRINT(j_comb1);
        EDGE_LEARNING_TEST_TRY(j_comb1.append(j_comb1));
        EDGE_LEARNING_TEST_PRINT(j_comb1);

        Json j_comb2 = j_dict;
        EDGE_LEARNING_TEST_TRY(j_comb2["j_list"] = j_list);
        EDGE_LEARNING_TEST_PRINT(j_comb2);
        EDGE_LEARNING_TEST_TRY(j_comb2["j_dict"] = j_dict);
        EDGE_LEARNING_TEST_PRINT(j_comb2);
        EDGE_LEARNING_TEST_TRY(j_comb2["j_comb"] = j_comb2);
        EDGE_LEARNING_TEST_PRINT(j_comb2);
    }

    void test_stream()
    {
        JsonLeaf jl_in;

        JsonLeaf jl_out = 10;
        std::ofstream ofs("tmp.json", std::ofstream::trunc);
        EDGE_LEARNING_TEST_TRY(ofs << jl_out);
        ofs.close();

        std::ifstream ifs("tmp.json");
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

        JsonList jlist_out({"string", 10, 1.0, true});
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

        JsonDict jdict_out({{"a", 10}, {"b", "string"}, {"c", true}});
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



