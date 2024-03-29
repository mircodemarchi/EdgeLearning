/***************************************************************************
 *            parser/test_type_checker.cpp
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
#include "parser/type_checker.hpp"

#include <vector>
#include <map>
#include <string>


using namespace std;
using namespace EdgeLearning;

class TestTypeChecker {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_type());
        EDGE_LEARNING_TEST_CALL(test_parse());
        EDGE_LEARNING_TEST_CALL(test_is());
        EDGE_LEARNING_TEST_CALL(test_convert());
    }

private:
    void test_type() {
        std::vector<TypeChecker::Type> type_vec({
            TypeChecker::Type::NONE,
            TypeChecker::Type::AUTO,
            TypeChecker::Type::FLOAT,
            TypeChecker::Type::INT,
            TypeChecker::Type::BOOL,
            TypeChecker::Type::STRING,
            TypeChecker::Type::OBJECT,
            static_cast<TypeChecker::Type>(100)
        });
        EDGE_LEARNING_TEST_PRINT(type_vec);
    }


    void test_parse() {
        auto parser = TypeChecker();
        auto test_vec = std::map<std::string, TypeChecker::Type>{
            { "1.2",           TypeChecker::Type::FLOAT  },
            { "+0.0",          TypeChecker::Type::FLOAT  },
            { "-0.0",          TypeChecker::Type::FLOAT  },
            { "+1e-10",        TypeChecker::Type::FLOAT  },
            { "true",          TypeChecker::Type::BOOL   },
            { "1",             TypeChecker::Type::INT    },
            { "-1",            TypeChecker::Type::INT    },
            { "+0",            TypeChecker::Type::INT    },
            { "-0",            TypeChecker::Type::INT    },
            { "\"string\"",    TypeChecker::Type::STRING },
            { "123edgelearning456", TypeChecker::Type::STRING },
            { std::string{},   TypeChecker::Type::NONE   },
        };

        auto keys = std::vector<std::string>{};
        auto values = std::vector<TypeChecker::Type>{};
        for (const auto& [key, value]: test_vec)
        {
            keys.push_back(key);
            values.push_back(value);
        }

        auto parsed_values = parser(keys);
        EDGE_LEARNING_TEST_ASSERT(parsed_values == values);
        EDGE_LEARNING_TEST_PRINT(parsed_values);

        auto values_cpy = std::vector<TypeChecker::Type>(values);
        values_cpy.push_back(TypeChecker::Type::NONE);
        EDGE_LEARNING_TEST_ASSERT(parsed_values != values_cpy);

        EDGE_LEARNING_TEST_EQUAL(parser(1), "1");
        EDGE_LEARNING_TEST_EQUAL(parser.operator()<float>(1), "1.000000");
        EDGE_LEARNING_TEST_EQUAL(parser.operator()<bool>(1), "true");
    }

    void test_is() {
        EDGE_LEARNING_TEST_ASSERT(TypeChecker::is_float("-0.3"));
        EDGE_LEARNING_TEST_ASSERT(TypeChecker::is_float(".245"));
        EDGE_LEARNING_TEST_ASSERT(TypeChecker::is_bool("false"));
        EDGE_LEARNING_TEST_ASSERT(TypeChecker::is_int("1234"));
        EDGE_LEARNING_TEST_ASSERT(TypeChecker::is_string("edgelearning123dl"));
    }

    void test_convert() {
        float f;
        EDGE_LEARNING_TEST_ASSERT(convert("1.2", f));
        EDGE_LEARNING_TEST_WITHIN(f, 1.2, 0.0000001);
        bool b;
        EDGE_LEARNING_TEST_ASSERT(convert("true", b));
        EDGE_LEARNING_TEST_EQUAL(b, true);
        int i;
        EDGE_LEARNING_TEST_ASSERT(convert("1", i));
        EDGE_LEARNING_TEST_EQUAL(i, 1);
        std::string s;
        EDGE_LEARNING_TEST_ASSERT(convert("1", s));
        EDGE_LEARNING_TEST_EQUAL(s, "1");

        EDGE_LEARNING_TEST_ASSERT(!convert("1string", i));

        EDGE_LEARNING_TEST_EQUAL(convert<int>(1), "1");
        EDGE_LEARNING_TEST_EQUAL(convert<float>(1), "1.000000");
        EDGE_LEARNING_TEST_EQUAL(convert<bool>(1), "true");
    }
};

int main() {
    TestTypeChecker
().test();
    return EDGE_LEARNING_TEST_FAILURES;
}



