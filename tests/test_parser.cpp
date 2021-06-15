/***************************************************************************
 *            tests/test_parser.cpp
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

#include "test.hpp"
#include "parser/parser.hpp"

#include <vector>
#include <map>
#include <string>


using namespace std;
using namespace Ariadne;

class TestParser {
public:
    void test() {
        ARIADNE_TEST_CALL(test_parse());
        ARIADNE_TEST_CALL(test_is());
        ARIADNE_TEST_CALL(test_convert());
    }

private:
    void test_parse() {
        auto parser = Parser();
        auto test_vec = std::map<std::string, ParserType>{
            { "1.2",           ParserType::FLOAT  },
            { "+0.0",          ParserType::FLOAT  },
            { "-0.0",          ParserType::FLOAT  },
            { "+1e-10",        ParserType::FLOAT  },
            { "true",          ParserType::BOOL   },
            { "1",             ParserType::INT    },
            { "-1",            ParserType::INT    },
            { "+0",            ParserType::INT    },
            { "-0",            ParserType::INT    },
            { "\"string\"",    ParserType::STRING },
            { "123ariadne456", ParserType::STRING },
            { std::string{},   ParserType::NONE   },
        };

        auto keys = std::vector<std::string>{};
        auto values = std::vector<ParserType>{};
        for (const auto& [key, value]: test_vec)
        {
            keys.push_back(key);
            values.push_back(value);
        }

        auto parsed_values = parser(keys);
        ARIADNE_TEST_ASSERT(parsed_values == values);
        ARIADNE_TEST_PRINT(parsed_values);

        auto values_cpy = std::vector<ParserType>(values);
        values_cpy.push_back(ParserType::NONE);
        ARIADNE_TEST_ASSERT(parsed_values != values_cpy);
    }

    void test_is() {
        ARIADNE_TEST_ASSERT(Parser::is_float("-0.3"));
        ARIADNE_TEST_ASSERT(Parser::is_float(".245"));
        ARIADNE_TEST_ASSERT(Parser::is_bool("false"));
        ARIADNE_TEST_ASSERT(Parser::is_int("1234"));
        ARIADNE_TEST_ASSERT(Parser::is_string("ariadne123dl"));
    }

    void test_convert() {
        float f;
        ARIADNE_TEST_ASSERT(convert("1.2", &f));
        ARIADNE_TEST_WITHIN(f, 1.2, 0.0000001);
        bool b;
        ARIADNE_TEST_ASSERT(convert("true", &b));
        ARIADNE_TEST_EQUAL(b, true);
        int i;
        ARIADNE_TEST_ASSERT(convert("1", &i));
        ARIADNE_TEST_EQUAL(i, 1);
        std::string s;
        ARIADNE_TEST_ASSERT(convert("1", &s));
        ARIADNE_TEST_EQUAL(s, "1");

        ARIADNE_TEST_ASSERT(!convert("1string", &i));
    }
};

int main() {
    TestParser().test();
    return ARIADNE_TEST_FAILURES;
}



