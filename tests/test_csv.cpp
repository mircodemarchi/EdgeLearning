/***************************************************************************
 *            tests/test_replaceme.cpp
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
#include "parser/csv.hpp"

#include <filesystem>
#include <vector>

using namespace std;
using namespace Ariadne;

class TestCSV {
public:
    void test() {
        ARIADNE_TEST_CALL(test_csv_field());
        ARIADNE_TEST_CALL(test_csv_row());
    }
private:
    void test_csv_field() {
        auto csv_field_int_t   = ParserType::AUTO;
        auto csv_field_str_t   = ParserType::AUTO;
        auto csv_field_float_t = ParserType::FLOAT;
        auto csv_field_bool_t  = ParserType::BOOL;

        auto csv_field_int     = CSVField{"123",   csv_field_int_t,   0};
        auto csv_field_str     = CSVField{"\"\"",  csv_field_str_t,   1};
        auto csv_field_float   = CSVField{"1",     csv_field_float_t, 2};
        auto csv_field_bool    = CSVField{"false", csv_field_bool_t,  3};

        int i; csv_field_int.as(&i);
        std::string s; csv_field_str.as(&s);
        ARIADNE_TEST_EQUAL(i, 123);
        ARIADNE_TEST_EQUAL(s, "\"\"");
        ARIADNE_TEST_WITHIN(csv_field_float.as<float>(), 1.0, 0.0000001);
        ARIADNE_TEST_EQUAL(csv_field_bool.as<bool>(), false);

        ARIADNE_TEST_EQUAL(csv_field_int.idx(),   0);
        ARIADNE_TEST_EQUAL(csv_field_str.idx(),   1);
        ARIADNE_TEST_EQUAL(csv_field_float.idx(), 2);
        ARIADNE_TEST_EQUAL(csv_field_bool.idx(),  3);

        ARIADNE_TEST_EQUAL(csv_field_int.type(), ParserType::INT);
        ARIADNE_TEST_EQUAL(csv_field_str.type(), ParserType::STRING);
    }

    void test_csv_row() {
        auto types = std::vector<ParserType> {ParserType::AUTO};
        auto csv_row = CSVRow("10,1.3,ariadnedl,true", 0, 4, types, ',');

        ARIADNE_TEST_PRINT(csv_row);
        ARIADNE_TEST_EQUAL(csv_row[0].as<int>(),    10);
        ARIADNE_TEST_WITHIN(csv_row[1].as<float>(), 1.3, 0.0000001);
        ARIADNE_TEST_EQUAL(csv_row[2].as<std::string>(), "ariadnedl");
        ARIADNE_TEST_EQUAL(csv_row[3].as<bool>(),   true);
        ARIADNE_TEST_FAIL(csv_row[4]);
        ARIADNE_TEST_EXECUTE(auto v = std::vector<float>(csv_row));
        ARIADNE_TEST_PRINT(std::vector<float>(csv_row).at(2));

        ARIADNE_TEST_EQUAL(csv_row.empty(), false);
        ARIADNE_TEST_EQUAL(csv_row.size(),  4);
        ARIADNE_TEST_EQUAL(csv_row.idx(),   0);
        auto types_groundtruth = std::vector<ParserType>{
            ParserType::INT, ParserType::FLOAT,
            ParserType::STRING, ParserType::BOOL};
        auto types_to_test = csv_row.types();
        for (size_t i = 0; i < csv_row.size(); ++i)
        {
            ARIADNE_TEST_EQUAL(types_groundtruth.at(i), types_to_test.at(i));
        }

        csv_row = CSVRow("10,1.3", 3, types, ',');
        ARIADNE_TEST_EQUAL(csv_row.size(), 2);
        ARIADNE_TEST_FAIL(csv_row[3]);
        ARIADNE_TEST_EXECUTE(auto v = std::vector<float>(csv_row));
        ARIADNE_TEST_EXECUTE(auto v = std::vector<int>(csv_row));
        ARIADNE_TEST_EXECUTE(auto v = std::vector<std::string>(csv_row));
        ARIADNE_TEST_EXECUTE(auto v = std::vector<CSVField>(csv_row));

        csv_row = CSVRow(types, ',');
        ARIADNE_TEST_EQUAL(csv_row.types().size(), 0);
        ARIADNE_TEST_EQUAL(csv_row.empty(), true);
        ARIADNE_TEST_FAIL(csv_row[0]);
    }

    static const std::string DATA_TRAINING_FN;
    static const std::filesystem::path data_training_fp;
};

const std::string TestCSV::DATA_TRAINING_FN = "execution-time.csv";
const std::filesystem::path TestCSV::data_training_fp = 
    std::filesystem::path(__FILE__).parent_path() 
        / ".." / "data" / TestCSV::DATA_TRAINING_FN;


int main() {
    TestCSV().test();
    return ARIADNE_TEST_FAILURES;
}



