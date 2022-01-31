/***************************************************************************
 *            tests/test_csv.cpp
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
#include "parser/csv.hpp"

#include <filesystem>
#include <vector>
#include <stdexcept>

using namespace std;
using namespace EdgeLearning;

class TestCSV {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_csv_field());
        EDGE_LEARNING_TEST_CALL(test_csv_row());
        EDGE_LEARNING_TEST_CALL(test_csv());
        EDGE_LEARNING_TEST_CALL(test_csv_iterator(10));
    }
private:
    const std::string DATA_TRAINING_FN = "execution-time.csv";
    const std::filesystem::path data_training_fp = 
        std::filesystem::path(__FILE__).parent_path() 
            / ".." / ".." / "data" / DATA_TRAINING_FN;

    void test_csv_field() {
        auto csv_field_int_t   = Type::AUTO;
        auto csv_field_str_t   = Type::AUTO;
        auto csv_field_float_t = Type::FLOAT;
        auto csv_field_bool_t  = Type::BOOL;

        auto csv_field_int     = CSVField{"123",   csv_field_int_t,   0};
        auto csv_field_str     = CSVField{"\"\"",  csv_field_str_t,   1};
        auto csv_field_float   = CSVField{"1",     csv_field_float_t, 2};
        auto csv_field_bool    = CSVField{"false", csv_field_bool_t,  3};

        int i; csv_field_int.as(&i);
        std::string s; csv_field_str.as(&s);
        EDGE_LEARNING_TEST_EQUAL(i, 123);
        EDGE_LEARNING_TEST_EQUAL(s, "\"\"");
        EDGE_LEARNING_TEST_WITHIN(csv_field_float.as<float>(), 1.0, 0.0000001);
        EDGE_LEARNING_TEST_EQUAL(csv_field_bool.as<bool>(), false);

        EDGE_LEARNING_TEST_EQUAL(csv_field_int.idx(),   0);
        EDGE_LEARNING_TEST_EQUAL(csv_field_str.idx(),   1);
        EDGE_LEARNING_TEST_EQUAL(csv_field_float.idx(), 2);
        EDGE_LEARNING_TEST_EQUAL(csv_field_bool.idx(),  3);

        EDGE_LEARNING_TEST_EQUAL(csv_field_int.type(), Type::INT);
        EDGE_LEARNING_TEST_EQUAL(csv_field_str.type(), Type::STRING);

        auto csv_field_cpy = csv_field_int;
        EDGE_LEARNING_TEST_EQUAL(csv_field_cpy.idx(), csv_field_int.idx());
        EDGE_LEARNING_TEST_EQUAL(csv_field_cpy.type(), csv_field_int.type());
    }

    void test_csv_row() {
        auto types = std::vector<Type>{Type::AUTO};
        auto csv_row = CSVRow("10,1.3,edge_learning,true", 0, 4, types, ',');

        EDGE_LEARNING_TEST_PRINT(csv_row);
        EDGE_LEARNING_TEST_EQUAL(csv_row[0].as<int>(),    10);
        EDGE_LEARNING_TEST_WITHIN(csv_row[1].as<float>(), 1.3, 0.0000001);
        EDGE_LEARNING_TEST_EQUAL(csv_row[2].as<std::string>(), "edge_learning");
        EDGE_LEARNING_TEST_EQUAL(csv_row[3].as<bool>(),   true);
        EDGE_LEARNING_TEST_FAIL(csv_row[4]);
        EDGE_LEARNING_TEST_EXECUTE(auto v = std::vector<float>(csv_row));
        EDGE_LEARNING_TEST_PRINT(std::vector<float>(csv_row).at(2));
        EDGE_LEARNING_TEST_WITHIN(std::vector<float>(csv_row).at(2), 
            csv_row[2].as<float>(), 0.0000001);
        EDGE_LEARNING_TEST_EQUAL(
            csv_row.to_vec<float>().size(), csv_row.size());
        EDGE_LEARNING_TEST_EQUAL(std::string(csv_row), csv_row.line());

        auto csv_row_compare = CSVRow(
            "10,1.3,edge_learning,true", 10, 4, types, ',');
        EDGE_LEARNING_TEST_EQUAL(csv_row, csv_row_compare);
        auto csv_row_compare_not_equal = CSVRow(
            "10,1.3,edge_learning,true,notequal", 0, 4, types, ',');
        EDGE_LEARNING_TEST_NOT_EQUAL(csv_row, csv_row_compare_not_equal);

        EDGE_LEARNING_TEST_EQUAL(csv_row.empty(), false);
        EDGE_LEARNING_TEST_EQUAL(csv_row.size(),  4);
        EDGE_LEARNING_TEST_EQUAL(csv_row.idx(),   0);
        auto types_groundtruth = std::vector<Type>{
            Type::INT, Type::FLOAT,
            Type::STRING, Type::BOOL};
        auto types_to_test = csv_row.types();
        EDGE_LEARNING_TEST_EQUAL(types_groundtruth, types_to_test);

        csv_row = CSVRow("10,1.3", 3, types, ',');
        EDGE_LEARNING_TEST_EQUAL(csv_row.size(), 2);
        EDGE_LEARNING_TEST_THROWS(csv_row[3], std::runtime_error);
        EDGE_LEARNING_TEST_EXECUTE(auto v = std::vector<float>(csv_row));
        EDGE_LEARNING_TEST_EXECUTE(auto v = std::vector<int>(csv_row));
        EDGE_LEARNING_TEST_EXECUTE(auto v = std::vector<std::string>(csv_row));
        EDGE_LEARNING_TEST_EXECUTE(auto v = std::vector<CSVField>(csv_row));

        csv_row = CSVRow(types, ',');
        EDGE_LEARNING_TEST_EQUAL(csv_row.types().size(), 0);
        EDGE_LEARNING_TEST_EQUAL(csv_row.empty(), true);
        EDGE_LEARNING_TEST_FAIL(csv_row[0]);

        auto csv_row_cpy = CSVRow{csv_row};
        EDGE_LEARNING_TEST_EQUAL(csv_row_cpy.idx(), csv_row.idx());
        EDGE_LEARNING_TEST_EQUAL(csv_row_cpy.size(), csv_row.size());
        EDGE_LEARNING_TEST_EQUAL(csv_row_cpy.types(), csv_row.types());

        csv_row = CSVRow("10,1.3,edge_learning,true", 1, 5, types, ',');
        EDGE_LEARNING_TEST_THROWS(csv_row[4], std::runtime_error);
    }

    void test_csv() {
        auto csv = CSV(data_training_fp.string());
        auto types_groundtruth = std::vector<Type>{6, Type::INT};

        EDGE_LEARNING_TEST_EQUAL(csv.cols_size(), 6);
        EDGE_LEARNING_TEST_EQUAL(csv.rows_size(), 3201);
        EDGE_LEARNING_TEST_EQUAL(csv.types(), types_groundtruth);

        EDGE_LEARNING_TEST_PRINT(csv.header());
        EDGE_LEARNING_TEST_ASSERT(!csv.header().empty());
        EDGE_LEARNING_TEST_EQUAL(csv.header().size(), csv.cols_size());
        EDGE_LEARNING_TEST_EQUAL(csv.header().types(), types_groundtruth);
        EDGE_LEARNING_TEST_EQUAL(csv.header().idx(), 0);
        EDGE_LEARNING_TEST_EQUAL(csv.header(), csv[0]);
        EDGE_LEARNING_TEST_EQUAL(csv.header(), csv[0]);
        EDGE_LEARNING_TEST_NOT_EQUAL(csv.header(), csv[1]);
        EDGE_LEARNING_TEST_EQUAL(csv[1], csv[1]);
        EDGE_LEARNING_TEST_NOT_EQUAL(csv[1], csv[2]);
    
        EDGE_LEARNING_TEST_PRINT(csv[1]);
        EDGE_LEARNING_TEST_ASSERT(!csv[1].empty());
        EDGE_LEARNING_TEST_EQUAL(csv[1].size(), csv.cols_size());
        EDGE_LEARNING_TEST_EQUAL(csv[1].idx(), 1);
        EDGE_LEARNING_TEST_EQUAL(types_groundtruth, csv[1].types());

        EDGE_LEARNING_TEST_PRINT(csv[2]);
        EDGE_LEARNING_TEST_ASSERT(!csv[2].empty());
        EDGE_LEARNING_TEST_EQUAL(csv[2].size(), csv.cols_size());
        EDGE_LEARNING_TEST_EQUAL(csv[2].idx(), 2);
        EDGE_LEARNING_TEST_EQUAL(types_groundtruth, csv[2].types());

        EDGE_LEARNING_TEST_THROWS(CSV{""}, std::runtime_error);

        csv = CSV(data_training_fp.string(), std::vector<Type>{});
        EDGE_LEARNING_TEST_EQUAL(csv.types(), types_groundtruth);

        csv = CSV(data_training_fp.string(), 
            std::vector<Type>{Type::INT});
        EDGE_LEARNING_TEST_EQUAL(csv.types(), types_groundtruth);

        csv = CSV(data_training_fp.string(), 
            std::vector<Type>{6, Type::FLOAT});
        EDGE_LEARNING_TEST_NOT_EQUAL(csv.types(), types_groundtruth);

        EDGE_LEARNING_TEST_EXECUTE(
            auto v = std::vector<float>(csv));
        EDGE_LEARNING_TEST_EQUAL(csv.to_vec<float>().size(),
            csv.rows_size() * csv.cols_size());

        EDGE_LEARNING_TEST_EXECUTE(
            auto v = std::vector<std::vector<float>>(csv));
        EDGE_LEARNING_TEST_EQUAL(csv.to_mat<float>().size(), csv.rows_size());
        EDGE_LEARNING_TEST_EQUAL(
            csv.to_mat<float>()[0].size(), csv.cols_size());

        EDGE_LEARNING_TEST_EXECUTE(auto v = std::vector<std::string>(csv));
        EDGE_LEARNING_TEST_EQUAL(
            std::vector<std::string>(csv).size(), csv.rows_size());
        EDGE_LEARNING_TEST_EQUAL(
            std::vector<std::string>(csv)[0], std::string(csv[0]));

        EDGE_LEARNING_TEST_EXECUTE(auto v = std::vector<CSVRow>(csv));
        EDGE_LEARNING_TEST_EQUAL(std::vector<CSVRow>(csv).size(), 
            csv.rows_size());
        EDGE_LEARNING_TEST_EQUAL(std::vector<CSVRow>(csv)[1], csv[1]);
    }

    void test_csv_iterator(const std::size_t num_lines) {
        auto csv = CSV(data_training_fp.string());
        std::size_t i = 0;
        for (auto &row: csv)
        {
            if (i == num_lines)
            {
                break;
            }
            EDGE_LEARNING_TEST_PRINT(row);
            ++i;
        }

        auto iterator = csv.begin();
        EDGE_LEARNING_TEST_EQUAL(iterator->idx(), csv[1].idx());
        EDGE_LEARNING_TEST_EQUAL(iterator->line(), csv[1].line());
        EDGE_LEARNING_TEST_EQUAL((*iterator).line(), csv[1].line());

        auto iterator_cpy1 = CSVIterator{iterator++};
        ++iterator_cpy1;
        EDGE_LEARNING_TEST_EQUAL(iterator_cpy1->idx(), csv[2].idx());
        EDGE_LEARNING_TEST_EQUAL((*iterator_cpy1).line(), csv[2].line());
        EDGE_LEARNING_TEST_ASSERT(iterator == iterator_cpy1);

        auto iterator_cpy2 = CSVIterator{iterator_cpy1--};
        --iterator_cpy2;
        --iterator_cpy2;
        EDGE_LEARNING_TEST_EQUAL(iterator_cpy2->idx(), csv[0].idx());
        EDGE_LEARNING_TEST_EQUAL((*iterator_cpy2).line(), csv[0].line());
        EDGE_LEARNING_TEST_ASSERT(--iterator_cpy1 == iterator_cpy2);

        auto iterator_cpy3 = CSVIterator{iterator};
        EDGE_LEARNING_TEST_EQUAL(iterator_cpy3->idx(), csv[2].idx());
        EDGE_LEARNING_TEST_EQUAL((*iterator_cpy3).line(), csv[2].line());
        iterator_cpy3--;
        iterator_cpy3--;
        EDGE_LEARNING_TEST_EQUAL(iterator_cpy3->idx(), csv[0].idx());
        EDGE_LEARNING_TEST_EQUAL((*iterator_cpy3).line(), csv[0].line());
        EDGE_LEARNING_TEST_ASSERT(iterator_cpy1 == iterator_cpy3);
    }
};

int main() {
    TestCSV().test();
    return EDGE_LEARNING_TEST_FAILURES;
}



