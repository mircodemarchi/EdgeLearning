/***************************************************************************
 *            tests/parser.cpp
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
#include "parser/parser.hpp"


using namespace std;
using namespace EdgeLearning;

class TestParser {
public:
    void test() {
        EDGE_LEARNING_TEST_CALL(test_parser());
    }

private:
    void test_parser() {
        EDGE_LEARNING_TEST_CALL(Parser());
    }
};

int main() {
    TestParser().test();
    return EDGE_LEARNING_TEST_FAILURES;
}



