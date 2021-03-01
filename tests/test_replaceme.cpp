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
#include "replaceme.hpp"

using namespace std;
using namespace Ariadne;

class TestReplaceMe {
  public:
    void test() {
        ARIADNE_TEST_CALL(test_something());
    }
  private:
    void test_something() {
        ARIADNE_TEST_PRINT(replaceme(2));
        ARIADNE_TEST_EQUAL(replaceme(3),9);
    }
};

int main() {
    TestReplaceMe().test();
    return ARIADNE_TEST_FAILURES;
}



