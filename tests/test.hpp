/***************************************************************************
 *            test.hpp
 *
 *  Copyright  2007  Alberto Casagrande, Pieter Collins, Ivan S. Zapreev
 *  casagrande@dimi.uniud.it, pieter.collins@cwi.nl, ivan.zapreev@gmail.com
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

/*!\file test.hpp
 * \brief Macros for test suite.
 */

#ifndef EDGE_LEARNING_TEST_HPP
#define EDGE_LEARNING_TEST_HPP

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <exception>
#include <complex>

int EDGE_LEARNING_TEST_FAILURES=0;
int EDGE_LEARNING_TEST_SKIPPED=0;
std::string EDGE_LEARNING_CURRENT_TESTING_CLASS = "???";

typedef bool Bool;

// This needs to be a function since we do not want to evaluate the result twice,
// and can't store it in a variable since we don't know it's type.
template<class R, class ER>
bool
edgelearning_check(std::ostream& os, const R& r, const ER& er) {
    os << r << std::flush; return (r==er);
}


//This is the variable that stores counter for the number of test cases
//The value is used and updated in the next two macro definitions
int test_case_counter = 0;

#if defined(linux) || defined(__linux) || defined(__linux__)
#define EDGELEARNING_PRETTY_FUNCTION __PRETTY_FUNCTION__
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#define EDGELEARNING_PRETTY_FUNCTION __FUNCSIG__
#elif defined(darwin) || defined(__darwin) || defined(__darwin__) || defined(__APPLE__)
#define EDGELEARNING_PRETTY_FUNCTION __PRETTY_FUNCTION__
#else
#define EDGELEARNING_PRETTY_FUNCTION ""
#endif

/*! \brief Tests a class function */
#define EDGE_LEARNING_TEST_CLASS(classname,testclassconstruct)                       \
    { \
        std::cout << "****************************************\n"       \
                  << "TESTING CLASS " << #classname << "\n"                    \
                  << "****************************************\n" << std::endl; \
        EDGE_LEARNING_CURRENT_TESTING_CLASS=#classname; \
        testclassconstruct.test(); \
    } \

/*! \brief Print the title for the test case */
#define EDGE_LEARNING_PRINT_TEST_CASE_TITLE( pTitle )                         \
    {                                                                   \
        std::cout << std::endl << "*** " << ++test_case_counter << ": "<< pTitle << " ***" << std::endl; \
        std::cout.flush();                                                   \
    }                                                                   \

/*! \brief Print the comment for the test */
#define EDGE_LEARNING_PRINT_TEST_COMMENT( pComment )                          \
    {                                                                   \
        std::cout << "* COMMENT: " << pComment << "" << std::endl;                \
        std::cout.flush();                                                   \
    }                                                                   \


/*! \brief Provide a warning message */
#define EDGE_LEARNING_TEST_WARN( message )                                    \
    {                                                                   \
        std::cout << "WARNING: " << message << "" << std::endl;                \
        std::cerr << "WARNING: " << message << "" << std::endl;                \
    }                                                                   \


/*! \brief Notify the user about a possibly unintuitive feature */
#define EDGE_LEARNING_TEST_NOTIFY( message )                                    \
    {                                                                   \
        std::cout << "NOTIFICATION: " << message << "" << std::endl;                \
        std::cerr << "NOTIFICATION: " << message << "" << std::endl;                \
    }                                                                   \


/*! \brief Catches an exception and writes a diagnostic to standard output and standard error. */
#define EDGE_LEARNING_TEST_CATCH(message)                                     \
    catch(const std::exception& except) {                                    \
        ++EDGE_LEARNING_TEST_FAILURES;                                        \
        std::cout << "exception: \"" << except.what() << "\"\n" << std::endl; \
        std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": " << message << " throwed \"" << except.what() << "\"." << std::endl;     \
    }                                                                   \
    catch(...) {                                                        \
        ++EDGE_LEARNING_TEST_FAILURES;                                        \
        std::cout << "unknown exception\n" << std::endl;                \
        std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": " << message << " throwed an unknown exception." << std::endl;       \
    }                                                                   \


/*! \brief Calls a function */
#define EDGE_LEARNING_TEST_CALL(function)                                     \
    {                                                                   \
        std::cout << "****************************************\n"       \
                  << "CALLING " << #function << "\n"                    \
                  << "****************************************\n" << std::endl; \
        try {                                                           \
            function;                                                   \
        } catch(const std::exception& except) {                              \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "ERROR: exception '" << except.what() << "' in " << #function << ": "    \
                      << except.what() << std::endl;                         \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": calling " \
                      << #function << ": " << except.what() << std::endl; \
            std::cout << std::endl;                                     \
        }                                                               \
    }                                                                   \


/*! \brief Omits a test, with a warning message */
#define EDGE_LEARNING_TEST_SKIP(function)                                     \
    {                                                                   \
        std::cout << "****************************************\n"       \
                  << "SKIPPING " << #function << "\n"                   \
                  << "****************************************\n" << std::endl; \
        ++EDGE_LEARNING_TEST_SKIPPED;                                         \
        std::cout << std::endl;                                         \
    }                                                                   \


/*! \brief Executes \a statement, writing the statement to standard output. Does not check for any errors. */
#define EDGE_LEARNING_TEST_EXECUTE(statement)                                 \
    {                                                                   \
        std::cout << #statement << ": " << std::flush;                  \
        statement;                                                      \
        std::cout << " (ok)\n" << std::endl;                            \
    }                                                                   \


/*! \brief Tries to execute \a statement, writing the statement to standard output. Writes a diagnostic report to standard error if an exception is thrown. <br> <b>Important:</b> Use the EDGE_LEARNING_TEST_CONSTRUCT() macro if \a statement declares a variable and calls a constructor. */
#define EDGE_LEARNING_TEST_TRY(statement)                                     \
    {                                                                   \
        std::cout << #statement << ": " << std::flush;                  \
        try {                                                           \
            statement;                                                  \
            std::cout << " (ok)\n" << std::endl;                        \
        }                                                               \
            EDGE_LEARNING_TEST_CATCH("Statement `" << #statement << "'")      \
        }                                                               \


/*! \brief Writes the expression to the output. Does not catch errors. */
#define EDGE_LEARNING_TEST_PRINT(expression)                                  \
    {                                                                   \
        std::cout << #expression << " = " << std::flush;                \
        std::cout << (expression) << "\n" << std::endl;                 \
    }                                                                   \


/*! \brief Tries to evaluate \a expression, writing the expression and the result to standard ouput. Writes a diagnostic report to standard error if an exception is thrown. */
#define EDGE_LEARNING_TEST_EVALUATE(expression)                               \
    {                                                                   \
        std::cout << #expression << ": " << std::flush;                 \
        try {                                                           \
            std::cout << (expression) << "\n" << std::endl;             \
        }                                                               \
            EDGE_LEARNING_TEST_CATCH("Expression `" << #expression << "'")    \
        }                                                               \


/*! \brief Evaluates \a expression in a boolean context and checks if the result is \a true. */
#define EDGE_LEARNING_TEST_ASSERT(expression)                                 \
    {                                                                   \
        std::cout << #expression << ": " << std::flush;                 \
        auto result = (expression);                                     \
        if(result) {                                                    \
            std::cout << "true\n" << std::endl;                         \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: false" << std::endl;                 \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__ << ": Assertion `" << #expression << "' failed." << std::endl; \
        }                                                               \
    }                                                                   \


/*! \brief Evaluates \a expression and checks if the result is equal to \a expected. */
#define EDGE_LEARNING_TEST_CHECK_WARN(expression,expected)                         \
    {                                                                   \
        std::cout << #expression << ": " << std::flush; \
        Bool ok = edgelearning_check(std::cout,expression,expected);         \
        if(ok) {                                                        \
            std::cout << "\n" << std::endl;                             \
        } else {                                                        \
            std::cout << "\nWARNING: expected " << #expression << " = " << #expected << " == " << (expected) << " \n" << std::endl; \
            std::cerr << "WARNING: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": Check `" << #expression << "==" << #expected << "' failed; obtained " << (expression) << std::endl; \
        }                                                               \
    }                                                                   \


/*! \brief Evaluates \a expression and checks if the result is equal to \a expected. */
#define EDGE_LEARNING_TEST_CHECK(expression,expected)                         \
    {                                                                   \
        std::cout << #expression << ": " << std::flush; \
        Bool ok = edgelearning_check(std::cout,expression,expected);         \
        if(ok) {                                                        \
            std::cout << "\n" << std::endl;                             \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: expected " << #expression << " = " << #expected << " == " << (expected) << " \n" << std::endl; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": Check `" << #expression << "==" << #expected << "' failed; obtained " << (expression) << std::endl; \
        }                                                               \
    }                                                                   \


/*! \brief Evaluates \a expression1 and expression2 and checks if the results are equal. */
#define EDGE_LEARNING_TEST_SAME(expression1,expression2)                         \
    {                                                                   \
        std::cout << "same(" << #expression1 << "," << #expression2 << "): " << std::flush; \
        Bool ok = same((expression1), (expression2));                       \
        if(ok) {                                                        \
            std::cout << "true\n" << std::endl;                         \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: " << #expression1 << ":\n           " << (expression1) \
                      << "\n     : " << #expression2 << ":\n           " << (expression2) << std::endl; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": Identity `" << #expression1 << " === " << #expression2 << "' failed; " << #expression1 << "=" << (expression1) << "; " << #expression2 << "=" << (expression2) << std::endl; \
        }                                                               \
    }                                                                   \

/*! \brief Evaluates \a expression and checks if the result is the same as \a expected. */
#define EDGE_LEARNING_TEST_SAME_AS(expression,expected)                         \
    {                                                                   \
        std::cout << #expression << " == " << #expected << ": " << std::flush; \
        Bool ok = same((expression), (expected));                       \
        if(ok) {                                                        \
            std::cout << "true\n" << std::endl;                         \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: " << #expression << ":\n           " << (expression) << std::endl; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": Sameness of `" << #expression << " and " << #expected << "' failed;" << std::endl; \
            std::cerr << "  " << #expression << "=" << (expression) << std::endl; \
            std::cerr << "  " << #expected << "=" << (expected) << std::endl; \
        }                                                               \
    }                                                                   \


/*! \brief Evaluates \a expression1 and expression2 and checks if the results are equal. */
#define EDGE_LEARNING_TEST_EQUAL(expression1,expression2)                         \
    {                                                                   \
        std::cout << #expression1 << " == " << #expression2 << ": " << std::flush; \
        Bool ok = ((expression1) == (expression2));                       \
        if(ok) {                                                        \
            std::cout << "true\n" << std::endl;                         \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: " << #expression1 << ":\n           " << (expression1) \
                      << "\n     : " << #expression2 << ":\n           " << (expression2) << std::endl; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": Equality `" << #expression1 << " == " << #expression2 << "' failed; " << #expression1 << "=" << (expression1) << "; " << #expression2 << "=" << (expression2) << std::endl; \
        }                                                               \
    }                                                                   \

/*! \brief Evaluates \a expression1 and expression2 and checks if the results are not equal. */
#define EDGE_LEARNING_TEST_NOT_EQUAL(expression1,expression2)                 \
    {                                                                   \
        std::cout << #expression1 << " != " << #expression2 << ": " << std::flush; \
        Bool ok = ((expression1) == (expression2));               \
        if(ok) {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: " << #expression1 << ":\n           " << (expression1) \
                      << "\n     : " << #expression2 << ":\n           " << (expression2) << std::endl; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": Inequality `" << #expression1 << " != " << #expression2 << "' failed; " << #expression1 << "=" << (expression1) << "; " << #expression2 << "=" << (expression2) << std::endl; \
        } else {                                                        \
            std::cout << "true\n" << std::endl;                         \
        }                                                               \
    }                                                                   \

/*! \brief Evaluates \a expression and checks if the result is equal to \a expected. */
#define EDGE_LEARNING_TEST_EQUALS(expression,expected)                         \
    {                                                                   \
        std::cout << #expression << " == " << #expected << ": " << std::flush; \
        Bool ok = ((expression) == (expected));                       \
        if(ok) {                                                        \
            std::cout << "true\n" << std::endl;                         \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: " << #expression << ":\n           " << (expression) << std::endl; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": Equality `" << #expression << " == " << #expected << "' failed;" << std::endl; \
            std::cerr << "  " << #expression << "=" << (expression) << std::endl; \
            std::cerr << "  " << #expected << "=" << (expected) << std::endl; \
        }                                                               \
    }                                                                   \

/*! \brief Evaluates \a expression and checks if the result is within \a tolerance of \a expected. */
#define EDGE_LEARNING_TEST_WITHIN(expression,expected,tolerance)                         \
    {                                                                   \
        std::cout << #expression << " ~ " << #expected << ": " << std::flush; \
        auto error=std::abs((expression)-(expected)); \
        Bool ok = (error <= (tolerance));                       \
        if(ok) {                                                        \
            std::cout << "true\n" << std::endl;                         \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: " << #expression << ":\n           " << (expression) \
                      << "\n     : " << #expected << ":\n           " << (expected) \
                      << "\n     : error: " << (error) \
                      << "\n     : tolerance " << (tolerance) << std::endl; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": ApproximateTag equality `" << #expression << " ~ " << #expected << "' failed; " << #expression << "=" << (expression) << "; " << #expected << "=" << (expected)<< "; error=" << (error) << "; tolerance=" << (tolerance) << std::endl; \
        }                                                               \
    }                                                                   \
                                                                   \


/*! \brief Evaluates \a expression and checks if the result is less than \a expected. */
#define EDGE_LEARNING_TEST_LESS(expression,expected)                         \
    {                                                                   \
        std::cout << #expression << " < " << #expected << ": " << std::flush; \
        Bool ok = ((expression) < (expected));                       \
        if(ok) {                                                        \
            std::cout << "true\n" << std::endl;                         \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: " << #expression << ":\n           " << (expression) << std::endl; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": Equality `" << #expression << " < " << #expected << "' failed; " << #expression << "=" << (expression) << std::endl; \
        }                                                               \
    }                                                                   \


/*! \brief Evaluates \a predicate(\a argument) and checks if the result is \tt true. */
#define EDGE_LEARNING_TEST_UNARY_PREDICATE(predicate,argument)    \
    {                                                                   \
        std::cout << #predicate << "(" << #argument << ") with " << #argument << "=" << (argument) << ": " << std::flush; \
        Bool ok = (predicate((argument)));                  \
        if(ok) {                                                        \
            std::cout << "true\n" << std::endl;                         \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: false" << std::endl;                 \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": Predicate `" << #predicate << "(" << #argument << ")' with " << #argument << "=" << (argument) << " is false." << std::endl; \
        }                                                               \
    }


/*! \brief Evaluates \a predicate(argument1,argument2) and checks if the result is \tt true. */
#define EDGE_LEARNING_TEST_BINARY_PREDICATE(predicate,argument1,argument2)    \
    {                                                                   \
        std::cout << #predicate << "(" << (#argument1) << "," << (#argument2) << ") with " << #argument1 << "=" << (argument1) << ", " << #argument2 << "=" << (argument2) << ": " << std::flush; \
        Bool ok = (predicate((argument1),(argument2)));                  \
        if(ok) {                                                        \
            std::cout << "true\n" << std::endl;                         \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: false" << std::endl;                 \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": Predicate `" << #predicate << "(" << #argument1 << "," << #argument2 << ")' with\n  " << #argument1 << "=" << (argument1) << ";\n  " << #argument2 << "=" << (argument2) << " is false." << std::endl; \
        }                                                               \
    }


/*! \brief Evaluates \a expression and checks if the result compares correctly with \a expected. */
#define EDGE_LEARNING_TEST_COMPARE(expression,comparison,expected)           \
    {                                                                   \
        std::cout << #expression << ": " << (expression) << std::flush; \
        Bool ok = ((expression) comparison (expected));               \
        if(ok) {                                                        \
            std::cout << " " << #comparison << " " << (expected) << ": true\n" << std::endl; \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: expected: " << #expression << #comparison << #expected << "=" << (expected) << std::endl; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": Comparison `" << #expression << #comparison << #expected << "' failed; " << #expression << "=" << (expression) << "; " << #expected << "=" << (expected) << std::endl; \
        }                                                               \
    }                                                                   \


/*! \brief Evaluates \a expression, converts to \a Type, and checks if the result compares correctly with \a expected. */
#define EDGE_LEARNING_TEST_RESULT_COMPARE(Type,expression,comparison,expected) \
    {                                                                   \
        Type result=(expression);                                       \
        std::cout << #expression << ": " << result << std::flush; \
        Bool ok = result comparison (expected);               \
        if(ok) {                                                        \
            std::cout << " " << #comparison << " " << (expected) << "\n" << std::endl; \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: expected: " << #expression << #comparison << #expected << std::endl; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": Comparison `" << #expression << #comparison << #expected << "' failed; " << #expression << "=" << result << "; " << #expected << "=" << (expected) << std::endl; \
        }                                                               \
    }                                                                   \


/*! \brief Declares an object \a variable of type \a Class (uses the default constructor). */
#define EDGE_LEARNING_TEST_DECLARE(Class,variable)                            \
    {                                                                   \
        std::cout << #Class << " " << #variable << ": " << std::flush;  \
        try {                                                           \
            Class variable;                                             \
            std::cout << #variable << "==" << variable << "\n" << std::endl; \
        }                                                               \
        EDGE_LEARNING_TEST_CATCH("Constructor `" << #Class << "" << #variable << "'") \
    }                                                                   \
    Class variable;                                                     \


/*! \brief Constructs object \a variable of type \a Class from \a expression. */
#define EDGE_LEARNING_TEST_CONSTRUCT(Class,variable,expression)               \
    std::cout << #Class << " " << #variable << "" << #expression << ": " << std::flush; \
    Class variable expression;                                          \
    std::cout << #variable << "==" << variable << "\n" << std::endl;    \

/*! \brief Constructs default object \a variable of type \a Class. */
#define EDGE_LEARNING_TEST_DEFAULT_CONSTRUCT(Class,variable)               \
    std::cout << #Class << " " << #variable << ": " << std::flush; \
    Class variable;                                                \
    std::cout << #variable << "==" << (variable) << "\n" << std::endl;    \


/*! \brief Constructs object \a variable of type \a Class from \a expression. */
#define EDGE_LEARNING_TEST_NAMED_CONSTRUCT(Class,variable,expression)               \
    {                                                                   \
        std::cout << #Class << " " << #variable << " = " << #Class << "::" << #expression << ": " << std::flush; \
        try {                                                           \
            Class variable = Class :: expression;                                  \
            std::cout << #variable << "==" << variable << "\n" << std::endl; \
        }                                                               \
        EDGE_LEARNING_TEST_CATCH("Named constructor `" << #variable << "=" << #Class << "::" << #expression << "'") \
    }                                                                   \
    Class variable = Class :: expression;                                          \


/*! \brief Construct object \a variable of type \a Class from \a expression using assignment syntax. */
#define EDGE_LEARNING_TEST_ASSIGN_CONSTRUCT(Class,variable, expression)       \
    {                                                                   \
        std::cout << #Class << " " << #variable << " = " << #expression << ": " << std::flush; \
        try {                                                           \
            Class variable = expression;                                \
            std::cout << #variable << "==" << variable << "\n" << std::endl;                 \
        }                                                               \
        EDGE_LEARNING_TEST_CATCH("Assignment `" << #variable << "=" << #expression << "'") \
    }                                                                   \
    Class variable = expression;                                        \

/*! \brief Assigns object \a variable from \a expression. */
#define EDGE_LEARNING_TEST_ASSIGN(variable, expression)                       \
    {                                                                   \
        std::cout << #variable << " = " << #expression << ": " << std::flush; \
        try {                                                           \
            variable=(expression);                                      \
            std::cout << variable << "\n" << std::endl;                 \
        }                                                               \
        EDGE_LEARNING_TEST_CATCH("Assignment `" << #variable << "=" << #expression << "'") \
            }                                                           \


/*! \brief Evaluates expression and expects an exception. */
#define EDGE_LEARNING_TEST_THROWS(statement,error)                             \
    {                                                                   \
        std::cout << #statement << ": " << std::flush;                  \
        try {                                                           \
            statement;                                                  \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: expected " << #error << "; no exception thrown\n"; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": expected " << #error << "; no exception thrown." << std::endl; \
        }                                                               \
        catch(const error& err) {                                         \
            std::cout << "caught " << #error << " as expected\n" << std::endl; \
        }                                                               \
        catch(const std::exception& except) {                                \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: caught exception " << except.what() << "; expected " << #error << "\n"; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": caught exception " << except.what() << "; expected " << #error << std::endl; \
        }                                                               \
    }                                                                   \


/*! \brief Evaluates expression and expects an exception. */
#define EDGE_LEARNING_TEST_FAIL(statement)                                    \
    {                                                                   \
        std::cout << #statement << ": " << std::flush;                  \
        try {                                                           \
            statement;                                                  \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "\nERROR: expected exception; none thrown\n";  \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << EDGELEARNING_PRETTY_FUNCTION << ": expected exception; no exception thrown." << std::endl; \
        }                                                               \
        catch(...) {                                                    \
            std::cout << "caught exception as expected\n" << std::endl; \
        }                                                               \
    }                                                                   \


/*! \brief Evaluates \a expression in a boolean context and checks if the result is \a true. */
/*! Use variadic macro argument to allow template parameters */
#define EDGE_LEARNING_TEST_STATIC_ASSERT(...)                          \
    {                                                                   \
        std::cout << #__VA_ARGS__ << ": " << std::flush;                 \
        bool result = ((__VA_ARGS__::value));                                   \
        if(result) {                                                    \
            std::cout << "true\n" << std::endl;                         \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "false\n" << std::endl;                 \
            std::cout << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__ << ": " << EDGE_LEARNING_CURRENT_TESTING_CLASS << ": Static assertion `" << #__VA_ARGS__ << "' failed." << "\n" << std::endl; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__ << ": " << EDGE_LEARNING_CURRENT_TESTING_CLASS << ": Static assertion `" << #__VA_ARGS__ << "' failed." << std::endl; \
        }                                                               \
    }                                                                   \


/*! \brief Evaluates \a expression in a boolean context and checks if the result is \a true. */
/*! Use variadic macro argument to allow template parameters */
#define EDGE_LEARNING_TEST_SAME_TYPE(...)                          \
    {                                                                   \
        std::cout << "IsSame<" << #__VA_ARGS__ << ">: " << std::flush;                 \
        bool result = ((IsSame<__VA_ARGS__>::value));                                   \
        if(result) {                                                    \
            std::cout << "true\n" << std::endl;                         \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "false\n" << std::endl;                 \
            std::cout << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__ << ": " << EDGE_LEARNING_CURRENT_TESTING_CLASS << ": Static assertion `IsSame<" << #__VA_ARGS__ << ">' failed." << " First type is " << class_name<typename First<__VA_ARGS__>::Type>() << "\n" << std::endl; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__ << ": " << EDGE_LEARNING_CURRENT_TESTING_CLASS << ": Static assertion `IsSame<" << #__VA_ARGS__ << ">' failed." << " First type is " << class_name<typename First<__VA_ARGS__>::Type>() << std::endl; \
        }                                                               \
    }                                                                   \


/*! \brief Tests if two types are equivalent. */
/*! Use variadic macro argument to allow template parameters */
#define EDGE_LEARNING_TEST_EQUIVALENT_TYPE(...)                          \
    {                                                                   \
        std::cout << "IsEquivalent<" << #__VA_ARGS__ << ">: " << std::flush;                 \
        bool result = ((IsEquivalent<__VA_ARGS__>::value));                                   \
        if(result) {                                                    \
            std::cout << "true\n" << std::endl;                         \
        } else {                                                        \
            ++EDGE_LEARNING_TEST_FAILURES;                                    \
            std::cout << "false\n" << std::endl;                 \
            std::cout << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__ << ": " << EDGE_LEARNING_CURRENT_TESTING_CLASS << ": Static assertion `IsEquivalent<" << #__VA_ARGS__ << ">' failed." << " First type is " << class_name<typename First<__VA_ARGS__>::Type>() << "\n" << std::endl; \
            std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << ": " << __FUNCTION__ << ": " << EDGE_LEARNING_CURRENT_TESTING_CLASS << ": Static assertion `IsEquivalent<" << #__VA_ARGS__ << ">' failed." << " First type is " << class_name<typename First<__VA_ARGS__>::Type>() << std::endl; \
        }                                                               \
    }                                                                   \


/*! \brief Check the Iterator of the GridTreeSubpaving by iterating through all it's values and
 * comparing them with the valus in the vector \a expected_result, the total number of iterated
 * elements should coincide with the value of \a expected_number_elements
 */
#define EDGE_LEARNING_TEST_GRID_TREE_SUBPAVING_ITERATOR( expected_result, theGridTreeSubpaving, expected_number_elements ) \
    {                                                                   \
        SizeType elements_count = 0;                                         \
        for (GridTreeSubpaving::ConstIterator it = theGridTreeSubpaving.begin(), end = theGridTreeSubpaving.end(); it != end; it++, elements_count++) { \
            if( elements_count < expected_number_elements ) {           \
                EDGE_LEARNING_PRINT_TEST_COMMENT("The next Iterator node is: "); \
                EDGE_LEARNING_TEST_COMPARE( (*expected_result[elements_count]), == , (*it) ); \
            }                                                           \
        }                                                               \
        EDGE_LEARNING_PRINT_TEST_COMMENT("Test that we iterated through the right number of nodes"); \
        EDGE_LEARNING_TEST_EQUAL( elements_count , expected_number_elements ); \
    }                                                                   \


/*! \brief clean std::vector, i.e. delete memory of it's non NULL elements and set them to NULL in the vector */
#define EDGE_LEARNING_CLEAN_TEST_VECTOR( vector ) \
    { \
        for(SizeType i = 0; i < vector.size(); i++ ) { \
            if( vector[i] != NULL ) { \
                delete vector[i]; vector[i] = NULL; \
            } \
        } \
    } \


#endif // EDGE_LEARNING_TEST_HPP

