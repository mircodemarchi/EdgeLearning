/***************************************************************************
 *            util.hpp
 *
 *  Copyright  2021  Mirco De Marchi
 *
 ****************************************************************************/

/*
 * This file is part of EdgeLearning, under the MIT license.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef EDGELEARNING_UTIL_HPP
#define EDGELEARNING_UTIL_HPP


#include <chrono>
#include <string>
#include <sstream>

/**
 * \brief Time type utility to calculate and print elapsed execution time.
 */
class Time {
public:
    Time()
        : _start()
        , _end()
    {

    }

    ~Time() = default;

    void start()
    {
        _start = std::chrono::high_resolution_clock::now().time_since_epoch();
    }

    void stop()
    {
        _end = std::chrono::high_resolution_clock::now().time_since_epoch();
    }

    double elapsed()
    {
        std::chrono::duration<double, std::nano> elapsed = _end - _start;
        return elapsed.count();
    }

    operator std::string()
    {
        auto cnt = elapsed();
        std::stringstream ss;
        if (cnt < 1000)
            ss << cnt << " ns";
        else if (cnt < 1000000)
            ss << static_cast<double>(cnt) / 1000 << " us";
        else if (cnt < 1000000000)
            ss << static_cast<double>(cnt) / 1000000 << " ms";
        else if (cnt < 1000000000000)
            ss << static_cast<double>(cnt) / 1000000000 << " sec";
        else
            ss << static_cast<double>(cnt) / 60000000000 << " min";
        return ss.str();
    }

private:
    std::chrono::nanoseconds _start;
    std::chrono::nanoseconds _end;
};

#endif //EDGELEARNING_UTIL_HPP
