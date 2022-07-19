/***************************************************************************
 *            data/path.hpp
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

/*! \file data/path.hpp
 *  \brief Wrapper of filesystem standard library.
 */

#ifndef EDGE_LEARNING_DATA_PATH_HPP

// Test macro for <filesystem>.
#if defined(__cpp_lib_filesystem)
#    define EDGE_LEARNING_DATA_PATH_HPP 0

// Test macro for <experimental/filesystem>.
#elif defined(__cpp_lib_experimental_filesystem) || !defined(__has_include)
#    define EDGE_LEARNING_DATA_PATH_HPP 1

// Check if the header <filesystem> exists.
#elif __has_include(<filesystem>)

#    ifdef _MSC_VER //< Check for Visual Studio. 
//       Check and include header that defines "_HAS_CXX17"
#        if __has_include(<yvals_core.h>)
#            include <yvals_core.h>
#            if defined(_HAS_CXX17) && _HAS_CXX17
#                define EDGE_LEARNING_DATA_PATH_HPP 0
#            endif // #if defined(_HAS_CXX17) && _HAS_CXX17
#        endif // #if __has_include(<yvals_core.h>)

#        ifndef EDGE_LEARNING_DATA_PATH_HPP
#            define EDGE_LEARNING_DATA_PATH_HPP 1
#        endif // #ifndef EDGE_LEARNING_DATA_PATH_HPP
#    else  // #ifdef _MSC_VER
#        define EDGE_LEARNING_DATA_PATH_HPP 0
#    endif // #ifdef _MSC_VER

// Check if the header <experimental/filesystem> exists.
#elif __has_include(<experimental/filesystem>)
#    define EDGE_LEARNING_DATA_PATH_HPP 1

#else
#    error Could not find header "<filesystem>" or "<experimental/filesystem>"
#endif // Tests and checks.

// Include.
#if EDGE_LEARNING_DATA_PATH_HPP
#    include <experimental/filesystem>
namespace std {
    namespace filesystem = experimental::filesystem;
}
#else
#    include <filesystem>
#endif // #if EDGE_LEARNING_DATA_PATH_HPP

#endif // EDGE_LEARNING_DATA_PATH_HPP