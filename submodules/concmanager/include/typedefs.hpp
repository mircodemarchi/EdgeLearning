/***************************************************************************
 *            utility/typedefs.hpp
 *
 *  Copyright  2013-20  Pieter Collins
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

/*! \file utility/typedefs.hpp
 *  \brief
 */



#ifndef ARIADNE_TYPEDEFS_HPP
#define ARIADNE_TYPEDEFS_HPP

#include <cstdint>
#include <iosfwd>
#include <type_traits>
#include <memory>
#include <utility>
#include <mutex>
#include <thread>
#include <future>
#include <vector>

namespace ConcManager {

using uchar = unsigned char;
using uint = unsigned int;

//! \brief Internal name for standard output stream.
using OutputStream = std::ostream;
//! \brief Internal name for standard input stream.
using InputStream = std::istream;
//! Internal name for standard string stream.
using StringStream = std::stringstream;

//! Internal name for void type.
using Void = void;
//! Internal name for builtin boolean type.
using Bool = bool;
//! Internal name for builtin char type.
using Char = char;
//! Internal name for builtin byte type (8 bits).
using Byte = std::int8_t;
//! Internal name for builtin unsigned integers.
using Nat = uint;
//! Internal name for builtin integers.
using Int = int;
//! Internal name for builtin double-precision floats.
using Dbl = double;
//! Internal name for builtin double-precision floats.
using Double = double;


//! Internal name for builtin double-precision floats.
using StringType = std::string;
//! A thin wrapper around a std::string.
using String = std::string;


//! Internal name for standard size type, used for sizes of containers.
using SizeType = std::size_t;
//! Internal name for standard difference type of container indices and pointers.
using PointerDifferenceType = std::ptrdiff_t;

//! Internal name for unsigned integer used as a step counter
using CounterType = std::uint32_t;

class Nat32;
class Int32;
class Nat64;
class Int64;

using std::declval;

//! Internal alias for standard shared pointer.
template<class T> using SharedPointer = std::shared_ptr<T>;
//! Internal alias for standard initializer list.
template<class T> using InitializerList = std::initializer_list<T>;
//! Internal alias for standard pair.
template<class T1, class T2> using Pair = std::pair<T1,T2>;
//! Internal alias for standard tuple.
template<class... TS> using Tuple = std::tuple<TS...>;

//! A thin wrapper around a std::vector.
template<class T> using List = std::vector<T>;

//! A tag for the size of a scalar object.
struct SizeOne { operator SizeType() const { return 1u; } };
//! A tag for an index into a scalar object.
struct IndexZero { operator SizeType() const { return 0u; } };

//! The type used for the degree of an index.
typedef std::uint16_t DegreeType;
//! The type used for the dimension of a geometric object.
typedef SizeType DimensionType;

//! Metaprogramming.
template<class F, class... AS> using InvokeResult = typename std::invoke_result<F,AS...>::type;
template<class SIG> struct ResultOfTrait;
template<class F, class... AS> struct ResultOfTrait<F(AS...)> { typedef typename std::invoke_result<F,AS...>::type Type; };
template<class SIG> using ResultOf = typename ResultOfTrait<SIG>::Type;

//! Concurrency typedefs.
using ConditionVariable = std::condition_variable;
using Mutex = std::mutex;
template<class T> using LockGuard = std::lock_guard<T>;
template<class T> using UniqueLock = std::unique_lock<T>;
using ThreadId = std::thread::id;
using VoidFunction = std::function<Void()>;
template<class T> using Future = std::future<T>;
template<class T> using Promise = std::promise<T>;
template<class T> using PackagedTask = std::packaged_task<T>;


} // namespace ConcManager

#endif
