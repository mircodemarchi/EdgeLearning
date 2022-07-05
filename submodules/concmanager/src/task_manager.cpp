/***************************************************************************
 *            concurrency/task_manager.cpp
 *
 *  Copyright  2007-21  Luca Geretti
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

#include "task_manager.hpp"

namespace ConcManager {

TaskManager::TaskManager() : _maximum_concurrency(std::thread::hardware_concurrency()), _concurrency(0), _pool(0) {}

SizeType TaskManager::maximum_concurrency() const {
    return _maximum_concurrency;
}

SizeType TaskManager::concurrency() const {
    LockGuard<Mutex> lock(_concurrency_mutex);
    return _concurrency;
}

void TaskManager::set_concurrency(SizeType value) {
    if(value > _maximum_concurrency)
    {
        throw std::runtime_error("TaskManager set concurrency precondition failed");
    }
    LockGuard<Mutex> lock(_concurrency_mutex);
    _concurrency = value;
    _pool.set_num_threads(value);
}

void TaskManager::set_maximum_concurrency() {
    set_concurrency(_maximum_concurrency);
}

} // namespace ConcManager