// ReSharper disable CppInconsistentNaming
#pragma once
#include <functional>

/**
 * Native (std::thread) replacement for the OpenMP parallel loops.
 *
 * A single process-wide pool of worker threads is created lazily on first use
 * and reused for every parallel loop afterwards. Keeping the threads alive is
 * essential: every worker owns a lazily created ThreadState (a deep copy of the
 * master globals, see threadstate.hpp) that must persist across loops.
 *
 * Tasks are handed out dynamically (an atomic counter, like OpenMP
 * schedule(dynamic, 1)), in increasing task order. The calling thread takes
 * part as worker 0, so n_threads == 1 runs the loop inline with no
 * synchronization at all.
 */

/**
 * @brief Runs body(task) for every task in [0, n_tasks) on up to n_threads
 *        threads (the calling thread included) and returns when all are done.
 */
void ParallelFor(int n_threads, int n_tasks, const std::function<void(int task)>& body);

/**
 * @brief Like ParallelFor, but also passes the index of the executing worker
 *        (0 = calling thread, 1 .. n_threads - 1 = pool threads). A worker index
 *        always maps to the same OS thread, so it can index per-thread data.
 */
void ParallelForWorker(int n_threads, int n_tasks, const std::function<void(int task, int worker)>& body);

/**
 * @brief Number of hardware threads (std::thread::hardware_concurrency(), at least 1).
 */
int HardwareThreads();
