#pragma once

/**
 * @brief Runs the SIMD self-test benchmark followed by an OpenMP thread-scaling
 *        benchmark (1..maxThreads threads) with the fastest SIMD implementation.
 *
 * @param maxThreads Upper thread count for the scaling sweep; <= 0 means
 *                   omp_get_max_threads().
 */
int RunBenchmark(int maxThreads = 0);
