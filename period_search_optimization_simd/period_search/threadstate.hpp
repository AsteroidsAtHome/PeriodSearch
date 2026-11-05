// ReSharper disable CppInconsistentNaming
#pragma once
#include <vector>
#include "arrayHelpers.hpp"
#include "Enums.h"

/**
 * Per-worker-thread state for the parallel period scan.
 *
 * The application is parallelized over independent (period, pole) fits with
 * native worker threads (threadpool.hpp). Every fit needs its own mutable working state, because:
 *   - the calculation strategy (CalcStrategy*) keeps per-call scratch state
 *     as member variables (tmat, dtm, incl, dbr, ...), and
 *   - the globals struct holds per-fit tables and LM scratch (Area, Dg,
 *     alpha, covar, ytemp, dytemp, dyda, Chisq, atry, ...).
 *
 * A ThreadState therefore owns:
 *   - a deep copy of the master `globals` (shared setup data such as Nor,
 *     Darea, Lpoints, Inrel, Weight plus the per-fit scratch), and
 *   - its own `cg` parameter vector.
 *
 * Shared read-only data (the Fc/Fs/Tc/Ts/Pleg/Dsph tables and the Lmax/Mmax/
 * Ncoef/Numfac/Nphpar/Phi_0/Alamda_start/Alamda_incr constants) intentionally
 * remain file-scope globals and are shared by all threads.
 *
 * Instances are created lazily, once per worker thread (the pool threads are
 * created once and reused, so the state - and the memory it owns - persists across
 * parallel regions and is released at program exit).
 */
struct ThreadState
{
	globals gl;               // per-thread working copy of the master globals
	std::vector<double> cg;   // per-thread parameter vector (indexed 1..ma)

	// Deep-copies the master globals and creates the per-thread strategy
	// instance for the SIMD selected with SelectThreadStrategy().
	ThreadState();
};

/**
 * @brief Returns the calling thread's state, creating it on first use.
 *
 * Must only be called after the master `globals gl` has been fully set up
 * (i.e. inside the ParallelFor loops of the period scan, never during setup).
 */
ThreadState& GetThreadState();

/**
 * @brief Selects the SIMD strategy used by worker threads created afterwards.
 *
 * Called by the master thread before the first parallel region.
 */
void SelectThreadStrategy(SIMDEnum simd);
