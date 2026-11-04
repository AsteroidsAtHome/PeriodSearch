#include "threadstate.hpp"
#include "CalcStrategy.hpp"
#include "declarations.h"

#include <memory>
#include <mutex>
#include <vector>

namespace
{
	SIMDEnum g_threadSimd = SIMDEnum::Undefined;

	std::mutex g_stateMutex;
	// Owns all ThreadState objects; the thread_local pointers below observe
	// them. Freed at program exit (after the OpenMP threads have been joined).
	std::vector<std::unique_ptr<ThreadState>> g_statePool;

	thread_local ThreadState* g_tlsState = nullptr;
}

void SelectThreadStrategy(const SIMDEnum simd)
{
	g_threadSimd = simd;
}

ThreadState::ThreadState()
{
	// Deep copy of the master state. This is safe because setup is complete
	// (single-threaded) before the first parallel region runs, and the master
	// only reads `gl` afterwards.
	gl = ::gl;

	// Each thread needs its own strategy instance: the strategies keep
	// per-call scratch state (tmat, dtm, incl, dbr, ...) as members.
	gl.ctx.SetStrategy(CreateCalcStrategy(g_threadSimd));
}

ThreadState& GetThreadState()
{
	if (!g_tlsState)
	{
		std::lock_guard<std::mutex> lock(g_stateMutex);
		if (!g_tlsState) // double-checked
		{
			g_tlsState = new ThreadState();
			g_statePool.push_back(std::unique_ptr<ThreadState>(g_tlsState));
		}
	}
	return *g_tlsState;
}