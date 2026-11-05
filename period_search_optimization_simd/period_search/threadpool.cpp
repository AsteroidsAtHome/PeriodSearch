#include "threadpool.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace
{
	using TaskFn = std::function<void(int, int)>;

	class ThreadPool
	{
	public:
		void Run(int n_threads, const int n_tasks, const TaskFn& body)
		{
			if (n_tasks <= 0)
				return;

			n_threads = (std::min)(n_threads, n_tasks);
			if (n_threads <= 1)
			{
				for (int task = 0; task < n_tasks; task++)
					body(task, 0);
				return;
			}

			// One loop at a time (the period search only ever runs one).
			std::lock_guard<std::mutex> runLock(runMutex_);
			EnsureWorkers(n_threads - 1);

			{
				std::lock_guard<std::mutex> lock(mutex_);
				job_ = &body;
				jobThreads_ = n_threads;
				nTasks_ = n_tasks;
				next_.store(0, std::memory_order_relaxed);
				pending_ = n_threads - 1;
				++generation_;
			}
			wake_.notify_all();

			Work(0);

			std::unique_lock<std::mutex> lock(mutex_);
			done_.wait(lock, [this] { return pending_ == 0; });
			job_ = nullptr;
		}

	private:
		// Grabs tasks until none are left.
		void Work(const int worker)
		{
			for (;;)
			{
				const int task = next_.fetch_add(1, std::memory_order_relaxed);
				if (task >= nTasks_)
					break;
				(*job_)(task, worker);
			}
		}

		void WorkerMain(const int worker)
		{
			unsigned long long seen = 0;
			for (;;)
			{
				{
					std::unique_lock<std::mutex> lock(mutex_);
					wake_.wait(lock, [&] { return generation_ != seen; });
					seen = generation_;
					if (worker >= jobThreads_)
						continue; // not needed for this loop
				}

				Work(worker);

				std::lock_guard<std::mutex> lock(mutex_);
				if (--pending_ == 0)
					done_.notify_one();
			}
		}

		void EnsureWorkers(const int count)
		{
			while (static_cast<int>(threads_.size()) < count)
			{
				const int worker = static_cast<int>(threads_.size()) + 1;
				threads_.emplace_back(&ThreadPool::WorkerMain, this, worker);
			}
		}

		std::mutex runMutex_;
		std::mutex mutex_;
		std::condition_variable wake_;
		std::condition_variable done_;
		std::vector<std::thread> threads_;

		const TaskFn* job_ = nullptr;
		int jobThreads_ = 0;
		int nTasks_ = 0;
		int pending_ = 0;
		unsigned long long generation_ = 0;
		std::atomic<int> next_{ 0 };
	};

	ThreadPool& Pool()
	{
		// Intentionally never destroyed: the idle workers block on a condition
		// variable and are ended with the process (also when it leaves through
		// exit() / boinc_finish()), so there is no join during static destruction.
		static ThreadPool* pool = new ThreadPool();
		return *pool;
	}
}

void ParallelFor(const int n_threads, const int n_tasks, const std::function<void(int task)>& body)
{
	Pool().Run(n_threads, n_tasks, [&body](const int task, int) { body(task); });
}

void ParallelForWorker(const int n_threads, const int n_tasks, const std::function<void(int task, int worker)>& body)
{
	Pool().Run(n_threads, n_tasks, body);
}

int HardwareThreads()
{
	const unsigned n = std::thread::hardware_concurrency();
	return n > 0 ? static_cast<int>(n) : 1;
}
