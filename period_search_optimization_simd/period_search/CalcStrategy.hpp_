// ReSharper disable CppInconsistentNaming
#pragma once
#include <memory>
#include <iostream>
#include <vector>
#include "arrayHelpers.hpp"

// look-ahead (in visible facets) of the Dg row prefetch in bright()
constexpr int DG_PREFETCH_ROWS = 8;

/**
 * The Strategy interface declares operations common to all supported versions
 * of some algorithm.
 *
 * The Context uses this interface to call the algorithm defined by Concrete
 * Strategies.
 */
class CalcStrategy
{
public:
	CalcStrategy() = default;
	virtual ~CalcStrategy() = default;

	virtual void mrqcof(std::vector<std::vector<double>>& x1, std::vector<std::vector<double>>& x2, std::vector<double>& x3, std::vector<double>& y,
		std::vector<double>& sig, std::vector<double>& a, std::vector<int>& ia, int ma,
		std::vector<double>& beta, int mfit, int lastone, int lastma, double& trial_chisq, globals& gl, const bool isCovar) = 0;

	virtual void bright(double t, std::vector<double>& cg, int ncoef, globals &gl) = 0;

	virtual void conv(int nc, int ma, globals &gl) = 0;

	virtual void curv(std::vector<double>& cg, globals &gl) = 0;

	virtual void gauss_errc(struct globals& gl, const int n, std::vector<double>& b, int &error) = 0;
};

/**
 * The Context defines the interface of interest to clients.
 */

class CalcContext
{
	/**
	 * @var Strategy The Context maintains a reference to one of the Strategy
	 * objects. The Context does not know the concrete class of a strategy. It
	 * should work with all strategies via the Strategy interface.
	 */
private:
	std::shared_ptr<CalcStrategy> strategy_;

	/**
	 * Usually, the Context accepts a strategy through the constructor, but also
	 * provides a setter to change it at runtime.
	 */
public:
	explicit CalcContext(std::shared_ptr<CalcStrategy>&& strategy = {}) : strategy_(std::move(strategy))
	{
	}

	/**
	 * Usually, the Context allows replacing a Strategy object at runtime.
	 */
	//void set_strategy(std::unique_ptr<CalcStrategy>&& strategy)
	void SetStrategy(std::shared_ptr<CalcStrategy> strategy)
	{
		strategy_ = std::move(strategy);
	}

	/**
	 * The Context delegates some work to the Strategy object instead of
	 * implementing +multiple versions of the algorithm on its own.
	 */

	void CalculateMrqcof(std::vector<std::vector<double>>& x1, std::vector<std::vector<double>>& x2, std::vector<double>& x3, std::vector<double>& y,
		std::vector<double>& sig, std::vector<double>& a, std::vector<int>& ia, int ma,
		std::vector<double>& beta, int mfit, int lastone, int lastma, double& mrq, globals& gl, const bool isCovar) const
	{
		if (strategy_)
		{
			strategy_->mrqcof(x1, x2, x3, y, sig, a, ia, ma, beta, mfit, lastone, lastma, mrq, gl, isCovar);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}
	}

	void CalculateBright(const double t, std::vector<double>& cg, const int ncoef, globals &gl) const
    {
		if (strategy_)
		{
			strategy_->bright(t, cg, ncoef, gl);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}
	}

	void CalculateConv(const int nc, const int ma, globals &gl) const
    {
		if (strategy_)
		{
			strategy_->conv(nc, ma, gl);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}
	}

	void CalculateCurv(std::vector<double>& cg, globals &gl) const
    {
		if (strategy_)
		{
			strategy_->curv(cg, gl);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}
	}

	void CalculateGaussErrc(struct globals& gl, const int n, std::vector<double>& b, int &error) const
    {
		if (strategy_)
		{
			strategy_->gauss_errc(gl, n, b, error);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}
	}
};