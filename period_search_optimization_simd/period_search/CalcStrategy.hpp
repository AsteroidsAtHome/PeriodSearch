#pragma once
#include <memory>
#include <iostream>
#include "Enums.h"

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
	virtual ~CalcStrategy() = default;
	//virtual std::string doAlgorithm(std::string_view data) const = 0;

	virtual double mrqcof(double** x1, double** x2, double x3[], double y[],
							double sig[], double a[], int ia[], int ma,
							double** alpha, double beta[], int mfit, int lastone, int lastma) = 0;

	virtual double bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef) = 0;

	virtual double conv(int nc, double dres[], int ma) = 0;

	virtual void curv(double cg[]) = 0;

	virtual int gauss_errc(double** a, int n, double b[]) = 0;
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
	std::unique_ptr<CalcStrategy> strategy_;
	/**
	 * Usually, the Context accepts a strategy through the constructor, but also
	 * provides a setter to change it at runtime.
	 */
public:
	explicit CalcContext(std::unique_ptr<CalcStrategy>&& strategy = {}) : strategy_(std::move(strategy))
	{
	}
	/**
	 * Usually, the Context allows replacing a Strategy object at runtime.
	 */
	void set_strategy(std::unique_ptr<CalcStrategy>&& strategy)
	{
		strategy_ = std::move(strategy);
	}
	/**
	 * The Context delegates some work to the Strategy object instead of
	 * implementing +multiple versions of the algorithm on its own.
	 */
	/*void doSomeBusinessLogic() const
	{
		if (strategy_) {
			std::cout << "Context: Sorting data using the strategy (not sure how it'll do it)\n";
			std::string result = strategy_->doAlgorithm("aecbd");
			std::cout << result << "\n";
		}
		else {
			std::cout << "Context: Strategy isn't set\n";
		}
	}*/
	double CalculateMrqcof(double** x1, double** x2, double x3[], double y[],
							double sig[], double a[], int ia[], int ma,
							double** alpha, double beta[], int mfit, int lastone, int lastma) const
	{
		double result = 0.0;
		if (strategy_)
		{
			result = strategy_->mrqcof(x1, x2, x3, y, sig, a, ia, ma, alpha, beta, mfit, lastone, lastma);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}

		return result;
	}

	double CalculateBright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef)
	{
		double result = 0.0;
		if (strategy_)
		{
			result = strategy_->bright(ee, ee0, t, cg, dyda, ncoef);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}

		return result;
	}

	double CalculateConv(int nc, double dres[], int ma)
	{
		double result = 0.0;
		if (strategy_)
		{
			result = strategy_->conv(nc, dres, ma);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}

		return result;
	}

	void CalculateCurv(double cg[])
	{
		if (strategy_)
		{
			strategy_->curv(cg);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}
	}

	int CalculateGaussErrrc(double** a, int n, double b[])
	{
		int result = 0;
		if (strategy_)
		{
			result = strategy_->gauss_errc(a, n, b);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}

		return result;
	}
};