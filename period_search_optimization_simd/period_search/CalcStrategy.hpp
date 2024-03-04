#pragma once
#include <memory>
#include <iostream>
#include "Enums.h"
#if defined __x86_64__ || defined(__i386__) || defined(_WIN32)
  #include <immintrin.h>
#endif
#include "constants.h"

/**
 * The Strategy interface declares operations common to all supported versions
 * of some algorithm.
 *
 * The Context uses this interface to call the algorithm defined by Concrete
 * Strategies.
 */
class alignas(64) CalcStrategy
{
public:
	virtual ~CalcStrategy() = default;

	virtual void mrqcof(double** x1, double** x2, double x3[], double y[],
							double sig[], double a[], int ia[], int ma,
							double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq) = 0;

	virtual void bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double &br) = 0;

	virtual void conv(int nc, double dres[], int ma, double &result) = 0;

	virtual void curv(double cg[]) = 0;

	virtual void gauss_errc(double** a, int n, double b[], int &error) = 0;

//protected:
//	double alpha;
//	double cos_alpha;
//	double cl;
//	double cls;
//	double e[4]{};
//	double e0[4]{};
//	double php[N_PHOT_PAR + 1]{};
//	double dphp[N_PHOT_PAR + 1]{};
//	double de[4][4];
//	double de0[4][4]{};
//	double tmat[4][4]{};
//	double dtm[4][4][4]{};
//
//	int	ncoef0;
//	int incl_count;
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

	void CalculateMrqcof(double** x1, double** x2, double x3[], double y[],
							double sig[], double a[], int ia[], int ma,
							double** alpha, double beta[], int mfit, int lastone, int lastma, double &mrq) const
	{
		double result = 0.0;
		if (strategy_)
		{
			strategy_->mrqcof(x1, x2, x3, y, sig, a, ia, ma, alpha, beta, mfit, lastone, lastma, mrq);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}

		//return result;
	}

	void CalculateBright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double &br)
	{
		//double result = 0.0;
		if (strategy_)
		{
			//result = strategy_->bright(ee, ee0, t, cg, dyda, ncoef);
			strategy_->bright(ee, ee0, t, cg, dyda, ncoef, br);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}

		//return result;
	}

	void CalculateConv(int nc, double dres[], int ma, double &result)
	{
		//double result = 0.0;
		if (strategy_)
		{
			strategy_->conv(nc, dres, ma, result);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}

		//return result;
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

	void CalculateGaussErrc(double** a, int n, double b[], int &error)
	{
		//int result = 0;
		if (strategy_)
		{
			strategy_->gauss_errc(a, n, b, error);
		}
		else
		{
			std::cerr << "CalcContext: Strategy isn't set" << std::endl;
		}

		//return result;
	}
};