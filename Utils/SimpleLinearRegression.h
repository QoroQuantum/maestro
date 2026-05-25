/**
 * @file SimpleLinearRegression.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Simple linear regression.
 *
 * Used for rough estimation, for example for execution time for circuits in various simulators.
 */

#pragma once

#ifndef __SIMPLE_LINEAR_REGRESSION_H__
#define __SIMPLE_LINEAR_REGRESSION_H__

#include <vector>

namespace Utils {

	// WARNING: it doesn't let it go lower than the minimum value, if the prediction is lower, the min value is returned!
	// set "TrueLinearRegression" to true if you want to use the real linear regression
	template<typename T1 = size_t, typename T2 = double> class SimpleLinearRegression {
	public:
		void SetSamples(const std::vector<T1>& x, const std::vector<T2>& y)
		{
			assert(x.size() == y.size());
			assert(!x.empty());

			if (x.empty() || y.empty())
				return;

			T2 sumX = 0;
			T2 sumY = 0;
			T2 sumXY = 0;
			T2 sumX2 = 0;

			const size_t cnt = std::min(x.size(), y.size());
			for (size_t i = 0; i < cnt; ++i)
			{
				sumX += x[i];
				sumY += y[i];
				sumXY += x[i] * y[i];
				sumX2 += x[i] * x[i];
			}

			W = (cnt * sumXY - sumX * sumY) / (cnt * sumX2 - sumX * sumX);
			b = (sumY - W * sumX) / cnt;
		}

		void SetSamplesSingleRegressor(const std::vector<T1>& x, const std::vector<T2>& y)
		{
			assert(x.size() == y.size());

			b = 0;

			T2 sumXY = 0;
			T2 sumX2 = 0;

			const size_t cnt = std::min(x.size(), y.size());
			for (size_t i = 0; i < cnt; ++i)
			{
				sumXY += x[i] * y[i];
				sumX2 += x[i] * x[i];
			}

			W = sumXY / sumX2;
		}

		T2 Predict(T1 x) const
		{
			const auto val = W * x + b;

			if (trueLinearRegression)
				return val;

			const auto m = 1E-12;
			if (val < m)
				return m;

			return val;
		}

		void SetTrueLinearRegression(bool reg)
		{
			trueLinearRegression = reg;
		}

		T2 GetW() const
		{
			return W;
		}

		T2 GetB() const
		{
			return b;
		}

	private:
		T2 W = 0;
		T2 b = 0;

		bool trueLinearRegression = false;
	};

}

#endif // __SIMPLE_LINEAR_REGRESSION_H__