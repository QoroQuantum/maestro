/**
 * @file BivariateHermiteInterpolation.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Interpolates in the first dimension using the normal Hermite interpolation,
 * then the obtained values are used for a second Hermite interpolation in the second direction.
 */

#pragma once

#ifndef __UTILS_BIVARIATE_HERMITE_INTERPOLATION_H__
#define __UTILS_BIVARIATE_HERMITE_INTERPOLATION_H__

#include "HermiteInterpolation.h"

namespace Utils {

	class BivariateHermiteInterpolation {
	public:
		// assumes the values in the proper order, if they are not, sort them out before calling this
		void SetSamples(const std::vector<std::vector<double>>& x, const std::vector<double>& y)
		{
			assert(x.size() == y.size());
			assert(!x.empty());
			assert(x[0].size() == 2);

			if (x.empty() || y.empty())
				return;

			interpolators.clear();
			xValues.clear();

			interpolators.emplace_back();
			interpolators.back().SetTrueInterpolation(trueInterpolation);
			xValues.push_back(x[0][0]);

			std::vector<double> xvals;
			std::vector<double> yvals;

			minValue = y[0];

			for (size_t i = 0; i < x.size(); ++i)
			{
				if (xValues.back() != x[i][0])
				{
					interpolators.back().SetSamples(xvals, yvals);
					xvals.clear();
					yvals.clear();
					interpolators.emplace_back();
					interpolators.back().SetTrueInterpolation(trueInterpolation);
					xValues.push_back(x[i][0]);
				}

				xvals.push_back(x[i][1]);
				yvals.push_back(y[i]);

				minValue = std::min(minValue, y[i]);
			}

			if (!xvals.empty())
				interpolators.back().SetSamples(xvals, yvals);
		}

		double Predict(const std::vector<double>& x) const
		{
			assert(x.size() == 2);

			if (x.size() < 2)
				return 0;

			std::vector<double> vals;
			vals.reserve(interpolators.size());

			for (size_t i = 0; i < interpolators.size(); ++i)
			{
				const double pred = interpolators[i].Predict(x[1]);
				//std::cout << "Pred: " << pred << std::endl;
				vals.push_back(pred);
			}

			HermiteInterpolation secondInterpolator;
			secondInterpolator.SetTrueInterpolation(trueInterpolation);
			secondInterpolator.SetSamples(xValues, vals);

			const double val = secondInterpolator.Predict(x[0]);

			if (trueInterpolation)
				return val;

			const auto m = 1E-12;//minValue / divisor;
			if (val < m)
				return m;

			return val;
		}

		void SetTrueInterpolation(bool reg)
		{
			trueInterpolation = reg;

			for (auto& interpolator : interpolators)
				interpolator.SetTrueInterpolation(reg);
		}

	private:
		std::vector<HermiteInterpolation> interpolators;
		std::vector<double> xValues;

		static constexpr double divisor = 8;
		double minValue = 0;
		bool trueInterpolation = false;
	};

}

#endif // __UTILS_BIVARIATE_HERMITE_INTERPOLATION_H__
