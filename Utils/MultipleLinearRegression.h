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

#ifndef __MULTIPLE_LINEAR_REGRESSION_H__
#define __MULTIPLE_LINEAR_REGRESSION_H__

#include <vector>
#include <Eigen/Eigen>

namespace Utils {

	// WARNING: it doesn't let it go lower than the minimum value, if the prediction is lower, the min value is returned!
	// set "TrueLinearRegression" to true if you want to use the real linear regression
	class MultipleLinearRegression {
	public:
        MultipleLinearRegression() = default;

		MultipleLinearRegression(const Eigen::VectorXd& initialWeights, double initialBias = 0.0, double initialMinValue = 0.0, bool initialTrueLinearRegression = false)
			: W(initialWeights), b(initialBias), minValue(initialMinValue), trueLinearRegression(initialTrueLinearRegression) {}

		void SetSamples(const std::vector<std::vector<double>>& x, const std::vector<double>& y)
		{
			assert(x.size() == y.size());
			assert(!x.empty());

			if (x.empty() || y.empty())
				return;

			const size_t n = std::min(x.size(), y.size());
			const size_t m = x[0].size();

			// if x is extended with 1 (for the bias term), then it's just a matrix thing, like this:
			// W = (X^t * X)^-1 * X^t * y
			// but from there the real W must be extracted and also b is going to be saved separately

			Eigen::MatrixXd X;
			X.resize(n, m + 1);
			X.col(0).setOnes();

			for (size_t i = 0; i < n; ++i)
				for (size_t j = 1; j <= m; ++j)
					X(i, j) = x[i][j - 1];


			minValue = y[0];

			Eigen::VectorXd Y;
			Y.resize(n);
			for (size_t i = 0; i < n; ++i)
			{
				Y(i) = y[i];
				minValue = std::min(minValue, y[i]);
			}

			Eigen::MatrixXd Xt = X.transpose();
			Eigen::MatrixXd XtX = Xt * X;
			
			Eigen::MatrixXd Wa;
			
			if (abs(XtX.determinant()) < 1E-10)
				Wa = XtX.completeOrthogonalDecomposition().pseudoInverse() * Xt * Y;
			else
				Wa = XtX.inverse() * Xt * Y;
			
			b = Wa(0);
			W.resize(m);
			for (size_t i = 0; i < m; ++i)
				W(i) = Wa(i + 1);
			//std::cout << "W: " << W << std::endl;
			//std::cout << "b: " << b << std::endl;
		}

		double Predict(const std::vector<double>& x) const
		{
			assert(x.size() == static_cast<size_t>(W.rows()));

			if (x.size() != static_cast<size_t>(W.size()))
				return 0.;

			Eigen::RowVectorXd xRow;
			xRow.resize(x.size());
			for (size_t i = 0; i < x.size(); ++i)
				xRow(i) = x[i];
			
			const auto val = xRow * W + b;

			if (trueLinearRegression)
				return val;

			const auto m = 1E-12;//minValue / divisor;
			if (val < m)
				return m;

			return val;
		}

		void SetTrueLinearRegression(bool reg)
		{
			trueLinearRegression = reg;
		}

		const Eigen::VectorXd& GetWeights() const { return W; }
		double GetBias() const { return b; }
		double GetMinValue() const { return minValue; }
		bool IsTrueLinearRegression() const { return trueLinearRegression; }

	private:
		Eigen::VectorXd W;
		double b = 0.;

		double minValue = 0.;
        bool trueLinearRegression = false;

		static constexpr double divisor = 8;
	};

}

#endif // __MULTIPLE_LINEAR_REGRESSION_H__
