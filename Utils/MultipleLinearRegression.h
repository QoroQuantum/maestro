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

#include <cmath>
#include <vector>
#include <Eigen/Eigen>

namespace Utils {

	// WARNING: it doesn't let it go lower than the minimum value, if the prediction is lower, the min value is returned!
	// set "TrueLinearRegression" to true if you want to use the real linear regression
	class MultipleLinearRegression {
	public:
        MultipleLinearRegression() = default;

		MultipleLinearRegression(const Eigen::VectorXd& initialWeights, double initialBias = 0.0, bool initialTrueLinearRegression = false)
			: W(initialWeights), b(initialBias), trueLinearRegression(initialTrueLinearRegression) {}

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

			Eigen::VectorXd Y;
			Y.resize(n);
			for (size_t i = 0; i < n; ++i)
				Y(i) = y[i];

			Eigen::MatrixXd Wa;

			/*
			Eigen::MatrixXd Xt = X.transpose();
			Eigen::MatrixXd XtX = Xt * X;

			if (abs(XtX.determinant()) < 1E-10)
				Wa = XtX.completeOrthogonalDecomposition().pseudoInverse() * Xt * Y;
			else
				Wa = XtX.inverse() * Xt * Y;
			*/

			if (alpha > 0.)
			{
				// Ridge (L2) regularization: minimize ||X * w - Y||^2 + lambda * ||w||^2.
				// The penalty strength is derived from the data so it adapts to the
				// (unscaled) feature magnitudes: lambda = alpha * trace(X^t * X) / m,
				// using only the feature columns (the bias column is left unregularized).
				// It is solved by augmenting the system with sqrt(lambda) * I rows (and
				// zeros in Y), which keeps the well-conditioned QR solve instead of
				// forming X^t * X.
				double trace = 0.;
				for (size_t j = 1; j <= m; ++j)
					trace += X.col(j).squaredNorm();

				const double lambda = (m > 0) ? alpha * trace / static_cast<double>(m) : 0.;

				Eigen::MatrixXd Xa(n + m, m + 1);
				Xa.topRows(n) = X;
				Xa.bottomRows(m).setZero();

				const double sqrtLambda = std::sqrt(lambda);
				for (size_t j = 0; j < m; ++j)
					Xa(n + j, j + 1) = sqrtLambda;

				Eigen::VectorXd Ya(n + m);
				Ya.head(n) = Y;
				Ya.tail(m).setZero();

				Wa = Xa.colPivHouseholderQr().solve(Ya);
			}
			else
				Wa = X.colPivHouseholderQr().solve(Y);

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

			const auto m = 1E-12;
			if (val < m)
				return m;

			return val;
		}

		void SetTrueLinearRegression(bool reg)
		{
			trueLinearRegression = reg;
		}

		// Ridge (L2) regularization factor (alpha). The effective penalty is derived
		// from the data as alpha * trace(X^t * X) / m, so it adapts to the (unscaled)
		// feature magnitudes. Set to 0 to disable. Applied before SetSamples.
		// A small value such as 1e-6 stabilizes collinear/rank-deficient fits without
		// materially biasing the weights.
		void SetRegularization(double reg)
		{
			alpha = reg;
		}

		const Eigen::VectorXd& GetWeights() const { return W; }
		double GetBias() const { return b; }
		bool IsTrueLinearRegression() const { return trueLinearRegression; }
		double GetRegularization() const { return alpha; }

	private:
		Eigen::VectorXd W;
		double b = 0.;

		bool trueLinearRegression = false;
		double alpha = 1e-6;
	};

}

#endif // __MULTIPLE_LINEAR_REGRESSION_H__
