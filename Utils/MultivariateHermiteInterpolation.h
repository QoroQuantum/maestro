/**
 * @file MultivariateHermiteInterpolation.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Multivariate generalization of the Hermite interpolation.
 *
 * Interpolation is performed recursively: for each unique value of the first
 * coordinate, the samples sharing that value are passed to a child interpolator
 * over the remaining coordinates. When predicting, the child interpolators are
 * evaluated on the trailing coordinates, then a 1D Hermite interpolation is
 * applied to the resulting values over the first coordinate.
 *
 * Falls back to the univariate HermiteInterpolation when only one coordinate
 * remains.
 */

#pragma once

#ifndef __UTILS_MULTIVARIATE_HERMITE_INTERPOLATION_H__
#define __UTILS_MULTIVARIATE_HERMITE_INTERPOLATION_H__

#include <cassert>
#include <memory>
#include <vector>

#include "BivariateHermiteInterpolation.h"

namespace Utils {

    class MultivariateHermiteInterpolation {
    public:
        // Samples are assumed to be sorted lexicographically by the coordinates in x.
        // All x[i] must have the same dimension (>= 1).
        void SetSamples(const std::vector<std::vector<double>>& x, const std::vector<double>& y)
        {
            assert(x.size() == y.size());
            assert(!x.empty());

            if (x.empty() || y.empty()) return;

            assert(x[0].size() >= 2);

            leafInterpolator.reset();
            children.clear();
            xValues.clear();

            dimension = x[0].size();
            minValue = y[0];
            for (size_t i = 0; i < y.size(); ++i)
                minValue = std::min(minValue, y[i]);

            if (dimension == 2)
            {
                // degenerate to the bivariate case
                std::vector<std::vector<double>> xv;
                xv.reserve(x.size());
                for (size_t i = 0; i < x.size(); ++i)
                  xv.push_back(x[i]);

                leafInterpolator = std::make_unique<BivariateHermiteInterpolation>();
                leafInterpolator->SetTrueInterpolation(trueInterpolation);
                leafInterpolator->SetSamples(xv, y);
                return;
            }

            // Group samples by the first coordinate, then recurse on the rest.
            std::vector<std::vector<double>> subX;
            std::vector<double> subY;

            auto flushGroup = [&]() {
                if (subX.empty())
                    return;
                children.emplace_back(std::make_unique<MultivariateHermiteInterpolation>());
                children.back()->SetTrueInterpolation(trueInterpolation);
                children.back()->SetSamples(subX, subY);
                subX.clear();
                subY.clear();
            };

            xValues.push_back(x[0][0]);
            for (size_t i = 0; i < x.size(); ++i)
            {
                assert(x[i].size() == dimension);

                if (x[i][0] != xValues.back())
                {
                    flushGroup();
                    xValues.push_back(x[i][0]);
                }

                subX.emplace_back(x[i].begin() + 1, x[i].end());
                subY.push_back(y[i]);
            }
            flushGroup();
        }

        double Predict(const std::vector<double>& x) const
        {
            assert(x.size() == dimension);

            if (dimension < 2 || x.size() != dimension)
                return 0;

            if (leafInterpolator)
                return leafInterpolator->Predict(x);

            const std::vector<double> tail(x.begin() + 1, x.end());

            std::vector<double> vals;
            vals.reserve(children.size());
            for (size_t i = 0; i < children.size(); ++i)
                vals.push_back(children[i]->Predict(tail));

            HermiteInterpolation firstInterpolator;
            firstInterpolator.SetTrueInterpolation(trueInterpolation);
            firstInterpolator.SetSamples(xValues, vals);

            const double val = firstInterpolator.Predict(x[0]);

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

            if (leafInterpolator)
                leafInterpolator->SetTrueInterpolation(reg);

            for (auto& child : children)
                if (child)
                    child->SetTrueInterpolation(reg);
        }

    private:
        // Used when dimension == 2.
        std::unique_ptr<BivariateHermiteInterpolation> leafInterpolator;

        // Used when dimension > 2: one child per unique value of the first coordinate.
        std::vector<std::unique_ptr<MultivariateHermiteInterpolation>> children;
        std::vector<double> xValues;

        size_t dimension = 0;

        //static constexpr double divisor = 8;
        double minValue = 0;
        bool trueInterpolation = false;
    };

}

#endif // __UTILS_MULTIVARIATE_HERMITE_INTERPOLATION_H__
