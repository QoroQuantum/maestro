/**
 * @file HermiteInterpolation.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Hermite interpolation/extrapolation using cubic Hermite polynomials.
 */

#pragma once

#ifndef __UTILS_HERMITE_INTERPOLATION_H__
#define __UTILS_HERMITE_INTERPOLATION_H__

#include <vector>
#include <boost/math/interpolators/pchip.hpp>

#include "SimpleLinearRegression.h"

namespace Utils {

class HermiteInterpolation {
 public:
  template <typename T = size_t>
  void SetSamples(const std::vector<T>& x, const std::vector<double>& y) {
    assert(x.size() == y.size());
    assert(!x.empty());
    if (x.empty() || y.empty()) return;

    spline.reset();

    std::vector<double> xvals(x.begin(), x.end());
    std::vector<double> yvals = y;
    x1 = xvals.front();
    x2 = xvals.back();
    y1 = yvals.front();
    y2 = yvals.back();

    if (xvals.size() == 1)
      slopel = sloper = 0;
    else {
      slopel = (yvals[1] - y1) / (xvals[1] - x1);
      sloper = (y2 - yvals[yvals.size() - 2]) / (x2 - xvals[xvals.size() - 2]);
    }


    bool fallbackToLinear = false;

    // needs 4 values to work, if we don't have them,
    // fallback to linear
    if (xvals.size() < 4) fallbackToLinear = true;
    else {
      // check the xvals to ensure they are strictly increasing, otherwise the
      // pchip interpolator will throw an exception
      // if that's the case, fallback to linear

      for (size_t i = 1; i < xvals.size(); ++i)
        if (xvals[i] <= xvals[i - 1]) {
          fallbackToLinear = true;
          break;
        }
    }

    if (fallbackToLinear) {
      linearExtrapolation = std::make_unique<SimpleLinearRegression<double, double>>();
      linearExtrapolation->SetTrueLinearRegression(trueInterpolation);
      linearExtrapolation->SetSamples(xvals, yvals);
      return;
    }

    spline = std::make_unique<
        boost::math::interpolators::pchip<std::vector<double>>>(
        std::move(xvals), std::move(yvals));
  }

  double Predict(double x) const {
    double result = 0;

    const bool smallx = x <= x1;
    const bool largex = x >= x2;
    if (smallx || largex) {
      if (smallx) {
        if (slopel == 0)
          result = y1;
        else
          result = y1 + slopel * (x - x1);
      } else {
        if (sloper == 0)
          result = y2;
        else
          result = y2 + sloper * (x - x2);
      }
    } else if (spline) {
      result = (*spline)(x);
    } else if (linearExtrapolation) {
      result = linearExtrapolation->Predict(x);
    } else {
      const double slope = (y2 - y1) / (x2 - x1);
      result = y1 + slope * (x - x1);
    }

    if (trueInterpolation) return result;

    const auto m = 1E-12;
    if (result < m) return m;

    return result;
  }

  void SetTrueInterpolation(bool reg) {
    trueInterpolation = reg;
    if (linearExtrapolation)
      linearExtrapolation->SetTrueLinearRegression(trueInterpolation);
  }

 private:
  std::unique_ptr<boost::math::interpolators::pchip<std::vector<double>>>
      spline;
  std::unique_ptr<SimpleLinearRegression<double, double>> linearExtrapolation;

  double x1 = 0;
  double x2 = 0;
  double y1 = 0;
  double y2 = 0;
  double slopel = 0;
  double sloper = 0;

  //static constexpr double divisor = 8;
  double minValue = 0;
  bool trueInterpolation = false;
};

}  // namespace Utils

#endif  // __UTILS_HERMITE_INTERPOLATION_H__
