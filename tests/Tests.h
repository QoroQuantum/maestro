/**
 * @file tests.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * A header file to include to ensure the InitSetup object is properly
 * initialized. Include it in every test file that needs time estimation for
 * execution.
 */

#pragma once

#ifndef _TESTS_H_

#ifdef COMPOSER
#include "../composer/Estimators/ExecutionEstimator.h"
#endif

class InitSetup {
 public:
  InitSetup() {
#ifdef COMPOSER
    Estimators::ExecutionEstimator<>::InitializeRegressors();
#endif
  }
};

#ifndef _TESTS_NO_EXCLUDE
extern
#endif
    InitSetup setup;

#endif
