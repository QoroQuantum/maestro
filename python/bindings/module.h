#pragma once
#include <nanobind/nanobind.h>

namespace maestro_bindings {
void bind_config(nanobind::module_& m);
void bind_simulator(nanobind::module_& m);
void bind_circuit(nanobind::module_& m);
void bind_api(nanobind::module_& m);
void bind_noise(nanobind::module_& m);
void bind_checkpoint(nanobind::module_& m);
}  // namespace maestro_bindings
