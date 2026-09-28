#pragma once

// Plain C ABI data types, also shared by dynamic loaders without importing
// the exported function declarations.
#include <stdint.h>

typedef struct MaestroGateFusionStatistics {
  uint64_t submittedGates;
  uint64_t backendGates;
  uint64_t fusedBlocks;
} MaestroGateFusionStatistics;
