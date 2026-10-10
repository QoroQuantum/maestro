#pragma once

// Transform entry points are consumed across libmaestro's shared-library boundary.
#if defined(_WIN32)
#if defined(maestro_EXPORTS)
#define MAESTRO_TRANSFORM_API __declspec(dllexport)
#else
#define MAESTRO_TRANSFORM_API __declspec(dllimport)
#endif
#else
#define MAESTRO_TRANSFORM_API
#endif
