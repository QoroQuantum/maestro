#include "RealizationWorkers.h"

#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__) && !defined(_OPENMP_LLVM_RUNTIME)
namespace Utils::RealizationDetail
{
ProcessOpenMPBudgetState &ProcessOpenMPBudget()
{
    static ProcessOpenMPBudgetState state;
    return state;
}
} // namespace Utils::RealizationDetail
#endif
