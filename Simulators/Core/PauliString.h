#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace Simulators
{
inline void ValidatePauliString(const std::string &word)
{
    const auto invalid = word.find_first_not_of("IXYZixyz");
    if (invalid != std::string::npos)
        throw std::invalid_argument("Invalid Pauli character '" + word.substr(invalid, 1) + "' in observable.");
}

inline void ValidatePauliStrings(const std::vector<std::string> &words)
{
    for (const auto &word : words)
        ValidatePauliString(word);
}

// Unallocated qubits are in |0>. An X or Y there makes the expectation zero.
inline bool RestrictPauliToRegister(std::string &word, size_t width)
{
    ValidatePauliString(word);
    if (word.size() <= width)
        return true;
    if (word.find_first_of("XxYy", width) != std::string::npos)
        return false;
    word.resize(width);
    return true;
}
} // namespace Simulators
