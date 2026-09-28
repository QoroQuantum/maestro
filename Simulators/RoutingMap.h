#pragma once
#include <type_traits>
#include <utility>
#include <vector>

namespace Simulators::Private {
// Keep Maestro buildable with older QCSim releases. Without an authoritative
// map, use the native local optimizer instead of a possibly stale mirror.
template <class T, class = void>
struct HasRoutingMap : std::false_type {};
template <class T>
struct HasRoutingMap<
    T, std::void_t<decltype(std::declval<const T&>().getQubitsMap())>>
    : std::true_type {};
template <class T>
std::vector<long long> ReadRoutingMap(const T* simulator) {
  if constexpr (HasRoutingMap<T>::value) {
    if (simulator) {
      const auto& map = simulator->getQubitsMap();
      return {map.begin(), map.end()};
    }
  }
  return {};
}
}  // namespace Simulators::Private
