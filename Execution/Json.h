// Validation helpers for the native request contract.
#pragma once
#include <boost/json.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace MaestroExecution {
namespace json = boost::json;
struct Error : std::runtime_error {
  std::string code;
  Error(std::string category, std::string message)
      : std::runtime_error(std::move(message)), code(std::move(category)) {}
};
inline void Require(bool condition, const std::string& message) {
  if (!condition) throw Error("invalid_input", message);
}
inline void Supported(bool condition, const std::string& message) {
  if (!condition) throw Error("unsupported_capability", message);
}
inline const json::object& Object(const json::value& value) {
  Require(value.is_object(), "Expected a JSON object");
  return value.as_object();
}
inline const json::array& Array(const json::value& value) {
  Require(value.is_array(), "Expected a JSON array");
  return value.as_array();
}
inline const json::value& Field(const json::object& object, const char* key) {
  const auto* value = object.if_contains(key);
  Require(value != nullptr, std::string("Missing field: ") + key);
  return *value;
}
inline const json::object& Sub(const json::object& object, const char* key) {
  static const json::object empty;
  const auto* value = object.if_contains(key);
  return value ? Object(*value) : empty;
}
inline std::string String(const json::value& value) {
  Require(value.is_string(), "Expected a string");
  std::string result(value.as_string());
  Require(result.find('\0') == std::string::npos,
          "NUL bytes are not supported");
  return result;
}
inline std::string String(const json::object& object, const char* key,
                          const std::string& fallback) {
  const auto* value = object.if_contains(key);
  return value ? String(*value) : fallback;
}
inline uint64_t UInt(const json::value& value) {
  if (value.is_uint64()) return value.as_uint64();
  Require(value.is_int64() && value.as_int64() >= 0,
          "Expected a nonnegative integer");
  return static_cast<uint64_t>(value.as_int64());
}
inline uint64_t UInt(const json::object& object, const char* key,
                     uint64_t fallback) {
  const auto* value = object.if_contains(key);
  return value ? UInt(*value) : fallback;
}
inline double Number(const json::value& value) {
  Require(value.is_number(), "Expected a finite number");
  const double number = value.to_number<double>();
  Require(std::isfinite(number), "Expected a finite number");
  return number;
}
inline double Number(const json::object& object, const char* key,
                     double fallback) {
  const auto* value = object.if_contains(key);
  return value ? Number(*value) : fallback;
}
inline bool Boolean(const json::value& value) {
  Require(value.is_bool(), "Expected a boolean");
  return value.as_bool();
}
inline bool Boolean(const json::object& object, const char* key,
                    bool fallback) {
  const auto* value = object.if_contains(key);
  return value ? Boolean(*value) : fallback;
}
inline void Keys(const json::object& object,
                 std::initializer_list<const char*> allowed) {
  std::unordered_set<std::string> names(allowed.begin(), allowed.end());
  for (const auto& entry : object)
    Require(names.count(std::string(entry.key())) != 0,
            "Unknown field: " + std::string(entry.key()));
}
inline json::array Complex(std::complex<double> value) {
  return {value.real(), value.imag()};
}
inline std::complex<double> Complex(const json::value& value) {
  if (value.is_number()) return {Number(value), 0};
  const auto& pair = Array(value);
  Require(pair.size() == 2, "A complex value must be [real, imaginary]");
  return {Number(pair[0]), Number(pair[1])};
}
inline std::string Scalar(const json::value& value) {
  if (value.is_string()) return String(value);
  Require(value.is_number() || value.is_bool(), "Expected a scalar option");
  return json::serialize(value);
}
}  // namespace MaestroExecution
