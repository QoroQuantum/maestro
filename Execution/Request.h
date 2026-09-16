#pragma once
#include <boost/json.hpp>
namespace MaestroExecution {
constexpr unsigned SchemaVersion = 2;
constexpr size_t MaxRequestBytes = 16 * 1024 * 1024;
constexpr size_t MaxResultBytes = 64 * 1024 * 1024;
boost::json::object Capabilities();
boost::json::object Run(const boost::json::object& request, bool validate = false, unsigned depth = 0);
}
