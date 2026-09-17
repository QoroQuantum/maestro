/* Compile, link and exercise every request symbol from a C consumer. */
#include "../maestrolib/Interface.h"
#include <string.h>
static int accepted(char *value) {
  if (!value) return 0;
  const int valid = strstr(value, "\"ok\":true") != 0;
  FreeResult(value);
  return valid;
}
int maestro_request_c_header_test(void) {
  const char *request = "{\"schema_version\":2,\"operation\":\"state_probability\",\"target_state\":\"1\",\"circuit\":{\"num_qubits\":1,\"source\":\"OPENQASM 2.0; qreg q[1]; x q[0];\"}}";
  return accepted(MaestroGetCapabilitiesJson()) &&
         accepted(MaestroValidateRequestJson(request)) &&
         accepted(MaestroRunRequestJson(request)) &&
         accepted(MaestroFinalizeDistributedMpiGpuJson());
}
#ifdef MAESTRO_ABI_CONSUMER_MAIN
int main(void) { return maestro_request_c_header_test() ? 0 : 1; }
#endif
