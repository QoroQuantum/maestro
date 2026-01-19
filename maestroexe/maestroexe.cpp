#include <iostream>
#include <boost/program_options.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "Simulator.hpp"

static std::string _get_env_var(const char* envs) {
  std::string val;

#ifdef _WIN32
  size_t sz;

  getenv_s(&sz, NULL, 0, envs);
  if (sz == 0 || sz > 49) return val;

  char buf[50];

  getenv_s(&sz, buf, sz, envs);

  buf[49] = 0;
  val = buf;
#else
  const char* env = getenv(envs);
  if (env) val = env;
#endif

  return val;
}

static std::string GetConfigJson(int num_shots, int maxBondDim) {
  std::string config = "{\"shots\": ";

  config += std::to_string(num_shots);

  if (maxBondDim != 0)
    config += ", \"matrix_product_state_max_bond_dimension\": " +
              std::to_string(maxBondDim);

  config += "}";

  return config;
}

int main(int argc, char** argv) {
  try {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    boost::program_options::options_description desc("Allowed options");
    desc.add_options()("help,h", "Print a help description")(
        "version,v", "Print version information")(
        "nrqubits,n", boost::program_options::value<int>(),
        "Specify the number of qubits")(
        "shots,s", boost::program_options::value<int>(),
        "Specify the number of shots for execution")(
        "mbd,m", boost::program_options::value<int>(),
        "Specify the max bond dimension for the MPS simulator")(
        "simulator,r", boost::program_options::value<std::string>(),
        "Simulator type, either aer, qcsim, composite_aer, composite_qcsim or "
        "gpu")(
        "type,t", boost::program_options::value<std::string>(),
        "Simulation type, either statevector, mps, stabilizer or tensor")(
        "file,f", boost::program_options::value<std::string>(),
        "Provide a qasm file for execution")(
        "output,o", boost::program_options::value<std::string>(),
        "Specify the json output file");

    boost::program_options::positional_options_description pos_desc;
    pos_desc.add("file", 1);
    pos_desc.add("output", 1);

    boost::program_options::variables_map vars;
    try {
      boost::program_options::basic_command_line_parser parser(argc, argv);
      parser.positional(pos_desc);
      parser.options(desc);

      boost::program_options::store(parser.run(), vars);
      boost::program_options::notify(vars);
    } catch (boost::program_options::error& e) {
      std::cerr << "ERROR: " << e.what() << "\n";
      return 1;
    }

    if (vars.count("version")) std::cout << "Version 1.0\n";

    if (vars.count("help")) std::cout << desc << "\n";

    int nrQubits = 64;
    int nrShots = 1;
    int maxBondDim = 0;
    int simulatorType = 0;
    int simulationType = 0;

    if (vars.count("nrqubits")) {
      nrQubits = vars["nrqubits"].as<int>();

      const std::string qstr = _get_env_var("maestro_nrqubits");
      if (!qstr.empty()) {
        const int nrQubitsMax = std::stoi(qstr);
        if (nrQubits > nrQubitsMax) nrQubits = nrQubitsMax;
      }
    } else {
      const std::string qstr = _get_env_var("maestro_nrqubits");
      if (!qstr.empty()) nrQubits = std::stoi(qstr);
    }

    if (nrQubits <= 0) {
      std::cerr << "Invalid number of qubits" << std::endl;
      return 2;
    }

    if (vars.count("shots")) {
      nrShots = vars["shots"].as<int>();

      const std::string sstr = _get_env_var("maestro_shots");
      if (!sstr.empty()) {
        const int nrShotsMax = std::stoi(sstr);
        if (nrShots > nrShotsMax) nrShots = nrShotsMax;
      }
    } else {
      const std::string sstr = _get_env_var("maestro_shots");
      if (!sstr.empty()) nrShots = std::stoi(sstr);
    }

    if (nrShots <= 0) nrShots = 1;

    if (vars.count("mbd")) {
      maxBondDim = vars["mbd"].as<int>();

      const std::string mbds = _get_env_var("maestro_max_bond_dim");
      if (!mbds.empty()) {
        const int mbdMax = std::stoi(mbds);
        if (maxBondDim > mbdMax || (maxBondDim <= 0 && mbdMax > 0))
          maxBondDim = mbdMax;
      }
    } else {
      const std::string mbds = _get_env_var("maestro_max_bond_dim");
      if (!mbds.empty()) maxBondDim = std::stoi(mbds);
    }

    if (maxBondDim < 0) maxBondDim = 0;

    std::string stype;
    if (vars.count("simulator"))
      stype = vars["simulator"].as<std::string>();
    else
      stype = _get_env_var("maestro_simulator_type");

    if (!stype.empty()) {
      std::transform(stype.begin(), stype.end(), stype.begin(),
                     [](unsigned char chr) { return std::tolower(chr); });

      if (stype == "aer")
        simulatorType = 0;
      else if (stype == "qcsim")
        simulatorType = 1;
      else if (stype == "composite_aer" || stype == "pblocks_aer")
        simulatorType = 2;
      else if (stype == "composite_qcsim" || stype == "pblocks_qcsim")
        simulatorType = 3;
      else if (stype == "gpu")
        simulatorType = 4;
      else
        simulatorType = 1000;  // something big, so it won't be set
    }

    if (simulatorType != 2 && simulatorType != 3) {
      stype.clear();
      if (vars.count("type"))
        stype = vars["type"].as<std::string>();
      else
        stype = _get_env_var("maestro_simulation_type");

      if (!stype.empty()) {
        std::transform(stype.begin(), stype.end(), stype.begin(),
                       [](unsigned char chr) { return std::tolower(chr); });

        if (stype == "statevector" || stype == "sv")
          simulationType = 0;
        else if (stype == "mps" || stype == "matrix_product_state")
          simulationType = 1;
        else if (stype == "stabilizer" || stype == "clifford")
          simulationType = 2;
        else if (stype == "tensor" || stype == "tensor_network" ||
                 stype == "tn")
          simulationType = 3;
        else
          simulationType = 1000;
      }
    } else
      simulationType = 0;  // statevector for composite

    if (vars.count("file") == 0) {
      std::cerr << "No qasm file provided" << std::endl;
      return 3;
    }

    const std::string qasmFileName = vars["file"].as<std::string>();
    if (qasmFileName.empty()) {
      std::cerr << "Invalid qasm file" << std::endl;
      return 4;
    }

    std::ifstream file(qasmFileName);
    if (!file.is_open()) {
      std::cerr << "Couldn't read the qasm file" << std::endl;
      return 5;
    }

    std::string qasmStr((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    if (qasmStr.empty()) {
      std::cerr << "Empty qasm" << std::endl;
      return 6;
    }

    // after getting all params and so on, execute:

    SimpleSimulator simulator;
    if (!simulator.Init(
#if defined(_WIN32)
            "maestro.dll"
#else
            "libmaestro.so"

#endif
            )) {
      std::cerr << "Couldn't load maestro library" << std::endl;
      return 6;
    }

    simulator.CreateSimpleSimulator(nrQubits);

    if (simulatorType < 2)  // qcsim or aer
    {
      if (simulationType < 4)
        simulator.RemoveAllOptimizationSimulatorsAndAdd(
            static_cast<int>(simulatorType), static_cast<int>(simulationType));
      else {
        simulator.RemoveAllOptimizationSimulatorsAndAdd(
            static_cast<int>(simulatorType), 0);
        simulator.AddOptimizationSimulator(static_cast<int>(simulatorType), 1);
        simulator.AddOptimizationSimulator(static_cast<int>(simulatorType), 2);
      }
    } else if (simulatorType <
               4)  // composite, ignore exec type and set statevector
    {
      simulator.RemoveAllOptimizationSimulatorsAndAdd(
          static_cast<int>(simulatorType), 0);
    } else if (simulatorType == 4)  // gpu
    {
      if (simulationType < 2)  // statevector or mps
        simulator.RemoveAllOptimizationSimulatorsAndAdd(
            static_cast<int>(simulatorType), static_cast<int>(simulationType));
      else  // other types are not supported yet on gpu, set statevector
        simulator.RemoveAllOptimizationSimulatorsAndAdd(
            static_cast<int>(simulatorType), 0);
    }

    static std::string configStr = GetConfigJson(nrShots, maxBondDim);

    std::string result;
    if (!qasmStr.empty()) {
      char* res = simulator.SimpleExecute(qasmStr.c_str(), configStr.c_str());
      if (res) {
        result = res;
        simulator.FreeResult(res);
      }
    }

    // depending on params, write it on stdout or in a file

    if (vars.count("output")) {
      const std::string fileName = vars["output"].as<std::string>();
      if (!fileName.empty()) {
        std::ofstream outFile(fileName);
        outFile << result;
      }
    } else
      std::cout << result << std::endl;
  } catch (std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 7;
  } catch (...) {
    std::cerr << "Exception of unknown type!" << std::endl;
    return 8;
  }

  return 0;
}
