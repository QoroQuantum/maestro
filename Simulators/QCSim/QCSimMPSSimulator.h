#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "MPSSimulator.h"
#include "QCSimTensorChainSimulator.h"
#define _USE_MATH_DEFINES
#include <math.h>

namespace Simulators::Private
{
// matrix_product_state backend. Owns exactly one native QCSim implementation.
class QCSimMPSSimulator : public QCSimTensorChainSimulator
{
  public:
    QCSimMPSSimulator() : QCSimTensorChainSimulator(false)
    {
        configuration.SetConfiguration("method", "matrix_product_state");
    }

    const char *MethodName() const override
    {
        return "matrix_product_state";
    }

    void Initialize() override
    {
        if (nrQubits != 0)
        {
            {
                mpsSimulator = std::make_unique<QC::TensorNetworks::MPSSimulator>(nrQubits);
                if (!useOptimalMeetingPosition)
                    mpsSimulator->SetUseOptimalMeetingPosition(false);
                InstallBondSummary(*mpsSimulator, bondDimensionCallback);
                curMaxBondDim = 1;
            }
            SetMultithreading(enableMultithreading);
            for (const auto &[key, value] : configuration.GetConfigMap())
                if (key != "method")
                    Configure(key.c_str(), value.c_str());
        }
    }

    void InitializeToBasisState(size_t num_qubits, Types::qubit_t basisState) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        mpsSimulator->setToBasisState(static_cast<size_t>(basisState));
    }

    void InitializeToBasisState(size_t num_qubits, const std::vector<bool> &basisState) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        mpsSimulator->setToBasisState(basisState);
    }

    void Reset() override
    {
        if (mpsSimulator)
        {
            mpsSimulator->Clear();
            curMaxBondDim = 1;
        }
        upcomingGateIndex = 0;
        ResetDummySimulator();
    }

    void SetInitialQubitsMap(const std::vector<long long int> &initialMap) override
    {
        if (mpsSimulator)
        {
            mpsSimulator->SetInitialQubitsMap(initialMap);
            if (!dummySim || dummySim->getNrQubits() != initialMap.size() || dummySim->IsOperatorChain() != false)
            {
                dummySim = std::make_unique<Simulators::MPSDummySimulator>(initialMap.size(), false);
                dummySim->SetMaxBondDimension(configuration.GetConfigurationAsInt(MaxBondDimensionConfigKey()));
            }
            dummySim->setGrowthFactorGate(growthFactorGate);
            dummySim->setGrowthFactorSwap(growthFactorSwap);
            dummySim->SetInitialQubitsMap(initialMap);
        }
    }

    void SetUseOptimalMeetingPosition(bool enable) override
    {
        useOptimalMeetingPosition = enable;
        if (mpsSimulator)
            mpsSimulator->SetUseOptimalMeetingPosition(enable);
        RefreshRoutingCallback();
    }

    void Configure(const char *key, const char *value) override
    {
        if (std::string("method") == key)
        {
            ValidateMethod(value);
            configuration.SetConfiguration(key, value);
            return;
        }
        if (!configuration.WasApplied(key, value))
            configuration.SetConfiguration(key, value);
        if (std::string("seed") == key)
        {
            const uint64_t seed = std::stoull(value);
            SeedAuxiliaryRng(seed);
            nextSeedStream = 0;
            rng.seed(seed);
            if (mpsSimulator)
                SeedBackend(mpsSimulator.get(), seed);
            return;
        }
        if (mpsSimulator)
        {
            if (std::string(key) == "matrix_product_state_max_bond_dimension")
            {
                mpsSimulator->setLimitBondDimension(configuration.GetConfigurationAsInt(key));
            }
            else if (std::string(key) == "matrix_product_state_truncation_threshold")
            {
                const double threshold = configuration.GetConfigurationAsDouble(key);
                if (threshold >= 0.)
                    mpsSimulator->setLimitEntanglement(threshold);
            }
            else if (std::string(key) == "matrix_product_state_truncation_mode")
            {
                if (std::string(value) == "relative_max")
                    mpsSimulator->setTruncationMode(QC::TensorNetworks::MPSSimulator::TruncationMode::RelativeToMax);
                else if (std::string(value) == "discarded_weight")
                    mpsSimulator->setTruncationMode(QC::TensorNetworks::MPSSimulator::TruncationMode::DiscardedWeight);
            }
        }
        if (std::string(key) == "pauli_propagator_workers")
        {
            const auto workers = configuration.GetConfigurationAsUnsigned(key);
            if (workers > 1024)
                throw std::invalid_argument("pauli_propagator_workers exceeds 1024");
        }
    }

    size_t AllocateQubits(size_t num_qubits) override
    {
        if (mpsSimulator)
            return 0;
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        mpsSimulator = nullptr;
        dummySim = nullptr;
        nrQubits = 0;
        upcomingGateIndex = 0;
        upcomingGates.clear();
    }

    size_t Measure(const Types::qubits_vector &qubits) override
    {
        if (qubits.size() > sizeof(size_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the size_t type, the outcome will be undefined"
                      << std::endl;
        size_t res = 0;
        size_t mask = 1ULL;
        DontNotify();
        {
            const std::set<Eigen::Index> qubitsSet(qubits.begin(), qubits.end());
            auto measured = mpsSimulator->MeasureQubits(qubitsSet);
            for (Types::qubit_t qubit : qubits)
            {
                if (measured[qubit])
                    res |= mask;
                mask <<= 1;
            }
        }
        Notify();
        NotifyObservers(qubits);
        return res;
    }

    std::vector<bool> MeasureMany(const Types::qubits_vector &qubits) override
    {
        std::vector<bool> res(qubits.size(), false);
        DontNotify();
        {
            const std::set<Eigen::Index> qubitsSet(qubits.begin(), qubits.end());
            auto measured = mpsSimulator->MeasureQubits(qubitsSet);
            for (size_t q = 0; q < qubits.size(); ++q)
                if (measured[qubits[q]])
                    res[q] = true;
        }
        Notify();
        NotifyObservers(qubits);
        return res;
    }

    void ApplyReset(const Types::qubits_vector &qubits) override
    {
        DontNotify();
        {
            for (size_t qubit : qubits)
                if (mpsSimulator->MeasureQubit(static_cast<unsigned int>(qubit)))
                    mpsSimulator->ApplyGate(xgate, static_cast<unsigned int>(qubit));
        }
        Notify();
        NotifyObservers(qubits);
    }

    void MoveAtBeginningOfChain(const Types::qubits_vector &qubits) override
    {
        if (!mpsSimulator)
            return IState::MoveAtBeginningOfChain(qubits);
        const auto selected = MPOValidation::Qubits(qubits, nrQubits);
        const std::set<Eigen::Index> targets(selected.begin(), selected.end());
        mpsSimulator->MoveAtBeginningOfChain(targets);
    }

    std::vector<std::complex<double>> GetStateVector() override
    {
        if (!mpsSimulator)
            return IState::GetStateVector();
        if (!nrQubits)
            return {};
        const auto count = TensorQueries::DenseSize(nrQubits);
        const auto values = mpsSimulator->getRegisterStorage();
        if (static_cast<size_t>(values.size()) != count)
            throw std::runtime_error("QCSim statevector query returned an invalid size");
        return {values.data(), values.data() + values.size()};
    }

    std::complex<double> ExpectationValueOperators(const Types::qubits_vector &qubits, const std::vector<Eigen::MatrixXcd> &matrices) override
    {
        if (!mpsSimulator)
            return IState::ExpectationValueOperators(qubits, matrices);
        TensorQueries::Operators(qubits, matrices, nrQubits);
        std::vector<QC::Gates::AppliedGate<>> gates;
        gates.reserve(qubits.size());
        for (size_t i = 0; i < qubits.size(); ++i)
            gates.emplace_back(matrices[i], qubits[i]);
        return mpsSimulator->ExpectationValue(gates);
    }

    void Trim() override
    {
        if (mpsSimulator)
            mpsSimulator->Trim();
        else
            throw std::runtime_error("Trim is only available for QCSim MPS and MPO");
    }

    void ReCanonicalize() override
    {
        if (mpsSimulator)
            mpsSimulator->ReCanonicalize();
        else
            throw std::runtime_error("Canonicalization is only available for QCSim MPS and MPO");
    }

    double Probability(Types::qubit_t outcome) override
    {
        return mpsSimulator->getBasisStateProbability(static_cast<unsigned int>(outcome));
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        return mpsSimulator->getBasisStateAmplitude(static_cast<unsigned int>(outcome));
    }

    std::complex<double> ProjectOnZero() override
    {
        return mpsSimulator->ProjectOnZero();
    }

    std::vector<double> AllProbabilities() override
    {
        const Eigen::VectorXcd probs = mpsSimulator->getRegisterStorage().cwiseAbs2();
        std::vector<double> result(probs.size());
        for (int i = 0; i < probs.size(); ++i)
            result[i] = probs[i].real();
        return result;
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        std::vector<double> result(qubits.size());
        {
            for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
                result[i] = mpsSimulator->getBasisStateProbability(qubits[i]);
        }
        return result;
    }

    std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (qubits.empty() || shots == 0)
            return {};
        if (qubits.size() > sizeof(size_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the size_t type, the outcome will be undefined"
                      << std::endl;
        std::unordered_map<Types::qubit_t, Types::qubit_t> result;
        DontNotify();
        {
            bool normal = true;
            if (!configuration.IsSet("mps_sample_measure_algorithm") || configuration.GetConfiguration("mps_sample_measure_algorithm") == "mps_probabilities")
            {
                const std::set<Eigen::Index> qset(qubits.begin(), qubits.end());
                if (qset.size() == GetNumberOfQubits())
                {
                    normal = false;
                    for (size_t shot = 0; shot < shots; ++shot)
                    {
                        const size_t measRaw = MeasureNoCollapse();
                        size_t meas = 0;
                        size_t mask = 1ULL;
                        for (auto q : qubits)
                        {
                            const size_t qubitMask = 1ULL << q;
                            if (measRaw & qubitMask)
                                meas |= mask;
                            mask <<= 1ULL;
                        }
                        ++result[meas];
                    }
                }
                else if (qset.size() > 1)
                {
                    mpsSimulator->MoveAtBeginningOfChain(qset);
                    normal = false;
                    for (size_t shot = 0; shot < shots; ++shot)
                    {
                        const auto measRaw = mpsSimulator->MeasureNoCollapse(qset);
                        size_t meas = 0;
                        size_t mask = 1ULL;
                        for (auto q : qubits)
                        {
                            if (measRaw.at(q))
                                meas |= mask;
                            mask <<= 1ULL;
                        }
                        ++result[meas];
                    }
                }
                else if (qset.size() == 1)
                {
                    normal = false;
                    const auto prob0 = mpsSimulator->GetProbability(qubits[0]);
                    for (size_t shot = 0; shot < shots; ++shot)
                    {
                        const size_t meas = uniformZeroOne(rng) < prob0 ? 0ULL : 1ULL;
                        size_t m = meas;
                        for (size_t i = 1; i < qubits.size(); ++i)
                        {
                            m <<= 1ULL;
                            m |= meas;
                        }
                        ++result[m];
                    }
                }
            }
            if (normal)
            {
                auto savedState = mpsSimulator->getState();
                for (size_t shot = 0; shot < shots; ++shot)
                {
                    const size_t meas = Measure(qubits);
                    ++result[meas];
                    mpsSimulator->setState(savedState);
                }
            }
        }
        Notify();
        NotifyObservers(qubits);
        return result;
    }

    std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (qubits.empty() || shots == 0)
            return {};
        std::unordered_map<std::vector<bool>, Types::qubit_t> result;
        DontNotify();
        {
            bool normal = true;
            if (!configuration.IsSet("mps_sample_measure_algorithm") || configuration.GetConfiguration("mps_sample_measure_algorithm") == "mps_probabilities")
            {
                const std::set<Eigen::Index> qset(qubits.begin(), qubits.end());
                if (qset.size() == GetNumberOfQubits())
                {
                    normal = false;
                    for (size_t shot = 0; shot < shots; ++shot)
                    {
                        const auto meas = MeasureNoCollapseMany();
                        std::vector<bool> measVec(qubits.size());
                        for (size_t i = 0; i < qubits.size(); ++i)
                            measVec[i] = meas[qubits[i]];
                        ++result[measVec];
                    }
                }
                else if (qset.size() > 1)
                {
                    mpsSimulator->MoveAtBeginningOfChain(qset);
                    normal = false;
                    for (size_t shot = 0; shot < shots; ++shot)
                    {
                        const auto meas = mpsSimulator->MeasureNoCollapse(qset);
                        std::vector<bool> measVec(qubits.size());
                        for (size_t i = 0; i < qubits.size(); ++i)
                            measVec[i] = meas.at(qubits[i]);
                        ++result[measVec];
                    }
                }
                else if (qset.size() == 1)
                {
                    normal = false;
                    const auto prob0 = mpsSimulator->GetProbability(qubits[0]);
                    for (size_t shot = 0; shot < shots; ++shot)
                    {
                        const size_t meas = uniformZeroOne(rng) < prob0 ? 0ULL : 1ULL;
                        const std::vector<bool> m(qubits.size(), meas);
                        ++result[m];
                    }
                }
            }
            if (normal)
            {
                auto savedState = mpsSimulator->getState();
                for (size_t shot = 0; shot < shots; ++shot)
                {
                    const auto meas = MeasureMany(qubits);
                    ++result[meas];
                    mpsSimulator->setState(savedState);
                }
            }
        }
        Notify();
        NotifyObservers(qubits);
        return result;
    }

    std::vector<double> ExpectationValues(const std::vector<std::string> &paulis) override
    {
        std::vector<double> result(paulis.size(), 1.);
        std::vector<std::string> selected;
        std::vector<size_t> indices;
        const size_t n = GetNumberOfQubits();
        for (size_t i = 0; i < paulis.size(); ++i)
        {
            if (paulis[i].empty())
                continue;
            bool zero = false;
            for (size_t q = n; q < paulis[i].size(); ++q)
            {
                const auto p = toupper(static_cast<unsigned char>(paulis[i][q]));
                if (p != 'I' && p != 'Z')
                {
                    zero = true;
                    break;
                }
            }
            if (zero)
            {
                result[i] = 0.;
                continue;
            }
            std::string pauli = paulis[i];
            pauli.resize(n, 'I');
            for (char &p : pauli)
            {
                p = static_cast<char>(toupper(static_cast<unsigned char>(p)));
                if (p != 'X' && p != 'Y' && p != 'Z')
                    p = 'I';
            }
            indices.push_back(i);
            selected.push_back(std::move(pauli));
        }
        if (selected.empty())
            return result;
        const auto values = TensorExpectationBatch(*mpsSimulator, selected);
        for (size_t i = 0; i < indices.size(); ++i)
            result[indices[i]] = values[i].real();
        return result;
    }

    double ExpectationValue(const std::string &pauliStringOrig) override
    {
        if (pauliStringOrig.empty())
            return 1.0;
        std::string pauliString = pauliStringOrig;
        if (pauliString.size() > GetNumberOfQubits())
        {
            for (size_t i = GetNumberOfQubits(); i < pauliString.size(); ++i)
            {
                const auto pauliOp = toupper(pauliString[i]);
                if (pauliOp != 'I' && pauliOp != 'Z')
                    return 0.0;
            }
            pauliString.resize(GetNumberOfQubits());
        }
        std::vector<QC::Gates::AppliedGate<Eigen::MatrixXcd>> pauliStringVec;
        pauliStringVec.reserve(pauliString.size());
        for (size_t q = 0; q < pauliString.size(); ++q)
        {
            switch (toupper(pauliString[q]))
            {
            case 'X': {
                QC::Gates::AppliedGate<Eigen::MatrixXcd> ag(xgate.getRawOperatorMatrix(), static_cast<Types::qubit_t>(q));
                pauliStringVec.emplace_back(std::move(ag));
            }
            break;
            case 'Y': {
                QC::Gates::AppliedGate<Eigen::MatrixXcd> ag(ygate.getRawOperatorMatrix(), static_cast<Types::qubit_t>(q));
                pauliStringVec.emplace_back(std::move(ag));
            }
            break;
            case 'Z': {
                QC::Gates::AppliedGate<Eigen::MatrixXcd> ag(zgate.getRawOperatorMatrix(), static_cast<Types::qubit_t>(q));
                pauliStringVec.emplace_back(std::move(ag));
            }
            break;
            case 'I':
                [[fallthrough]];
            default:
                break;
            }
        }
        if (pauliStringVec.empty())
            return 1.0;
        return mpsSimulator->ExpectationValue(pauliStringVec).real();
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kMatrixProductState;
    }

    void SaveState() override
    {
        mpsSimulator->SaveState();
    }

    void RestoreState() override
    {
        mpsSimulator->RestoreState();
    }

    void SetMultithreading(bool multithreading = true) override
    {
        enableMultithreading = multithreading;
        if (mpsSimulator)
            mpsSimulator->SetMultithreading(multithreading);
    }

    Types::qubit_t MeasureNoCollapse() override
    {
        if (GetNumberOfQubits() > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;
        {
            const auto measured = mpsSimulator->MeasureNoCollapse();
            Types::qubit_t result = 0;
            Types::qubit_t mask = 1;
            for (Types::qubit_t q = 0; q < measured.size(); ++q)
            {
                if (measured.at(q))
                    result |= mask;
                mask <<= 1;
            }
            return result;
        }
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        const auto measured = mpsSimulator->MeasureNoCollapse();
        std::vector<bool> res(nrQubits);
        for (size_t i = 0; i < nrQubits; ++i)
            res[i] = measured.at(i);
        return res;
    }

    void ApplyGenericOneQubitGate(Types::qubit_t qubit, const Eigen::Matrix2cd &gate) override
    {
        const QC::Gates::AppliedGate<> agate(gate, qubit);
        mpsSimulator->ApplyGate(agate);
        NotifyObservers({qubit});
    }

    void ApplyGenericTwoQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, const Eigen::Matrix4cd &gate) override
    {
        const QC::Gates::AppliedGate<> agate(gate, qubit0, qubit1);
        mpsSimulator->ApplyGate(agate);
        NotifyObservers({qubit0, qubit1});
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        pgate.SetPhaseShift(lambda);
        mpsSimulator->ApplyGate(pgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        mpsSimulator->ApplyGate(xgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        mpsSimulator->ApplyGate(ygate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        mpsSimulator->ApplyGate(zgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        mpsSimulator->ApplyGate(h, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        mpsSimulator->ApplyGate(sgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        mpsSimulator->ApplyGate(sdggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        mpsSimulator->ApplyGate(tgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        mpsSimulator->ApplyGate(tdggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        mpsSimulator->ApplyGate(sxgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        mpsSimulator->ApplyGate(sxdaggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        mpsSimulator->ApplyGate(k, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        rxgate.SetTheta(theta);
        mpsSimulator->ApplyGate(rxgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        rygate.SetTheta(theta);
        mpsSimulator->ApplyGate(rygate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        rzgate.SetTheta(theta);
        mpsSimulator->ApplyGate(rzgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        ugate.SetParams(theta, phi, lambda, gamma);
        mpsSimulator->ApplyGate(ugate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mpsSimulator->ApplyGate(cxgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mpsSimulator->ApplyGate(cygate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mpsSimulator->ApplyGate(czgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        cpgate.SetPhaseShift(lambda);
        mpsSimulator->ApplyGate(cpgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crxgate.SetTheta(theta);
        mpsSimulator->ApplyGate(crxgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crygate.SetTheta(theta);
        mpsSimulator->ApplyGate(crygate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crzgate.SetTheta(theta);
        mpsSimulator->ApplyGate(crzgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mpsSimulator->ApplyGate(ch, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mpsSimulator->ApplyGate(csx, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mpsSimulator->ApplyGate(csxdag, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        mpsSimulator->ApplyGate(swapgate, static_cast<unsigned int>(qubit1), static_cast<unsigned int>(qubit0));
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        const size_t q1 = qubit0;
        const size_t q2 = qubit1;
        const size_t q3 = qubit2;
        mpsSimulator->ApplyGate(csx, static_cast<unsigned int>(q3), static_cast<unsigned int>(q2));
        NotifyObservers({qubit1, qubit2});
        mpsSimulator->ApplyGate(cxgate, static_cast<unsigned int>(q2), static_cast<unsigned int>(q1));
        NotifyObservers({qubit0, qubit1});
        mpsSimulator->ApplyGate(csxdag, static_cast<unsigned int>(q3), static_cast<unsigned int>(q2));
        NotifyObservers({qubit1, qubit2});
        mpsSimulator->ApplyGate(cxgate, static_cast<unsigned int>(q2), static_cast<unsigned int>(q1));
        NotifyObservers({qubit0, qubit1});
        mpsSimulator->ApplyGate(csx, static_cast<unsigned int>(q3), static_cast<unsigned int>(q1));
        NotifyObservers({qubit0, qubit2});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        const size_t q1 = ctrl_qubit;
        const size_t q2 = qubit0;
        const size_t q3 = qubit1;
        mpsSimulator->ApplyGate(cxgate, static_cast<unsigned int>(q2), static_cast<unsigned int>(q3));
        NotifyObservers({qubit1, qubit0});
        mpsSimulator->ApplyGate(csx, static_cast<unsigned int>(q3), static_cast<unsigned int>(q2));
        NotifyObservers({qubit0, qubit1});
        mpsSimulator->ApplyGate(cxgate, static_cast<unsigned int>(q2), static_cast<unsigned int>(q1));
        NotifyObservers({ctrl_qubit, qubit0});
        pgate.SetPhaseShift(M_PI);
        mpsSimulator->ApplyGate(pgate, static_cast<unsigned int>(q3));
        NotifyObservers({qubit1});
        pgate.SetPhaseShift(-M_PI_2);
        mpsSimulator->ApplyGate(pgate, static_cast<unsigned int>(q2));
        NotifyObservers({qubit0});
        mpsSimulator->ApplyGate(csx, static_cast<unsigned int>(q3), static_cast<unsigned int>(q2));
        NotifyObservers({qubit0, qubit1});
        mpsSimulator->ApplyGate(cxgate, static_cast<unsigned int>(q2), static_cast<unsigned int>(q1));
        NotifyObservers({ctrl_qubit, qubit0});
        pgate.SetPhaseShift(M_PI);
        mpsSimulator->ApplyGate(pgate, static_cast<unsigned int>(q3));
        NotifyObservers({qubit1});
        mpsSimulator->ApplyGate(csx, static_cast<unsigned int>(q3), static_cast<unsigned int>(q1));
        NotifyObservers({ctrl_qubit, qubit1});
        mpsSimulator->ApplyGate(cxgate, static_cast<unsigned int>(q2), static_cast<unsigned int>(q3));
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        cugate.SetParams(theta, phi, lambda, gamma);
        mpsSimulator->ApplyGate(cugate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        auto cloned = std::make_unique<QCSimMPSSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->enableMultithreading = enableMultithreading;
        cloned->lookaheadDepth = lookaheadDepth;
        cloned->lookaheadDepthWithHeuristic = lookaheadDepthWithHeuristic;
        cloned->useOptimalMeetingPosition = useOptimalMeetingPosition;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        cloned->growthFactorGate = growthFactorGate;
        cloned->growthFactorSwap = growthFactorSwap;
        if (mpsSimulator)
        {
            cloned->mpsSimulator = mpsSimulator->Clone();
            cloned->dummySim = dummySim ? dummySim->Clone() : nullptr;
            cloned->gateCounterObserver = std::make_shared<GateCounterObserver>(cloned->upcomingGateIndex);
            cloned->RegisterObserver(cloned->gateCounterObserver);
            cloned->curMaxBondDim = curMaxBondDim;
            cloned->mpsSimulator->SetMeetingPositionCallback(cloned->meetingPositionCallback);
            InstallBondSummary(*cloned->mpsSimulator, cloned->bondDimensionCallback);
        }
        for (const auto &[key, value] : configuration.GetConfigMap())
            cloned->Configure(key.c_str(), value.c_str());
        if (configuration.IsSet("seed"))
            cloned->SetSeed(DeriveSeed(std::stoull(configuration.GetConfiguration("seed")), nextSeedStream++));
        return cloned;
    }

  protected:
    std::vector<long long> CurrentRoutingMap() const override
    {
        if (mpsSimulator)
            return ReadRoutingMap(mpsSimulator.get());
        return {};
    }

    void RefreshRoutingCallback() override
    {
        const bool active = IsRoutingLookaheadEnabled() && !upcomingGates.empty();
        if (mpsSimulator)
            mpsSimulator->SetMeetingPositionCallback(active ? meetingPositionCallback : nullptr);
    }

    std::unique_ptr<QC::TensorNetworks::MPSSimulator> mpsSimulator;
};
} // namespace Simulators::Private
#endif
