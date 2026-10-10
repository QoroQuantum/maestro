#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "QCSimGateSimulator.h"
#include "QubitRegister.h"
#define _USE_MATH_DEFINES
#include <math.h>

namespace Simulators::Private
{
// Composite joins/splits replace amplitudes within an existing execution.
// Keeping the native register alive also keeps its measurement RNG position.
class SamplingQubitRegister : public QC::QubitRegister<>
{
  public:
    using QC::QubitRegister<>::QubitRegister;

    size_t MeasureQubitReproducible(size_t qubit)
    {
        if (qubit >= NrQubits)
            throw std::out_of_range("Measured qubit is outside the register");
        if (measuredMask)
            return MeasureRemainingQubit(qubit);
        if (!UseMultithreading())
        {
            const size_t outcome = QC::QubitRegister<>::MeasureQubit(qubit);
            RememberMeasurement(qubit, outcome);
            return outcome;
        }

        // Match native serial MeasureRange exactly: one native RNG draw,
        // the same cumulative scan, and the same outcome-probability sum.
        // Only the independent amplitude updates are parallel. In particular,
        // do not compute p(0) as 1-p(1), which changes normalization rounding.
        const double draw = 1. - uniformZeroOne(rng);
        const size_t mask = size_t{1} << qubit;
        const size_t outcome = (BaseClass::SampleBasisState(NrBasisStates, registerStorage, draw, 0, false) >> qubit) & 1;
        const size_t lowMask = mask - 1;
        const size_t selected = outcome << qubit;
        double probability = 0.;
        for (size_t k = 0; k < NrBasisStates / 2; ++k)
            probability += std::norm(registerStorage(((k & ~lowMask) << 1) | selected | (k & lowMask)));
        const double norm = 1. / std::sqrt(probability);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (!omp_in_parallel())
#endif
        for (long long i = 0; i < static_cast<long long>(NrBasisStates); ++i)
            registerStorage(i) = (static_cast<size_t>(i) & mask) == selected ? registerStorage(i) * norm : std::complex<double>(0., 0.);
        if (std::isfinite(norm))
            RememberMeasurement(qubit, outcome);
        return outcome;
    }

    void InvalidateMeasurements()
    {
        measuredMask = measuredValues = 0;
    }

    void SaveState()
    {
        // Execution clones share the immutable checkpoint, not live amplitudes.
        snapshot = std::make_shared<const Eigen::VectorXcd>(registerStorage);
    }

    void RestoreState()
    {
        if (snapshot)
            registerStorage = *snapshot;
        InvalidateMeasurements();
    }

    void DiscardMeasurementDraw()
    {
        // Consume the native distribution, just as MeasureNoCollapse does.
        // Its engine-word usage is an implementation detail of the library.
        (void)uniformZeroOne(rng);
    }

    void ReplaceAmplitudes(size_t qubits, Eigen::VectorXcd &amplitudes)
    {
        NrQubits = qubits;
        NrBasisStates = static_cast<size_t>(amplitudes.size());
        registerStorage.swap(amplitudes);
        savedStateStorage.resize(0);
        snapshot.reset();
        computeGates.clear();
        recordGates = false;
        InvalidateMeasurements();
    }

  private:
    void RememberMeasurement(size_t qubit, size_t outcome)
    {
        measuredMask |= size_t{1} << qubit;
        measuredValues = (measuredValues & ~(size_t{1} << qubit)) | (outcome << qubit);
    }

    // Insert fixed measured bits into an index in the remaining subspace.
    // Called once per tile, not once per amplitude.
    size_t Expand(size_t packed) const
    {
        size_t result = measuredValues;
        for (size_t free = (NrBasisStates - 1) & ~measuredMask; packed; packed >>= 1)
        {
            const size_t bit = free & (~free + 1);
            if (packed & 1)
                result |= bit;
            free &= free - 1;
        }
        return result;
    }

    size_t MeasureRemainingQubit(size_t qubit)
    {
        const double draw = 1. - uniformZeroOne(rng);
        const size_t mask = size_t{1} << qubit;
        double cumulative = 0.;
        size_t sampled = 0;
        // This is the native left-to-right sum with known zero terms skipped.
        // Keep normalization and RNG consumption for duplicate measurements too.
        for (size_t i = measuredValues; i < NrBasisStates; i = (((i | measuredMask) + 1) & ~measuredMask) | measuredValues)
        {
            cumulative += std::norm(registerStorage(i));
            if (draw <= cumulative)
            {
                sampled = i;
                break;
            }
        }
        const size_t outcome = (sampled >> qubit) & 1;
        const size_t selected = outcome << qubit;
        const size_t fixed = measuredMask | mask;
        const size_t values = (measuredValues & ~mask) | selected;
        double probability = 0.;
        if (!(measuredMask & mask) || (measuredValues & mask) == selected)
            for (size_t i = values; i < NrBasisStates; i = (((i | fixed) + 1) & ~fixed) | values)
                probability += std::norm(registerStorage(i));
        const double norm = 1. / std::sqrt(probability);
        // Preserve the native full update for exceptional, unnormalizable input.
        if (!std::isfinite(norm))
        {
            for (size_t i = 0; i < NrBasisStates; ++i)
                registerStorage(i) = (i & mask) == selected ? registerStorage(i) * norm : std::complex<double>(0., 0.);
            InvalidateMeasurements();
            return outcome;
        }
        size_t active = NrBasisStates;
        for (size_t bits = measuredMask; bits; bits &= bits - 1)
            active >>= 1;
        constexpr size_t tileSize = 1024;
        const size_t tiles = (active + tileSize - 1) / tileSize;
        const int threads = Utils::Sampling::Threads(GetMultithreading(), active / 4);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (threads > 1) num_threads(threads)
#endif
        for (long long tile = 0; tile < static_cast<long long>(tiles); ++tile)
        {
            size_t index = Expand(static_cast<size_t>(tile) * tileSize);
            const size_t end = std::min(active, (static_cast<size_t>(tile) + 1) * tileSize);
            for (size_t k = static_cast<size_t>(tile) * tileSize; k < end; ++k)
            {
                registerStorage(index) = (index & mask) == selected ? registerStorage(index) * norm : std::complex<double>(0., 0.);
                index = (((index | measuredMask) + 1) & ~measuredMask) | measuredValues;
            }
        }
        RememberMeasurement(qubit, outcome);
        return outcome;
    }

    size_t measuredMask = 0, measuredValues = 0;
    std::shared_ptr<const Eigen::VectorXcd> snapshot;
};

// statevector backend. Owns exactly one native QCSim implementation.
class QCSimStatevectorSimulator : public QCSimGateSimulator
{
  public:
    QCSimStatevectorSimulator()
    {
        configuration.SetConfiguration("method", "statevector");
    }

    const char *MethodName() const override
    {
        return "statevector";
    }
    friend class IndividualSimulator;

    struct SamplingProbability
    {
        const Eigen::VectorXcd *storage;

        double operator()(size_t row) const
        {
            return std::norm((*storage)[row]);
        }
    };

    using SamplingPlan = Utils::Sampling::Prepared<SamplingProbability>;

    SamplingPlan PrepareSampling(size_t shots) const
    {
        return SamplingPlan(size_t{1} << samplingSupportQubits, shots, SamplingProbability{&state->getRegisterStorage()}, SamplingOptions());
    }

  public:
    void Initialize() override
    {
        if (nrQubits != 0)
        {
            InitializeSamplingSeed();
            state = std::make_unique<SamplingQubitRegister>(nrQubits);
            samplingSupportQubits = savedSamplingSupportQubits = 0;
            SeedBackend(state.get(), samplingSeed);
            SetMultithreading(enableMultithreading);
            for (const auto &[key, value] : configuration.GetConfigMap())
                if (key != "method")
                    Configure(key.c_str(), value.c_str());
        }
    }

    void InitializeState(size_t num_qubits, std::vector<std::complex<double>> &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        Eigen::VectorXcd amplitudesEigen(Eigen::Map<Eigen::VectorXcd, Eigen::Unaligned>(amplitudes.data(), amplitudes.size()));
        state->setRegisterStorageFastNoNormalize(amplitudesEigen);
        UpdateSamplingSupport();
    }

#ifndef NO_QISKIT_AER
    void InitializeState(size_t num_qubits, AER::Vector<std::complex<double>> &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        Eigen::VectorXcd amplitudesEigen(Eigen::Map<Eigen::VectorXcd, Eigen::Unaligned>(amplitudes.data(), amplitudes.size()));
        state->setRegisterStorageFastNoNormalize(amplitudesEigen);
        UpdateSamplingSupport();
    }

#endif

    void InitializeState(size_t num_qubits, Eigen::VectorXcd &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        {
            state = std::make_unique<SamplingQubitRegister>(nrQubits, amplitudes);
            state->SetMultithreading(enableMultithreading);
            SeedBackend(state.get(), samplingSeed);
            UpdateSamplingSupport();
        }
    }

    void InitializeToBasisState(size_t num_qubits, Types::qubit_t basisState) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        state->setToBasisState(static_cast<size_t>(basisState));
        samplingSupportQubits = 0;
        for (size_t value = basisState; value; value >>= 1)
            ++samplingSupportQubits;
    }

    void Reset() override
    {
        if (state)
        {
            state->InvalidateMeasurements();
            state->Reset();
        }
        samplingSupportQubits = 0;
        upcomingGateIndex = 0;
    }

    void Configure(const char *key, const char *value) override
    {
        if (ConfigureSampling(key, value))
            return;
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
            SeedSampling(seed);
            if (state)
                SeedBackend(state.get(), seed);
            return;
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
        if (state)
            return 0;
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        state = nullptr;
        samplingSupportQubits = savedSamplingSupportQubits = 0;
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
            for (size_t qubit : qubits)
            {
                if (MeasureQubit(qubit))
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
            for (size_t q = 0; q < qubits.size(); ++q)
                if (MeasureQubit(qubits[q]))
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
                if (MeasureQubit(qubit))
                {
                    state->InvalidateMeasurements();
                    state->ApplyGate(xgate, static_cast<unsigned int>(qubit));
                }
        }
        Notify();
        NotifyObservers(qubits);
    }

    std::vector<std::complex<double>> GetStateVector() override
    {
        if (!nrQubits)
            return {};
        const auto count = TensorQueries::DenseSize(nrQubits);
        const auto values = state->getRegisterStorage();
        if (static_cast<size_t>(values.size()) != count)
            throw std::runtime_error("QCSim statevector query returned an invalid size");
        return {values.data(), values.data() + values.size()};
    }

    double Probability(Types::qubit_t outcome) override
    {
        return state->getBasisStateProbability(static_cast<unsigned int>(outcome));
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        return state->getBasisStateAmplitude(static_cast<unsigned int>(outcome));
    }

    std::vector<double> AllProbabilities() override
    {
        const Eigen::VectorXcd probs = state->getRegisterStorage().cwiseAbs2();
        std::vector<double> result(probs.size());
        for (int i = 0; i < probs.size(); ++i)
            result[i] = probs[i].real();
        return result;
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        std::vector<double> result(qubits.size());
        {
            const Eigen::VectorXcd &reg = state->getRegisterStorage();
            for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
                result[i] = std::norm(reg[qubits[i]]);
        }
        return result;
    }

    std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (!LegacySampling())
            return SamplePrepared<false>(qubits, shots);
        if (qubits.empty() || shots == 0)
            return {};
        if (qubits.size() > sizeof(size_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the size_t type, the outcome will be undefined"
                      << std::endl;
        std::unordered_map<Types::qubit_t, Types::qubit_t> result;
        DontNotify();
        {
            if (shots > 1)
            {
                const auto &statev = state->getRegisterStorage();
                const Utils::Alias alias(statev);
                for (size_t shot = 0; shot < shots; ++shot)
                {
                    const double prob = 1. - uniformZeroOne(rng);
                    const size_t measRaw = alias.Sample(prob);
                    size_t meas = 0;
                    size_t mask = 1ULL;
                    for (auto q : qubits)
                    {
                        const size_t qubitMask = 1ULL << q;
                        if ((measRaw & qubitMask) != 0)
                            meas |= mask;
                        mask <<= 1ULL;
                    }
                    ++result[meas];
                }
            }
            else
            {
                for (size_t shot = 0; shot < shots; ++shot)
                {
                    const size_t measRaw = MeasureNoCollapse();
                    size_t meas = 0;
                    size_t mask = 1ULL;
                    for (auto q : qubits)
                    {
                        const size_t qubitMask = 1ULL << q;
                        if ((measRaw & qubitMask) != 0)
                            meas |= mask;
                        mask <<= 1ULL;
                    }
                    ++result[meas];
                }
            }
        }
        Notify();
        NotifyObservers(qubits);
        return result;
    }

    std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (!LegacySampling())
            return SamplePrepared<true>(qubits, shots);
        if (qubits.empty() || shots == 0)
            return {};
        std::unordered_map<std::vector<bool>, Types::qubit_t> result;
        DontNotify();
        {
            if (shots > 1)
            {
                const auto &statev = state->getRegisterStorage();
                const Utils::Alias alias(statev);
                std::vector<bool> meas(qubits.size());
                for (size_t shot = 0; shot < shots; ++shot)
                {
                    const double prob = 1. - uniformZeroOne(rng);
                    const size_t measRaw = alias.Sample(prob);
                    for (size_t i = 0; i < qubits.size(); ++i)
                        meas[i] = ((measRaw >> qubits[i]) & 1) != 0;
                    ++result[meas];
                }
            }
            else
            {
                for (size_t shot = 0; shot < shots; ++shot)
                {
                    const auto measRaw = MeasureNoCollapseMany();
                    std::vector<bool> meas(qubits.size(), false);
                    for (size_t i = 0; i < qubits.size(); ++i)
                        if (measRaw[qubits[i]])
                            meas[i] = true;
                    ++result[meas];
                }
            }
        }
        Notify();
        NotifyObservers(qubits);
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
        return state->ExpectationValue(pauliStringVec).real();
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kStatevector;
    }

    void SaveState() override
    {
        state->SaveState();
        savedSamplingSupportQubits = samplingSupportQubits;
    }

    void RestoreState() override
    {
        state->InvalidateMeasurements();
        state->RestoreState();
        samplingSupportQubits = std::max(samplingSupportQubits, savedSamplingSupportQubits);
    }

    void SetMultithreading(bool multithreading = true) override
    {
        enableMultithreading = multithreading;
        if (state)
            state->SetMultithreading(multithreading);
    }

    Types::qubit_t MeasureNoCollapse() override
    {
        if (GetNumberOfQubits() > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;
        return state->MeasureNoCollapse();
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        auto state = MeasureNoCollapse();
        std::vector<bool> res(nrQubits);
        for (size_t i = 0; i < nrQubits; ++i)
            res[i] = ((state >> i) & 1) == 1;
        return res;
    }

    void ApplyGenericOneQubitGate(Types::qubit_t qubit, const Eigen::Matrix2cd &gate) override
    {
        samplingSupportQubits = std::max(samplingSupportQubits, size_t(qubit) + 1);
        const QC::Gates::AppliedGate<> agate(gate, qubit);
        ApplyNativeGate(agate);
        NotifyObservers({qubit});
    }

    void ApplyGenericTwoQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, const Eigen::Matrix4cd &gate) override
    {
        samplingSupportQubits = std::max({samplingSupportQubits, size_t(qubit0) + 1, size_t(qubit1) + 1});
        const QC::Gates::AppliedGate<> agate(gate, qubit0, qubit1);
        ApplyNativeGate(agate);
        NotifyObservers({qubit0, qubit1});
    }

    void ApplyGenericThreeQubitGate(Types::qubit_t q0, Types::qubit_t q1, Types::qubit_t q2, const Matrix8cd &gate) override
    {
        samplingSupportQubits = std::max({samplingSupportQubits, size_t(q0) + 1, size_t(q1) + 1, size_t(q2) + 1});
        const QC::Gates::AppliedGate<> applied(gate, q0, q1, q2);
        ApplyNativeGate(applied);
        NotifyObservers({q0, q1, q2});
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        pgate.SetPhaseShift(lambda);
        ApplyNativeGate(pgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        ApplyNativeGate(xgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        ApplyNativeGate(ygate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        ApplyNativeGate(zgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        ApplyNativeGate(h, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        ApplyNativeGate(sgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        ApplyNativeGate(sdggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        ApplyNativeGate(tgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        ApplyNativeGate(tdggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        ApplyNativeGate(sxgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        ApplyNativeGate(sxdaggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        ApplyNativeGate(k, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        rxgate.SetTheta(theta);
        ApplyNativeGate(rxgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        rygate.SetTheta(theta);
        ApplyNativeGate(rygate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        rzgate.SetTheta(theta);
        ApplyNativeGate(rzgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        ugate.SetParams(theta, phi, lambda, gamma);
        ApplyNativeGate(ugate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(cxgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(cygate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(czgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        cpgate.SetPhaseShift(lambda);
        ApplyNativeGate(cpgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crxgate.SetTheta(theta);
        ApplyNativeGate(crxgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crygate.SetTheta(theta);
        ApplyNativeGate(crygate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crzgate.SetTheta(theta);
        ApplyNativeGate(crzgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(ch, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(csx, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(csxdag, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        ApplyNativeGate(swapgate, static_cast<unsigned int>(qubit1), static_cast<unsigned int>(qubit0));
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        ApplyNativeGate(ccxgate, static_cast<unsigned int>(qubit2), static_cast<unsigned int>(qubit1), static_cast<unsigned int>(qubit0));
        NotifyObservers({qubit2, qubit1, qubit0});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        ApplyNativeGate(cswapgate, static_cast<unsigned int>(qubit1), static_cast<unsigned int>(qubit0), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({qubit1, qubit0, ctrl_qubit});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        cugate.SetParams(theta, phi, lambda, gamma);
        ApplyNativeGate(cugate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        return CloneForExecution(NextCloneSeed());
    }

    std::unique_ptr<ISimulator> CloneForExecution(uint64_t seed) override
    {
        auto cloned = std::make_unique<QCSimStatevectorSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->enableMultithreading = enableMultithreading;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        cloned->samplingSupportQubits = samplingSupportQubits;
        cloned->savedSamplingSupportQubits = savedSamplingSupportQubits;
        if (state)
            cloned->state = std::make_unique<SamplingQubitRegister>(*state);
        for (const auto &[key, value] : configuration.GetConfigMap())
            cloned->Configure(key.c_str(), value.c_str());
        cloned->SetSeed(seed);
        return cloned;
    }

  protected:
    std::unique_ptr<SamplingQubitRegister> state;

  private:
    size_t MeasureQubit(size_t qubit)
    {
        if (!LegacySampling())
            return state->MeasureQubitReproducible(qubit);
        state->InvalidateMeasurements();
        return state->MeasureQubit(qubit);
    }

    // Internal state replacement, deliberately separate from public
    // initialization: preserve native/batch/readout RNGs and fork ordinals.
    void ReplaceState(size_t qubits, Eigen::VectorXcd &amplitudes)
    {
        if (!state || !qubits || qubits >= std::numeric_limits<size_t>::digits || static_cast<size_t>(amplitudes.size()) != (size_t{1} << qubits))
            throw std::invalid_argument("Invalid composite replacement state");
        state->ReplaceAmplitudes(qubits, amplitudes);
        nrQubits = qubits;
        savedSamplingSupportQubits = 0;
        upcomingGateIndex = 0;
        upcomingGates.clear();
        UpdateSamplingSupport();
    }

    template <bool Many> Utils::Sampling::Counts<Many> SamplePrepared(const Types::qubits_vector &qubits, size_t shots)
    {
        if (qubits.empty() || !shots)
            return {};
        Utils::Sampling::ValidateQubits(qubits, nrQubits, Many);
        if (shots == 1)
        {
            // Preserve the native one-shot fast path and its RNG stream. Fix
            // its summation path too: native parallel reductions depend on P.
            const size_t row = [&] {
                struct RestoreThreading
                {
                    QC::QubitRegister<> &state;
                    bool enabled;
                    ~RestoreThreading()
                    {
                        state.SetMultithreading(enabled);
                    }
                } restore{*state, enableMultithreading};
                state->SetMultithreading(false);
                return state->MeasureNoCollapse();
            }();
            auto result = Utils::Sampling::Single<Many>(qubits.size(), [&](size_t bit) { return ((row >> qubits[bit]) & 1) != 0; });
            NotifyObservers(qubits);
            return result;
        }
        const auto plan = PrepareSampling(shots);
        auto result = Utils::Sampling::Count<Many>(
            plan, shots, qubits.size(), [&](size_t row, size_t bit) { return ((row >> qubits[bit]) & 1) != 0; },
            [&] { return Utils::RandomStream::Uniform(rng); });
        NotifyObservers(qubits);
        return result;
    }

    void UpdateSamplingSupport()
    {
        // Imports already establish a new state. Inspect trailing entries once
        // here, instead of scanning the whole register on every sampling call.
        // This is exact support metadata: tiny/invalid nonzero entries are kept.
        const auto &storage = state->getRegisterStorage();
        size_t end = static_cast<size_t>(storage.size());
        while (end > 1 && storage[end - 1] == std::complex<double>{})
            --end;
        samplingSupportQubits = 0;
        for (size_t last = end ? end - 1 : 0; last; last >>= 1)
            ++samplingSupportQubits;
    }

    // An exact upper bound on occupied basis bits, never inferred from a probe.
    // Initialized/imported states set the bound; gates can only widen it.
    size_t samplingSupportQubits = 0, savedSamplingSupportQubits = 0;

    template <class Gate, class... Qubits> void ApplyNativeGate(const Gate &gate, Qubits... qubits)
    {
        state->InvalidateMeasurements();

        // The native parallel dispatcher's one-thread fallback can contract
        // complex products differently from its serial/actual-team kernels.
        // Use the serial dispatcher when no team will run, including builds
        // without OpenMP. Keep the public permission flag unchanged.
        struct RestoreThreading
        {
            SamplingQubitRegister &state;
            bool enabled;

            ~RestoreThreading()
            {
                state.SetMultithreading(enabled);
            }
        } restore{*state, state->GetMultithreading()};

        bool hasTeam = false;
#ifdef _OPENMP
        hasTeam = !omp_in_parallel() && omp_get_max_threads() > 1;
#endif
        if (!LegacySampling() && !hasTeam)
            state->SetMultithreading(false);
        state->ApplyGate(gate, qubits...);
        ((samplingSupportQubits = std::max(samplingSupportQubits, size_t(qubits) + 1)), ...);
    }
};
} // namespace Simulators::Private
#endif
