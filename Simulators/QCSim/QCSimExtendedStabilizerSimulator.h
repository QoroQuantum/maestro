#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "ImmediateQCSimSimulator.h"
#include "QCSimExtendedStabilizer.h"
#define _USE_MATH_DEFINES
#include <math.h>

#include <optional>
#include <type_traits>

namespace Simulators::Private
{
// extended_stabilizer backend. Owns exactly one native QCSim implementation.
class QCSimExtendedStabilizerSimulator : public ImmediateQCSimSimulator
{
  public:
    QCSimExtendedStabilizerSimulator()
    {
        configuration.SetConfiguration("method", "extended_stabilizer");
    }

    const char *MethodName() const override
    {
        return "extended_stabilizer";
    }

    void Initialize() override
    {
        if (nrQubits != 0)
        {
            {
                extendedStabilizer = std::make_unique<Simulators::QCSimExtendedStabilizer>(nrQubits);
            }
            SetMultithreading(enableMultithreading);
            for (const auto &[key, value] : configuration.GetConfigMap())
                if (key != "method")
                    Configure(key.c_str(), value.c_str());
        }
    }

    void Reset() override
    {
        if (extendedStabilizer)
            extendedStabilizer->Reset(nrQubits);
        upcomingGateIndex = 0;
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
            if (extendedStabilizer)
                SeedBackend(extendedStabilizer.get(), seed);
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
        if (extendedStabilizer)
            return 0;
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        extendedStabilizer = nullptr;
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
                if (extendedStabilizer->Measure(qubit))
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
                if (extendedStabilizer->Measure(qubits[q]))
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
                if (extendedStabilizer->Measure(qubit))
                    extendedStabilizer->ApplyX(qubit);
        }
        Notify();
        NotifyObservers(qubits);
    }

    double Probability(Types::qubit_t outcome) override
    {
        return ExtendedStabilizerBasisProbability(outcome);
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        throw std::runtime_error("QCSimState::Amplitude: Amplitudes are not exposed by the extended "
                                 "stabilizer simulator.");
    }

    std::vector<double> AllProbabilities() override
    {
        const size_t nrBasisStates = CheckedBasisStateCountForQueries();
        std::vector<double> result(nrBasisStates);
        for (size_t i = 0; i < nrBasisStates; ++i)
            result[i] = ExtendedStabilizerBasisProbability(i);
        return result;
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        std::vector<double> result(qubits.size());
        {
            for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
                result[i] = ExtendedStabilizerBasisProbability(qubits[i]);
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
        if (TryMarginalCounts<false>(qubits, shots, result))
        {
            NotifyObservers(qubits);
            return result;
        }
        DontNotify();
        {
            const std::vector<size_t> selected(qubits.begin(), qubits.end());
            auto sampler = extendedStabilizer->CloneWithSeed(rng());
            for (const auto &item : sampler->SampleCounts(selected, shots))
                result[item.first] += item.second;
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
        if (TryMarginalCounts<true>(qubits, shots, result))
        {
            NotifyObservers(qubits);
            return result;
        }
        DontNotify();
        {
            const std::vector<size_t> selected(qubits.begin(), qubits.end());
            auto sampler = extendedStabilizer->CloneWithSeed(rng());
            for (const auto &item : sampler->SampleCountsMany(selected, shots))
                result[item.first] += item.second;
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
        return extendedStabilizer->ExpectationValue(pauliString);
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kExtendedStabilizer;
    }

    void SaveState() override
    {
        extendedStabilizer->SaveState();
    }

    void RestoreState() override
    {
        extendedStabilizer->RestoreState();
    }

    void SetMultithreading(bool multithreading = true) override
    {
        enableMultithreading = multithreading;
        if (extendedStabilizer)
            extendedStabilizer->SetMultithreading(multithreading);
    }

    Types::qubit_t MeasureNoCollapse() override
    {
        if (GetNumberOfQubits() > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;
        {
            auto sampler = extendedStabilizer->CloneWithSeed(rng());
            Types::qubit_t result = 0;
            for (size_t qubit = 0; qubit < nrQubits; ++qubit)
                if (sampler->Measure(qubit))
                    result |= Types::BitAt(qubit);
            return result;
        }
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        auto sampler = extendedStabilizer->CloneWithSeed(rng());
        std::vector<bool> res(nrQubits);
        for (size_t i = 0; i < nrQubits; ++i)
            res[i] = sampler->Measure(i);
        return res;
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        extendedStabilizer->ApplyP(qubit, lambda);
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        extendedStabilizer->ApplyX(qubit);
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        extendedStabilizer->ApplyY(qubit);
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        extendedStabilizer->ApplyZ(qubit);
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        extendedStabilizer->ApplyH(qubit);
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        extendedStabilizer->ApplyS(qubit);
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        extendedStabilizer->ApplySDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        extendedStabilizer->ApplyT(qubit);
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        extendedStabilizer->ApplyTDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        extendedStabilizer->ApplySX(qubit);
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        extendedStabilizer->ApplySXDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        extendedStabilizer->ApplyK(qubit);
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        extendedStabilizer->ApplyRX(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        extendedStabilizer->ApplyRY(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        extendedStabilizer->ApplyRZ(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        extendedStabilizer->ApplyU(qubit, theta, phi, lambda, gamma);
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        extendedStabilizer->ApplyCX(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        extendedStabilizer->ApplyCY(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        extendedStabilizer->ApplyCZ(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        extendedStabilizer->ApplyCP(ctrl_qubit, tgt_qubit, lambda);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        extendedStabilizer->ApplyCRX(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        extendedStabilizer->ApplyCRY(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        extendedStabilizer->ApplyCRZ(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        extendedStabilizer->ApplyCH(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        extendedStabilizer->ApplyCSX(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        extendedStabilizer->ApplyCSXDAG(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        extendedStabilizer->ApplySWAP(qubit0, qubit1);
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        extendedStabilizer->ApplyCCX(qubit0, qubit1, qubit2);
        NotifyObservers({qubit2, qubit1, qubit0});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        extendedStabilizer->ApplyCSwap(ctrl_qubit, qubit0, qubit1);
        NotifyObservers({qubit1, qubit0, ctrl_qubit});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        extendedStabilizer->ApplyCU(ctrl_qubit, tgt_qubit, theta, phi, lambda, gamma);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        auto cloned = std::make_unique<QCSimExtendedStabilizerSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->enableMultithreading = enableMultithreading;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        if (extendedStabilizer)
            cloned->extendedStabilizer = configuration.IsSet("seed") ? extendedStabilizer->Clone() : extendedStabilizer->CloneWithSeed(rng());
        for (const auto &[key, value] : configuration.GetConfigMap())
            cloned->Configure(key.c_str(), value.c_str());
        if (configuration.IsSet("seed"))
            cloned->SetSeed(DeriveSeed(std::stoull(configuration.GetConfiguration("seed")), nextSeedStream++));
        return cloned;
    }

  protected:
    template <bool Many> bool TryMarginalCounts(const Types::qubits_vector &qubits, size_t shots, Utils::Sampling::Counts<Many> &counts)
    {
        using namespace Utils::Sampling;
        if (!PreferExtendedMarginalCounts(shots, 1) || !AllowMultinomialCounts())
            return false;
        ValidateQubits(qubits, nrQubits, Many);
        // A Z parity on a subset is a Fourier coefficient of the joint
        // distribution of those measured bits. Only the measured width enters
        // the transform. Keep both the number of frame queries and scratch bounded.
        // The output mapping is indexed by register qubit, so repeating a
        // qubit in the request allocates nothing per requested bit.
        std::vector<size_t> selected;
        std::vector<uint8_t> position(nrQubits, 0);
        for (const auto q : qubits)
            if (std::find(selected.begin(), selected.end(), q) == selected.end())
            {
                if (selected.size() == maxExtendedMarginalBits)
                    return false;
                position[q] = static_cast<uint8_t>(selected.size());
                selected.push_back(q);
            }
        // Each native frame shot may restore and collapse a private frame.
        // The measured crossover is earlier than for an existing alias table.
        if (!PreferExtendedMarginalCounts(shots, selected.size()))
            return false;
        // Pruning after a non-diagonal measurement can change the conditional
        // state for the next measured bit. Preserve that sequential behavior.
        // A single stabilizer component is measured without pruning.
        if (selected.size() > 1 && extendedStabilizer->GetApproximationPolicy().mode != QC::ExtendedStabilizerApproximationMode::Exact &&
            !SingleBasisComponent())
            return false;

        const auto weights = FrameMarginalWeights(selected);
        // FrameMarginalWeights already zeroes rounding residue. A materially
        // negative or nonfinite mass falls back before consuming RNG.
        if (std::any_of(weights.begin(), weights.end(), [](double p) { return !(p >= 0. && std::isfinite(p)); }))
            return false;
        auto options = SamplingOptions();
        options.family = Family::Multinomial;
        auto probability = [&weights](size_t row) { return weights[row]; };
        Prepared plan(weights.size(), shots, probability, options);
        counts = Count<Many>(
            plan, shots, qubits.size(), [&](size_t row, size_t bit) { return (row >> position[qubits[bit]]) & 1; },
            [&] { return Utils::RandomStream::Uniform(rng); });
        return true;
    }

    // One Clifford-basis component: an ordinary stabilizer state U|b>.
    bool SingleBasisComponent() const
    {
        const auto &frames = extendedStabilizer->GetFrames();
        return frames.size() == 1 && frames.front().GetFrameSize() == 1;
    }

    // Unnormalized joint weights of the selected bits, indexed by bit position.
    std::vector<double> FrameMarginalWeights(const std::vector<size_t> &selected) const
    {
        const size_t outcomes = size_t{1} << selected.size();
        if (SingleBasisComponent())
        {
            // One Clifford-basis component is uniform on an affine space.
            // QCSim enumerates its selected support directly in Gray-code order.
            std::vector<double> weights(outcomes, 0.);
            const auto &frame = extendedStabilizer->GetFrames().front();
            QC::Clifford::detail::BasisDistribution affine;
            affine.BuildFromInverse(frame.cliffordBasis.ZImages(), selected, frame.signs.LabelWords(0));
            affine.FillProbabilities(weights);
            return weights;
        }
        if (auto weights = DiagonalFrameWeights(selected))
            return std::move(*weights);
        auto weights = FrameParityMoments(selected);
        const double norm = weights[0];
        InverseParityTransform(weights);
        ClearRoundingResidue(weights, norm);
        return weights;
    }

    // When no selected qubit's Z image has an X part, every component U|b> has
    // a definite outcome: bit j is the image's sign xor parity(z_j & b). The
    // labels are distinct, as QCSim's own diagonal measurement assumes, so the
    // joint weights are sums of |amplitude|^2 by outcome. Positive sums keep
    // a tiny probability accurate; differences of parity moments over many
    // components would round it away. Costs one pass over the components.
    std::optional<std::vector<double>> DiagonalFrameWeights(const std::vector<size_t> &selected) const
    {
        const auto &frame = extendedStabilizer->GetFrames().front();
        using Image = std::decay_t<decltype(frame.cliffordBasis.ImageZ(0))>;
        std::vector<Image> images;
        images.reserve(selected.size());
        for (const auto q : selected)
        {
            images.push_back(frame.cliffordBasis.ImageZ(q));
            if (images.back().HasX())
                return std::nullopt;
        }
        std::vector<double> weights(size_t{1} << selected.size(), 0.);
        for (size_t component = 0; component < frame.GetFrameSize(); ++component)
        {
            const auto *label = frame.signs.LabelWords(component);
            size_t outcome = 0;
            for (size_t bit = 0; bit < images.size(); ++bit)
            {
                const auto &image = images[bit];
                std::decay_t<decltype(*label)> parity = 0;
                for (size_t word = 0; word < image.Words(); ++word)
                    parity ^= image.Z.words[word] & label[word];
                if (bool(image.PhaseSign) != ((QC::Clifford::detail::Popcount(parity) & 1U) != 0))
                    outcome |= size_t{1} << bit;
            }
            weights[outcome] += std::norm(frame.amplitudes[component]);
        }
        return weights;
    }

    // An outcome of zero probability can come out of the transform as a few
    // ulps below zero. The tolerance is 1e-12 in probability units; larger
    // negative weights are kept so that the caller falls back.
    static void ClearRoundingResidue(std::vector<double> &weights, double norm)
    {
        const double tolerance = 1e-12 * norm * static_cast<double>(weights.size());
        for (double &weight : weights)
            if (weight < 0. && weight >= -tolerance)
                weight = 0.;
    }

    // <psi|Z_S|psi> for every subset S of the selected bits (bit b of the index
    // selects selected[b]). The empty subset holds the squared norm, which the
    // native measurement also divides by, rather than assuming exactly one.
    std::vector<double> FrameParityMoments(const std::vector<size_t> &selected) const
    {
        const size_t outcomes = size_t{1} << selected.size();
        std::vector<double> moments(outcomes, 0.);
        for (const auto &amplitude : extendedStabilizer->GetFrames().front().amplitudes)
            moments[0] += std::norm(amplitude);
        std::string parity(*std::max_element(selected.begin(), selected.end()) + 1, 'I');
        for (size_t mask = 1; mask < outcomes; ++mask)
        {
            for (size_t bit = 0; bit < selected.size(); ++bit)
                parity[selected[bit]] = (mask >> bit) & 1 ? 'Z' : 'I';
            moments[mask] = extendedStabilizer->ExpectationValue(parity);
        }
        return moments;
    }

    // Inverse Walsh-Hadamard transform. The shared normalization factor
    // 1/outcomes cancels in the count sampler.
    static void InverseParityTransform(std::vector<double> &weights)
    {
        const size_t outcomes = weights.size();
        for (size_t stride = 1; stride < outcomes; stride *= 2)
            for (size_t block = 0; block < outcomes; block += 2 * stride)
                for (size_t i = block; i < block + stride; ++i)
                {
                    const double a = weights[i], b = weights[i + stride];
                    weights[i] = a + b;
                    weights[i + stride] = a - b;
                }
    }

    double ExtendedStabilizerBasisProbability(Types::qubit_t outcome) const
    {
        const size_t nrBasisStates = CheckedBasisStateCountForQueries();
        if (outcome >= nrBasisStates)
            return 0.0;
        double probability = 0.0;
        std::string pauliString(nrQubits, 'I');
        for (size_t mask = 0; mask < nrBasisStates; ++mask)
        {
            double sign = 1.0;
            size_t parityBits = mask & static_cast<size_t>(outcome);
            while (parityBits != 0)
            {
                sign = -sign;
                parityBits &= parityBits - 1;
            }
            for (size_t qubit = 0; qubit < nrQubits; ++qubit)
                pauliString[qubit] = ((mask >> qubit) & 1ULL) == 0 ? 'I' : 'Z';
            probability += sign * extendedStabilizer->ExpectationValue(pauliString);
        }
        probability /= static_cast<double>(nrBasisStates);
        return std::max(0.0, std::min(1.0, probability));
    }

    std::unique_ptr<Simulators::QCSimExtendedStabilizer> extendedStabilizer;
};
} // namespace Simulators::Private
#endif
