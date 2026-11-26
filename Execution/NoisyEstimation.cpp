#include "NoisyEstimation.h"

#include "../Simulators/PathIntegral/PathIntegralSimulator.h"
#include "../Utils/RandomStream.h"
#include "../Utils/Threads/RealizationWorkers.h"
#include "CircuitHelpers.h"
#include "PauliTransfer.h"
#include "TerminalChannels.h"
#include "TerminalExpectation.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace MaestroExecution
{
namespace
{
using Channel = Simulators::QuantumChannel;
using PauliTerms = std::map<std::string, double>;
namespace Transfer = QC::PauliDetail;
constexpr char PauliLetters[] = "IXZY"; // QCSim interleaves the X and Z bits.

struct TerminalFactor
{
    Types::qubits_vector targets;
    Transfer::CompiledChannel transfer;
};

Eigen::MatrixXcd PauliBasis(unsigned qubits)
{
    const unsigned dim = 1U << qubits, count = dim * dim;
    Eigen::MatrixXcd basis(count, count);
    for (unsigned p = 0; p < count; ++p)
    {
        const auto matrix = Transfer::LabelMatrix(p, qubits);
        for (unsigned row = 0; row < dim; ++row)
            for (unsigned col = 0; col < dim; ++col)
                basis(row + dim * col, p) = matrix[row * dim + col];
    }
    return basis;
}

std::optional<Transfer::CompiledChannel> CompileTerminalFactor(const Channel &channel, size_t index, bool whole)
{
    const auto &factor = channel.GetFactors()[index];
    const auto width = static_cast<unsigned>(whole ? channel.GetNumberOfQubits() : factor.qubits.size());
    constexpr double rounding = 32 * std::numeric_limits<double>::epsilon();
    if (factor.kind == Channel::FactorKind::kThermal)
        return Transfer::CompileRelaxation(factor.thermal.coherence, 1.0 - factor.thermal.decay,
                                           factor.thermal.decay * (1.0 - 2.0 * factor.thermal.excitedStatePopulation));
    if (factor.kind == Channel::FactorKind::kPauli && factor.operators.empty() && !whole)
    {
        std::array<double, 16> probabilities{};
        // Classifying phased Kraus operators or composing channels can leave the
        // total a few ulps off one (even p(X) > 1). Normalize roundoff only.
        const double total = std::accumulate(factor.probabilities.begin(), factor.probabilities.end(), 0.);
        if (!(std::abs(total - 1.) <= rounding))
            return std::nullopt;
        std::transform(factor.probabilities.begin(), factor.probabilities.end(), probabilities.begin(), [total](double p) { return p / total; });
        Transfer::CompiledChannel transfer;
        transfer.eigenvalues = Transfer::PauliChannelEigenvalues(probabilities, width);
        return transfer;
    }
    // Change the basis of the adjoint superoperator, keeping all coefficients
    // down to working precision. The Pauli propagator's Kraus compiler snaps
    // small values to zero to make its propagation sparse; a tiny rotation or
    // a rare branch is observable here even on a single qubit.
    static const Eigen::MatrixXcd one = PauliBasis(1), two = PauliBasis(2);
    const auto &basis = width == 1 ? one : two;
    const Channel &source = whole ? channel : channel.FactorChannel(index);
    const auto &operation = source.Superoperator();
    const Eigen::MatrixXd coefficients = (basis.adjoint() * operation.adjoint() * basis).real() / (1U << width);
    // A rounded Kraus set accepted by the public channel validator can need
    // state-dependent renormalization. In that case leave it on the trajectory.
    if (std::abs(coefficients(0, 0) - 1.) > rounding || coefficients.col(0).tail(coefficients.rows() - 1).cwiseAbs().maxCoeff() > rounding)
        return std::nullopt;
    Transfer::ChannelTransfer transfer{};
    transfer[0][0] = 1.;
    for (Eigen::Index input = 1; input < coefficients.cols(); ++input)
        for (Eigen::Index output = 0; output < coefficients.rows(); ++output)
            transfer[input][output] = coefficients(output, input);
    return Transfer::CompileChannel(transfer, width);
}

// Pull back one local channel factor. Keep tiny nonzero terms. If the sum
// grows too wide, abandon the transformation and use the original circuit.
bool PullBack(PauliTerms &terms, const TerminalFactor &factor, size_t limit)
{
    PauliTerms next;
    for (const auto &[word, coefficient] : terms)
    {
        unsigned input = 0;
        for (size_t j = 0; j < factor.targets.size(); ++j)
        {
            const auto q = factor.targets[j];
            if (q < word.size())
                input |= static_cast<unsigned>(std::string_view(PauliLetters).find(word[q])) << (2 * j);
        }
        if (factor.transfer.diagonal || input == 0)
        {
            const double value = input == 0 ? coefficient : coefficient * factor.transfer.eigenvalues[input];
            if (value != 0.)
                next[word] += value;
            continue;
        }
        const auto &table = *factor.transfer.table;
        for (auto pos = table.offsets[input]; pos < table.offsets[input + 1]; ++pos)
        {
            const auto &entry = table.entries[pos];
            std::string updated = word;
            for (size_t j = 0; j < factor.targets.size(); ++j)
            {
                const auto q = factor.targets[j];
                const char axis = PauliLetters[(entry.pauli >> (2 * j)) & 3];
                if (q >= updated.size())
                {
                    if (axis == 'I')
                        continue;
                    updated.resize(size_t(q) + 1, 'I');
                }
                updated[q] = axis;
            }
            next[updated] += coefficient * entry.coefficient;
            if (next.size() > limit)
                return false;
        }
    }
    terms = std::move(next);
    return true;
}
} // namespace

std::optional<TerminalExpectation> PrepareTerminalExpectation(const RecordCircuitPtr &circuit, const std::vector<std::string> &observables,
                                                              const SimulatorConfig &config)
{
    if (config.terminal_channel_policy != "auto" || !UsesChannelSamplingStatevector(config) || observables.empty() || !circuit)
        return std::nullopt;
    auto tail = SplitTerminalChannels(circuit);
    if (!tail.prefix || tail.channels.empty())
        return std::nullopt;
    if (std::none_of(tail.channels.begin(), tail.channels.end(), [](const auto &op) { return !op->IsDeterministic(); }))
        return std::nullopt;

    // An exact adjoint expansion of high-weight Z observables under damping
    // can double at each qubit. Bound work by switching back to trajectories;
    // short diagonal or disjoint channels never grow the sum.
    TerminalExpectation plan;
    plan.stochastic_prefix = HasStochasticEvolution(tail.prefix, false);
    const bool clusters_sampled_cones =
        plan.stochastic_prefix && config.allow_reset_marginalization && config.enable_causal_cone_reduction && config.enable_observable_clustering;
    std::vector<TerminalFactor> factors;
    std::unordered_map<const Channel::Factor *, Transfer::CompiledChannel> transfers;
    for (auto op = tail.channels.rbegin(); op != tail.channels.rend(); ++op)
    {
        const auto targets = (*op)->AffectedQubits();
        const auto &channel = (*op)->GetChannel();
        const auto &parts = channel.GetFactors();
        // A placed raw Kraus set can keep a narrower factor, or one with reversed targets.
        // Its approximate classification can hide a small rotation. Use its original map
        // and original target order, including the spectator, for exact expectations.
        const bool whole = parts.size() == 1 && parts.front().kind != Channel::FactorKind::kThermal && channel.GetPauliProbabilities().empty();
        for (size_t j = parts.size(); j-- > 0;)
        {
            TerminalFactor local;
            if (whole)
                local.targets = targets;
            else
                for (const auto index : parts[j].qubits)
                    local.targets.push_back(targets[index]);
            const auto [entry, inserted] = transfers.try_emplace(&parts[j]);
            if (inserted)
            {
                auto transfer = CompileTerminalFactor(channel, j, whole);
                if (!transfer)
                    return std::nullopt;
                entry->second = std::move(*transfer);
            }
            local.transfer = entry->second;
            factors.push_back(std::move(local));
        }
    }
    plan.original_width = std::max<size_t>(1, circuit->GetMaxQubitIndex() + 1);
    const size_t circuit_width = plan.original_width;
    plan.prefix = std::move(tail.prefix);
    // On wide registers each additional Pauli can require a full statevector scan.
    // Replaying a deep deterministic prefix on every trajectory is also expensive:
    // allow one extra scan per preparation gate, up to 512 terms per expanded row.
    constexpr size_t small_width = 16, wide_floor = 128, wide_ceiling = 512;
    size_t wide_limit = wide_floor;
    const auto wide_word = [&](const std::string &word) { return word.size() > small_width; };
    if (!plan.stochastic_prefix && (circuit_width > small_width || std::any_of(observables.begin(), observables.end(), wide_word)))
    {
        const auto &ops = plan.prefix->GetOperations();
        const size_t gates = std::count_if(ops.begin(), ops.end(), [](const auto &op) { return op->GetType() == Circuits::OperationType::kGate; });
        wide_limit += std::min(gates, wide_ceiling - wide_floor);
    }
    size_t remaining_wide_scans = wide_limit;
    std::map<std::string, size_t> unique;
    for (auto observable : observables)
    {
        if (observable.find_first_not_of("IXZYixzy") != std::string::npos)
            return std::nullopt;
        std::transform(observable.begin(), observable.end(), observable.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        plan.original_width = std::max(plan.original_width, observable.size());
        const bool wide = std::max(circuit_width, observable.size()) > small_width;
        const size_t deterministic_limit = wide ? wide_limit : 256;
        const size_t limit = plan.stochastic_prefix ? 4 : deterministic_limit;
        PauliTerms terms{{observable, 1.}};
        bool fits = std::all_of(factors.begin(), factors.end(), [&](const auto &factor) { return PullBack(terms, factor, limit); });
        if (fits && wide && terms.size() > wide_floor)
        {
            // Share the larger budget across rows. A word already evaluated for
            // another observable costs no additional statevector scan.
            const size_t scans = std::count_if(terms.begin(), terms.end(), [&](const auto &term) { return term.second != 0. && !unique.count(term.first); });
            fits = scans <= remaining_wide_scans;
            if (fits)
                remaining_wide_scans -= scans;
        }
        // Marginalization allows the planner to run separate sampled cones.
        // Several nonconstant terms in a row would need a common trajectory
        // for their covariance. Forcing all requested cones into one group
        // could require an exponentially larger state than the caller chose.
        auto &row = plan.rows.emplace_back();
        if (!fits || (clusters_sampled_cones && std::count_if(terms.begin(), terms.end(), [](const auto &term) {
                                                    return term.second != 0. && term.first.find_first_not_of('I') != std::string::npos;
                                                }) > 1))
        {
            plan.sampled_indices.push_back(plan.rows.size() - 1);
            plan.sampled_observables.push_back(std::move(observable));
            continue;
        }
        for (const auto &[word, coefficient] : terms)
        {
            if (coefficient == 0.)
                continue;
            const auto [found, added] = unique.emplace(word, plan.prefix_observables.size());
            if (added)
                plan.prefix_observables.push_back(word);
            row.emplace_back(found->second, coefficient);
        }
    }
    if (plan.sampled_indices.size() == observables.size())
        return std::nullopt;
    if (!plan.sampled_indices.empty())
    {
        plan.full_circuit = circuit;
        // The full shot must see the same stochastic operation order, physical register and
        // observable group as a standalone prefix shot. Composites can reschedule channels,
        // and independently marginalized cones can have different wires or branch streams.
        if (plan.stochastic_prefix && !HasCandidateList(config) && !config.allow_reset_marginalization &&
            (config.simulator_type == Simulators::SimulatorType::kQCSim
#ifndef NO_QISKIT_AER
             || config.simulator_type == Simulators::SimulatorType::kQiskitAer
#endif
             ) &&
            plan.prefix->HasDrawnChannels() && plan.prefix->GetQubits().size() == plan.original_width)
        {
            const auto &operations = circuit->GetOperations();
            auto end = operations.end();
            while (end != operations.begin() && (*(end - 1))->GetType() == Circuits::OperationType::kMeasurement)
                --end;
            const auto first = end - tail.channels.size();
            if (std::equal(first, end, tail.channels.begin()) &&
                std::none_of(tail.channels.begin(), tail.channels.end(), [](const auto &channel) { return channel->IsDeterministic(); }))
                plan.prefix_read_position = static_cast<size_t>(first - operations.begin());
        }
    }
    return plan;
}

std::vector<double> TerminalExpectation::Combine(const std::vector<double> &prefix_values, const std::vector<double> &sampled_values, bool bounds) const
{
    std::vector<double> values(rows.size(), 0.);
    for (size_t i = 0; i < rows.size(); ++i)
        for (const auto &[index, weight] : rows[i])
            values[i] += (bounds ? std::abs(weight) : weight) * prefix_values.at(index);
    for (size_t j = 0; j < sampled_values.size(); ++j)
        values[sampled_indices.at(j)] = sampled_values[j];
    return values;
}

namespace
{
using ChannelOp = Circuits::QuantumChannelOperation<double>;

// Keep the original trajectory's operations, including the channel draw itself. Read the
// folded Paulis just before its terminal layer. Clones created by the host share only the
// scalar readout for this realization. The backend can decline if observing
// the prefix would change the conditional branch order of the full shot.
class ReadPrefixBeforeChannel final : public ChannelOp
{
  public:
    ReadPrefixBeforeChannel(const std::shared_ptr<ChannelOp> &channel, const std::vector<std::string> &words, size_t width, Simulators::SimulatorType backend,
                            std::vector<double> &values)
        : ChannelOp(*channel), channel_(channel), words_(words), width_(width), backend_(backend), values_(values)
    {
    }

    void Execute(const std::shared_ptr<Simulators::ISimulator> &sim, Circuits::OperationState &state) const override
    {
        if (sim->GetNumberOfQubits() == width_ && sim->GetType() == backend_ && sim->GetSimulationType() == Simulators::SimulationType::kStatevector &&
            sim->TryPrefixExpectations(words_, values_))
        {
            for (size_t i = 0; i < words_.size(); ++i)
                if (words_[i].find_first_not_of('I') == std::string::npos)
                    values_[i] = 1.; // PrepareEstimation resolves identity exactly.
        }
        channel_->Execute(sim, state);
    }

    std::shared_ptr<Circuits::IOperation<>> Clone() const override
    {
        auto copy = channel_->Clone();
        const auto channel = std::dynamic_pointer_cast<ChannelOp>(copy);
        return channel ? std::make_shared<ReadPrefixBeforeChannel>(channel, words_, width_, backend_, values_) : copy;
    }

    std::shared_ptr<Circuits::IOperation<>> Remap(const Circuits::Circuit<double>::BitMapping &qubits,
                                                  const Circuits::Circuit<double>::BitMapping &bits) const override
    {
        auto mapped = channel_->Remap(qubits, bits);
        const auto channel = std::dynamic_pointer_cast<ChannelOp>(mapped);
        if (!channel || std::any_of(qubits.begin(), qubits.end(), [](const auto &entry) { return entry.first != entry.second; }))
            return mapped;
        return std::make_shared<ReadPrefixBeforeChannel>(channel, words_, width_, backend_, values_);
    }

  private:
    std::shared_ptr<ChannelOp> channel_;
    const std::vector<std::string> &words_;
    size_t width_;
    Simulators::SimulatorType backend_;
    std::vector<double> &values_;
};

EstimationResult EstimateFoldedRows(const TerminalExpectation &plan, const SimulatorConfig &config, EstimationCache *cache)
{
    auto result = EstimateCircuitTrajectory(plan.prefix, plan.prefix_observables, config, cache);
    result.values = plan.Combine(result.values, {});
    result.plan.error_bounds = plan.Combine(result.plan.error_bounds, {}, true);
    result.plan.qubits_before = plan.original_width;
    return result;
}
} // namespace

std::optional<EstimationResult> EstimateDeterministicTerminalPrefix(const TerminalExpectation &plan, const SimulatorConfig &config)
{
    if (plan.stochastic_prefix || plan.sampled_indices.empty())
        return std::nullopt;
    return EstimateFoldedRows(plan, config, nullptr);
}

EstimationResult EstimateTerminalExpectation(const TerminalExpectation &plan, const SimulatorConfig &config, EstimationCache *cache,
                                             const EstimationResult *deterministic_prefix, const RecordCircuitPtr &sampled_circuit)
{
    if (plan.sampled_indices.empty())
        return EstimateFoldedRows(plan, config, cache);
    // Both parts use the same realization seed, including on layouts that need two evolutions.
    auto trajectory = config;
    if (plan.stochastic_prefix && !trajectory.seed)
        trajectory.seed = Utils::RandomStream::FreshSeed();
    const auto &full = sampled_circuit ? sampled_circuit : plan.full_circuit;
    std::optional<EstimationResult> sampled;
    if (plan.prefix_read_position && full == plan.full_circuit)
    {
        std::vector<double> prefix_values;
        const auto observe = [&](const EstimationGroup &group) {
            const auto position = *plan.prefix_read_position;
            const auto &original = group.circuit->GetOperations();
            if (group.qubits != plan.original_width || original.size() <= position ||
                !std::equal(original.begin(), original.begin() + position + 1, full->GetOperations().begin()))
                return group.circuit;
            auto operations = original;
            operations[position] = std::make_shared<ReadPrefixBeforeChannel>(std::static_pointer_cast<ChannelOp>(original[position]), plan.prefix_observables,
                                                                             plan.original_width, config.simulator_type, prefix_values);
            return std::make_shared<Circuits::Circuit<double>>(operations);
        };
        sampled = EstimateCircuitTrajectory(full, plan.sampled_observables, trajectory, cache, observe);
        if (!prefix_values.empty())
        {
            sampled->values = plan.Combine(prefix_values, sampled->values);
            sampled->plan.error_bounds = plan.Combine(std::vector<double>(prefix_values.size()), sampled->plan.error_bounds, true);
            sampled->plan.qubits_before = plan.original_width;
            return std::move(*sampled);
        }
    }
    std::optional<EstimationResult> local;
    if (!deterministic_prefix)
    {
        local = EstimateFoldedRows(plan, trajectory, nullptr);
        deterministic_prefix = &*local;
    }
    auto result = sampled ? std::move(*sampled) : EstimateCircuitTrajectory(full, plan.sampled_observables, trajectory, cache);
    auto values = deterministic_prefix->values;
    auto bounds = deterministic_prefix->plan.error_bounds;
    for (size_t j = 0; j < plan.sampled_indices.size(); ++j)
    {
        values[plan.sampled_indices[j]] = result.values[j];
        bounds[plan.sampled_indices[j]] = result.plan.error_bounds[j];
    }
    result.values = std::move(values);
    result.plan.error_bounds = std::move(bounds);
    const auto &prefix = deterministic_prefix->plan;
    result.plan.qubits_before = plan.original_width;
    result.plan.qubits_after = std::max(result.plan.qubits_after, prefix.qubits_after);
    result.plan.total_cluster_qubits += prefix.total_cluster_qubits;
    result.plan.bounded_observables += prefix.bounded_observables;
    result.plan.auto_reduced = result.plan.auto_reduced || prefix.auto_reduced;
    result.plan.groups.insert(result.plan.groups.end(), prefix.groups.begin(), prefix.groups.end());
    result.cluster_methods.insert(result.cluster_methods.end(), deterministic_prefix->cluster_methods.begin(), deterministic_prefix->cluster_methods.end());
    result.gate_fusion = result.gate_fusion || deterministic_prefix->gate_fusion;
    result.gate_fusion_max_qubits = std::max(result.gate_fusion_max_qubits, deterministic_prefix->gate_fusion_max_qubits);
    if (local)
    {
        result.time_taken += local->time_taken;
        result.plan.reduction_time += prefix.reduction_time;
    }
    return result;
}

std::optional<TerminalExpectation> PrepareTerminalNoiseExpectation(const noise::NoiseInjectionPlan &plan, NoiseLayer layer,
                                                                   const RecordCircuitPtr &runtime_circuit, const std::vector<std::string> &observables,
                                                                   const SimulatorConfig &config)
{
    if (config.terminal_channel_policy != "auto" || !UsesChannelSamplingStatevector(config))
        return std::nullopt;
    if (runtime_circuit)
        return PrepareTerminalExpectation(runtime_circuit, observables, config);
    const auto &model = plan.model();
    if ((layer != NoiseLayer::kPauli && model.has_coherent()) || (layer == NoiseLayer::kCombined && model.has_correlated()) ||
        (!plan.circuit()->HasQuantumChannels() && !model.has_non_readout_noise()))
        return std::nullopt;
    // Pauli and short-T2 noise may already have been drawn into gates when the
    // first realization was prepared. Recover their distribution only if that
    // eliminates the sampled prefix too. Otherwise its draws and seeds stay as
    // they were in the scheduled realizations.
    auto candidate = PrepareTerminalExpectation(RecoverTerminalNoise(plan, layer), observables, config);
    if (candidate && !candidate->stochastic_prefix)
        return candidate;
    return std::nullopt;
}

RecordCircuitPtr WithoutCircuitChannels(const RecordCircuitPtr &circuit)
{
    auto out = std::make_shared<Circuits::Circuit<double>>();
    for (const auto &op : circuit->GetOperations())
        if (op->GetType() == Circuits::OperationType::kComposite)
        {
            // Nested channels are noise too; rebuild only circuits that hold one.
            const auto nested = std::static_pointer_cast<Circuits::Circuit<double>>(op);
            out->AddOperation(nested->HasQuantumChannels() ? WithoutCircuitChannels(nested) : op);
        }
        else if (op->GetType() != Circuits::OperationType::kQuantumChannel)
            out->AddOperation(op);
    return out;
}

NoisyEstimate EstimateNoisy(const RecordCircuitPtr &circuit, const std::vector<std::string> &observables, const noise::NoiseModel &noise_model,
                            std::optional<size_t> requested_realizations, const SimulatorConfig &config, std::optional<uint64_t> noise_seed, NoiseLayer layer,
                            std::optional<bool> exact_channels)
{
    if (!circuit)
        throw std::invalid_argument("Circuit is null.");
    Simulators::ValidatePauliStrings(observables);
    auto rng = MakeNoiseRng(config, noise_seed);
    NoisyEstimate out;
    out.policy = ResolveNoiseRealizations(requested_realizations, noise_model, config, layer, 100, circuit, true, exact_channels);
    out.realizations = out.policy.realizations;
    if (observables.empty())
    {
        out.ideal = EstimateCircuitTrajectory(WithoutCircuitChannels(circuit), observables, config);
        out.ideal_values = out.ideal.values;
        out.qubits_before = out.ideal.plan.qubits_before;
        out.qubits_after = out.ideal.plan.qubits_after;
        out.time_taken = out.ideal.time_taken;
        return out;
    }
    const size_t count = observables.size();
    const auto clean = circuit->RemoveEndMeasurements();
    const bool exact = out.policy.exact_channels;
    if (exact && SamplesQuantumChannels(config))
        throw std::invalid_argument("Exact noise requires a density-matrix or MPO backend, or the QCSim Pauli propagator: this backend takes a "
                                    "channel as one sampled trajectory");
    const noise::NoiseInjectionPlan plan(clean, noise_model, InjectsRuntimeChannels(config, exact), OrdersForChain(config, exact));
    // A backend that samples channels draws all the noise as each realization
    // runs (ResolveRuntimeNoise): they then share one circuit and differ by seed.
    RecordCircuitPtr runtime_circuit, first_realization;
    ResolveRuntimeNoise(layer, plan, rng, runtime_circuit, first_realization);
    const auto terminal = exact_channels.has_value() ? std::nullopt : PrepareTerminalNoiseExpectation(plan, layer, runtime_circuit, observables, config);
    bool deterministic_tail = false;
    if (terminal)
    {
        out.terminal_channels = true;
        deterministic_tail = !terminal->RequiresTrajectories();
        if (deterministic_tail)
        {
            out.policy.stochastic_execution = false;
            out.policy.stochastic_noise = false;
            out.policy.realizations = requested_realizations.value_or(1);
            out.realizations = out.policy.realizations;
        }
    }
    // Even an explicit request for several copies of one deterministic
    // result needs only one physical evolution. Report its requested schedule.
    const size_t noise_realizations = deterministic_tail ? 1 : out.realizations;
    const auto start = std::chrono::steady_clock::now();
    const auto deterministic_prefix = terminal ? EstimateDeterministicTerminalPrefix(*terminal, config) : std::nullopt;
    // Whether the executions draw channel branches, shared circuit or not: the noise seed then joins their seeds.
    RuntimeDraws draws(&plan, runtime_circuit, first_realization);
    // Called once per realization, in order: the one drawn to decide is the first.
    const auto realize = [&]() -> RecordCircuitPtr {
        if (terminal && terminal->sampled_indices.empty())
            return terminal->prefix;
        RecordCircuitPtr noisy = runtime_circuit     ? runtime_circuit
                                 : first_realization ? std::exchange(first_realization, nullptr)
                                                     : InjectPlanned(layer, plan, rng, exact);
        draws.See(noisy);
        return noisy;
    };

    // What the aggregate needs of each realization, kept in realization order.
    struct Realization
    {
        std::vector<double> values, bounds;
        std::vector<Simulators::SimulationType> methods;
        bool reduced = false;
        size_t before = 0, after = 0, groups = 0, cluster_qubits = 0, bounded = 0;
        double reduction_time = 0;
    };

    std::vector<Realization> realizations(noise_realizations);
    // Each realization's register also holds the idle qubits its observables
    // name: the host executor keeps every position of a Pauli string.
    size_t qubits = std::max({size_t{1}, clean->GetMaxQubitIndex() + 1, plan.RegisterWidth()});
    for (const auto &observable : observables)
        qubits = std::max(qubits, observable.size());
    const size_t workers = SamplingWorkers(config, noise_realizations, qubits);
    // The workers available to the realizations after the warm-up one below,
    // which runs alone: two realizations run one after the other whatever the
    // workers. An upper bound on what overlaps, not a measurement of it.
    out.realization_workers = workers > 1 ? std::max<size_t>(1, std::min(workers, noise_realizations - 1)) : 1;
    // A cache serves one thread (EstimationCache).
    std::vector<EstimationCache> caches(workers);
    const auto run = [&](EstimationCache &cache, size_t r, const RecordCircuitPtr &noisy) {
        const auto execution_config = draws ? RuntimeNoiseExecutionConfig(config, noise_seed, r) : NoiseExecutionConfig(config, noise_seed, r);
        auto result = terminal
                          ? EstimateTerminalExpectation(*terminal, execution_config, &cache, deterministic_prefix ? &*deterministic_prefix : nullptr, noisy)
                          : EstimateCircuitTrajectory(noisy, observables, execution_config, &cache);
        auto &record = realizations[r];
        record.values = std::move(result.values);
        record.bounds = std::move(result.plan.error_bounds);
        record.methods = std::move(result.cluster_methods);
        record.reduced = result.plan.auto_reduced;
        record.before = result.plan.qubits_before;
        record.after = result.plan.qubits_after;
        record.groups = result.plan.groups.size();
        record.cluster_qubits = result.plan.total_cluster_qubits;
        record.bounded = result.plan.bounded_observables;
        record.reduction_time = result.plan.reduction_time;
    };
    size_t first = 0, warm_hits = 0;
    if (workers > 1)
    {
        // Realization 0 warms one cache, which every worker then starts from,
        // so reuse matches a serial loop's whenever the noise keeps the
        // circuit's multi-qubit topology (the cache's signature).
        run(caches[0], 0, realize());
        first = 1;
        warm_hits = caches[0].hits;
        caches[0].hits = 0;
        for (size_t worker = 1; worker < workers; ++worker)
            caches[worker] = caches[0];
    }
    Utils::RunRealizations(
        noise_realizations - first, workers, [&](size_t) { return realize(); },
        [&](size_t worker, size_t i, RecordCircuitPtr noisy) { run(caches[worker], i + first, noisy); });
    out.time_taken = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    // Summed in realization order, as a serial loop would.
    std::vector<double> sums(count, 0.0), squares(count, 0.0);
    out.error_bounds.assign(count, 0.0);
    size_t total_clusters = 0;
    double total_cluster_qubits = 0;
    for (const auto &record : realizations)
    {
        if (record.values.size() != count || record.bounds.size() != count)
            throw std::runtime_error("Estimation returned an invalid expectation vector");
        out.auto_reduced = out.auto_reduced || record.reduced;
        out.qubits_before = std::max(out.qubits_before, record.before);
        out.qubits_after = std::max(out.qubits_after, record.after);
        out.clusters_count = std::max(out.clusters_count, record.groups);
        total_clusters += record.groups;
        const double mean = record.groups ? static_cast<double>(record.cluster_qubits) / record.groups : 0.0;
        total_cluster_qubits += record.groups * mean;
        out.reduction_time += record.reduction_time;
        out.bounded_observables = std::max(out.bounded_observables, record.bounded);
        for (size_t i = 0; i < count; ++i)
        {
            out.error_bounds[i] += record.bounds[i] / noise_realizations;
            sums[i] += record.values[i];
            squares[i] += record.values[i] * record.values[i];
        }
    }
    out.cluster_methods = realizations.back().methods;
    out.mean_cluster_qubits = total_clusters ? total_cluster_qubits / total_clusters : 0;
    out.cone_cache_hits = warm_hits;
    for (const auto &cache : caches)
        out.cone_cache_hits += cache.hits;
    out.values.resize(count);
    out.standard_errors.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        if (deterministic_prefix && terminal->IsExactRow(i))
        {
            out.values[i] = deterministic_prefix->values[i];
            out.error_bounds[i] = deterministic_prefix->plan.error_bounds[i];
            continue;
        }
        out.values[i] = sums[i] / noise_realizations;
        if (noise_realizations > 1)
            out.standard_errors[i] =
                std::sqrt(std::max(0.0, (squares[i] - noise_realizations * out.values[i] * out.values[i]) / (noise_realizations - 1) / noise_realizations));
    }
    const auto ideal = WithoutCircuitChannels(clean);
    // A dynamic baseline must average the same kind of measurement branches as
    // the noisy estimate. Returning one randomly collapsed ideal state makes
    // comparison with the noisy ensemble meaningless.
    const size_t ideal_realizations = HasStochasticEvolution(ideal, UsesExactQuantumChannels(config)) ? noise_realizations : 1;
    std::vector<double> ideal_sums(count, 0.0);
    EstimationCache ideal_cache;
    for (size_t r = 0; r < ideal_realizations; ++r)
    {
        const auto ideal_config = ideal_realizations > 1 ? NoiseExecutionConfig(config, noise_seed, r) : config;
        out.ideal = EstimateCircuitTrajectory(ideal, observables, ideal_config, &ideal_cache);
        for (size_t i = 0; i < count; ++i)
            ideal_sums[i] += out.ideal.values[i];
    }
    for (size_t i = 0; i < count; ++i)
        out.ideal.values[i] = ideal_sums[i] / ideal_realizations;
    out.ideal_values = out.ideal.values;
    return out;
}

namespace
{
struct SimulatorHandle
{
    unsigned long value;

    ~SimulatorHandle()
    {
        if (value)
            DestroySimpleSimulator(value);
    }
};

// A configured handle a realization takes over instead of allocating its
// register again: the backend probe of a candidate list. The first realization
// to run takes it, so it never outlives that realization: one of another
// width replaces its register (freeing the probe's) before allocating.
struct SpareRegister
{
    std::atomic<unsigned long> handle{0};

    unsigned long Take()
    {
        return handle.exchange(0);
    }

    ~SpareRegister()
    {
        if (const auto left = handle.exchange(0))
            DestroySimpleSimulator(left);
    }
};

RecordCircuitPtr BeforeFidelityReadout(const RecordCircuitPtr &circuit)
{
    const auto flat = circuit->HasCompositeOperations() ? circuit->InlinedComposites() : circuit;
    return flat->RemoveEndMeasurements();
}

double RealizationFidelityOn(const RecordCircuitPtr &ideal, const RecordCircuitPtr &noisy, const SimulatorConfig &config, SpareRegister *spare)
{
    auto combined = std::make_shared<Circuits::Circuit<double>>();
    // Fidelity is evaluated before terminal readout, as in inner_product.
    // This includes conditional terminal readout. NoisyFidelity validates that
    // no genuinely intermediate measurement remains in the input circuit.
    const auto before_readout = BeforeFidelityReadout(noisy);
    for (const auto &op : before_readout->GetOperations())
        combined->AddOperation(op->Clone());
    const auto &ideal_ops = ideal->GetOperations();
    for (auto it = ideal_ops.rbegin(); it != ideal_ops.rend(); ++it)
        if (auto adjoint = adjoint_gate(*it))
            combined->AddOperation(adjoint);

    const int num_qubits = std::max(1, static_cast<int>(combined->GetMaxQubitIndex()) + 1);
    GetMaestroObjectWithMute();
    // The spare was configured for this backend; the seed below replaces the
    // probe's, and a width mismatch replaces its register.
    const unsigned long adopted = spare ? spare->Take() : 0;

    SimulatorHandle handle{adopted ? adopted : CreateSimpleSimulator(num_qubits)};

    if (!handle.value)
        throw std::runtime_error("Failed to create simulator handle.");
    // Exactly the circuit's register: the default allocation also adds the
    // host's network-entangled qubit (2x statevector, 4x density-matrix work).
    const size_t register_qubits = static_cast<size_t>(num_qubits);
    auto network =
        adopted ? static_cast<Maestro *>(GetMaestroObject())->GetSimpleSimulator(handle.value) : ConfigureNetwork(handle.value, config, register_qubits);
    if (!network)
        throw std::runtime_error("Failed to configure network.");
    auto simulator = network->GetSimulator();
    // A fixed backend was already created at this width; replace only an
    // automatic-selection placeholder (or another width), releasing it first
    // so that the two registers never coexist.
    if (!simulator || simulator->GetType() != config.simulator_type || simulator->GetSimulationType() != config.simulation_type ||
        simulator->GetNumberOfQubits() != register_qubits)
    {
        simulator.reset();
        network->CreateSimulator(config.simulator_type, config.simulation_type, register_qubits);
        simulator = network->GetSimulator();
    }
    if (!simulator)
        throw std::runtime_error("noisy_fidelity: requested backend is unavailable.");
    if (config.seed)
        simulator->SetSeed(*config.seed);
    Circuits::OperationState state(std::max(register_qubits, combined->GetMaxCbitIndex() + 1));
    combined->ExecutePhysicalOperations(simulator, state);
    return simulator->Probability(0);
}
} // namespace

double RealizationFidelity(const RecordCircuitPtr &ideal, const RecordCircuitPtr &noisy, const SimulatorConfig &config)
{
    if (!ideal || !noisy)
        throw std::invalid_argument("RealizationFidelity requires both an ideal circuit and a noisy circuit.");
    const auto preparation = BeforeFidelityReadout(ideal);
    if (preparation->HasQuantumChannels() || !HasUnitaryIdeal(preparation))
        throw std::invalid_argument("RealizationFidelity requires a unitary ideal preparation. Mid-circuit measurements, resets on used qubits, "
                                    "classical control and quantum channels are unsupported in the ideal circuit.");
    return RealizationFidelityOn(PreparedUnitary(preparation), noisy, config, nullptr);
}

bool HasUnitaryIdeal(const RecordCircuitPtr &before_readout)
{
    std::unordered_set<Types::qubit_t> touched;
    return before_readout->HasDeterministicEvolutionWith(Circuits::Circuit<double>::ChannelDraws::kNone, false, true, touched);
}

// The unitary that the ideal circuit applies to a fresh register. A reset
// before any other operation on its wire (all NoisyFidelity admits) prepares
// its target from |0>: nothing for |0>, an X for |1>. The adjoint of that X
// then undoes the preparation in RealizationFidelity.
RecordCircuitPtr PreparedUnitary(const RecordCircuitPtr &circuit)
{
    auto out = std::make_shared<Circuits::Circuit<double>>();
    for (const auto &op : circuit->GetOperations())
        if (op->GetType() == Circuits::OperationType::kReset)
        {
            const auto reset = std::static_pointer_cast<Circuits::Reset<double>>(op);
            const auto &qubits = reset->GetQubits();
            const auto &targets = reset->GetResetTargets();
            for (size_t i = 0; i < qubits.size() && i < targets.size(); ++i)
                if (targets[i])
                    out->AddOperation(std::make_shared<Circuits::XGate<double>>(qubits[i]));
        }
        else if (op->GetType() == Circuits::OperationType::kComposite)
            out->AddOperation(PreparedUnitary(std::static_pointer_cast<Circuits::Circuit<double>>(op)));
        else
            out->AddOperation(op);
    return out;
}

NoisyFidelityResult NoisyFidelity(const RecordCircuitPtr &circuit, const noise::NoiseModel &noise_model, std::optional<size_t> requested_realizations,
                                  const SimulatorConfig &requested_config, std::optional<uint64_t> noise_seed)
{
    if (!circuit)
        throw std::invalid_argument("Circuit is null.");
    // Inspect measurements before stripping channels: a measurement before an
    // explicit channel is not a terminal readout even if the ideal omits it.
    // Gates, delays, barriers, channels (noise, absent from the ideal) and
    // nested circuits of them are admitted, as are resets that precede every
    // other operation on their wire: on the fresh register they prepare |0> or
    // |1>, which the ideal state includes (PreparedUnitary).
    const auto before_readout = BeforeFidelityReadout(circuit);
    if (!HasUnitaryIdeal(before_readout))
        throw std::invalid_argument("noisy_fidelity requires a unitary ideal circuit; mid-circuit measurements, resets after other operations "
                                    "on their qubit and classical control are unsupported.");
    const auto ideal = PreparedUnitary(WithoutCircuitChannels(before_readout));
    noise::NoiseInjectionPlan plan(circuit, noise_model);
    // The register each realization evaluates: fidelity drops terminal
    // readout, so a measured qubit no gate touches is not part of it, while a
    // crosstalk spectator of the noise is (RealizationFidelity).
    size_t qubits = std::max<size_t>(1, before_readout->GetMaxQubitIndex() + 1);
    for (const auto spectator : plan.Spectators())
        qubits = std::max(qubits, spectator + 1);
    // Each realization creates its simulator explicitly: policy, allocation and
    // workers must all describe the one backend a candidate list resolves to,
    // tested on that register. Probing several candidates allocates the
    // winner's register; the first realization takes it over.
    SpareRegister spare;
    unsigned long probed = 0;
    const auto config = ResolveCandidateBackend(requested_config, qubits, probed);
    spare.handle = probed;
    const auto policy = ResolveNoiseRealizations(requested_realizations, noise_model, config, NoiseLayer::kCombined, 100, circuit, true);
    const size_t noise_realizations = policy.realizations;
    auto rng = MakeNoiseRng(config, noise_seed);
    const bool exact = policy.exact_channels;
    // A backend that samples channels from its state draws their branches as
    // each realization runs: its fidelity is that of one trajectory, and the
    // mean over realizations that of the channels, with nothing approximated.
    plan.KeepRuntimeChannels(policy.runtime_channels, policy.chain_order);
    // When it draws all the noise that way (ResolveRuntimeNoise), the realizations share one circuit and differ by
    // seed. Whenever a realization keeps channels that are drawn, shared circuit or not (RuntimeDraws), the noise
    // seed has no injection to drive for them and joins the seed of each execution (RuntimeNoiseExecutionConfig),
    // as in EstimateNoisy: either seed alone still changes the errors drawn.
    RecordCircuitPtr runtime_circuit, first_realization;
    ResolveRuntimeNoise(NoiseLayer::kCombined, plan, rng, runtime_circuit, first_realization);
    RuntimeDraws draws(&plan, runtime_circuit, first_realization);

    struct Realization
    {
        RecordCircuitPtr circuit;
        SimulatorConfig config;
    };

    const size_t workers = SamplingWorkers(config, noise_realizations, qubits);
    std::vector<double> fidelities(noise_realizations);
    const auto start = std::chrono::steady_clock::now();
    Utils::RunRealizations(
        noise_realizations, workers,
        [&](size_t r) {
            // Called once per realization, in order: the one drawn to decide is the first.
            auto noisy = runtime_circuit     ? runtime_circuit
                         : first_realization ? std::exchange(first_realization, nullptr)
                                             : InjectPlanned(NoiseLayer::kCombined, plan, rng, exact);
            draws.See(noisy);
            // Reset collapse must use a fresh seed for each realization, drawn
            // after its injection from the same stream.
            auto realization_config = draws ? RuntimeNoiseExecutionConfig(config, noise_seed, r) : NoiseExecutionConfig(config, noise_seed, r);
            if (!realization_config.seed)
                realization_config.seed = rng();
            return Realization{std::move(noisy), std::move(realization_config)};
        },
        [&](size_t, size_t r, Realization realization) { fidelities[r] = RealizationFidelityOn(ideal, realization.circuit, realization.config, &spare); });

    NoisyFidelityResult out;
    out.policy = policy;
    out.time_taken = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    out.realizations = noise_realizations;
    out.realization_workers = workers;
    double sum = 0, squares = 0;
    for (const double fidelity : fidelities)
    {
        sum += fidelity;
        squares += fidelity * fidelity;
    }
    const double n = static_cast<double>(noise_realizations);
    out.fidelity = sum / n;
    const double variance = noise_realizations > 1 ? (squares - n * out.fidelity * out.fidelity) / (n - 1) : 0.0;
    out.std_error = std::sqrt(std::max(variance, 0.0) / n);
    return out;
}

void ValidateNoisyStateProbability(const RecordCircuitPtr &circuit, const std::vector<bool> &target)
{
    if (target.empty() || target.size() > QC::PathIntegral::FastVectorBool::MaxWords * 64)
        throw std::invalid_argument("target_state must have between 1 and 1024 bits.");
    Circuits::ForEachUnitaryGate(circuit, "Noisy path-integral probability", Circuits::UnitaryPreparationQuery::NoisyProbability, [&](const auto &gate) {
        // The path integral supports every named Maestro gate. Validation needs neither a simulator nor randomness.
        const auto type = gate->GetGateType();
        if (type < Circuits::QuantumGateType::kPhaseGateType || type >= Circuits::QuantumGateType::kNone)
            throw std::invalid_argument("Unsupported gate in noisy path-integral probability.");
    });
    if (circuit->GetMaxQubitIndex() >= target.size())
        throw std::invalid_argument("target_state must cover the circuit's qubits.");
}

namespace
{
// At most one recorded bit flips in the first-order readout model. Precompute the probability of the other bits
// staying fixed with prefix and suffix products, including when some bit always flips.
class TargetReadout
{
  public:
    TargetReadout(const std::vector<bool> &target, const noise::NoiseModel &model) : target_(target)
    {
        if (!model.has_readout_error())
            return;
        const size_t n = target.size();
        std::vector<double> keep(n, 1.), into(n, 0.), prefix(n + 1, 1.);
        for (size_t q = 0; q < n; ++q)
        {
            if (const auto *rate = model.get_readout_error(static_cast<int>(q)))
            {
                keep[q] -= target[q] ? rate->p_meas0_prep1 : rate->p_meas1_prep0;
                into[q] = target[q] ? rate->p_meas1_prep0 : rate->p_meas0_prep1;
            }
            prefix[q + 1] = prefix[q] * keep[q];
        }
        unchanged_ = prefix[n];
        double suffix = 1.;
        for (size_t q = n; q-- > 0;)
        {
            const double weight = into[q] * prefix[q] * suffix;
            if (weight > 0.)
                flips_.emplace_back(q, weight);
            suffix *= keep[q];
        }
    }

    template <class Probability> double Evaluate(Probability probability) const
    {
        double value = unchanged_ ? unchanged_ * probability(target_) : 0.;
        if (flips_.empty())
            return value;
        auto flipped = target_;
        for (const auto &[q, weight] : flips_)
        {
            flipped[q] = !target_[q];
            value += weight * probability(flipped);
            flipped[q] = target_[q];
        }
        return value;
    }

  private:
    const std::vector<bool> &target_;
    double unchanged_ = 1.;
    std::vector<std::pair<size_t, double>> flips_;
};
} // namespace

NoisyProbabilityResult NoisyStateProbability(const RecordCircuitPtr &circuit, const std::vector<bool> &target, const noise::NoiseModel &model,
                                             std::optional<size_t> requested_realizations, const SimulatorConfig &requested_config,
                                             std::optional<uint64_t> noise_seed, NoiseLayer layer)
{
    ValidateNoisyStateProbability(circuit, target);
    auto config = requested_config;
    config.Validate();
    if (HasCandidateList(config) || config.simulator_type != Simulators::SimulatorType::kQCSim ||
        config.simulation_type != Simulators::SimulationType::kPathIntegral)
        throw std::invalid_argument("noisy_prob requires the fixed QCSim path_integral method.");
    if (const auto *error = NoisyProbabilityLayerError(model, layer))
        throw std::invalid_argument(error);
    config.fixed_backend = true;
    // Native requests promote a lone noise seed to the execution seed. Do the same here so that a runtime
    // Kraus branch uses identical streams through C++, JSON and Python, including when only one seed is given.
    if (!EffectiveSamplingSeed(config) && noise_seed)
        config.seed = *noise_seed;

    const auto start = std::chrono::steady_clock::now();
    const auto clean = circuit->RemoveEndMeasurements();
    auto rng = MakeNoiseRng(config, noise_seed);
    NoisyProbabilityResult out;
    out.policy = ResolveNoiseRealizations(requested_realizations, model, config, layer, kMeasurementTrajectories, clean, true);
    const TargetReadout readout(target, model);
    if (!model.has_non_readout_noise() && !clean->HasQuantumChannels())
    {
        // The recorded unitary circuit can prune paths to each endpoint. Using the full evolved frontier here
        // would lose that advantage on large coherent queries with or without classical readout error.
        Simulators::PathIntegralSimulator paths;
        if (config.path_integral_threshold)
            paths.SetTrimValue(*config.path_integral_threshold);
        else if (const auto it = config.native_options.find("path_integral_threshold"); it != config.native_options.end())
            paths.SetTrimValue(std::stod(it->second));
        paths.SetStartZeroState(target.size());
        std::string error;
        if (!paths.SetCircuit(clean, error))
            throw std::invalid_argument(error);
        out.probability = readout.Evaluate([&](const auto &bits) { return std::norm(paths.AmplitudeFromZero(bits)); });
        out.std_error = 0.;
        out.policy.realizations = requested_realizations.value_or(1);
        out.realizations_evaluated = 1;
    }
    else
    {
        const noise::NoiseInjectionPlan plan(clean, model, out.policy.runtime_channels);
        RecordCircuitPtr shared, first;
        ResolveRuntimeNoise(layer, plan, rng, shared, first);
        RuntimeDraws draws(&plan, shared, first);
        GetMaestroObjectWithMute();
        SimulatorHandle handle{CreateSimpleSimulator(static_cast<int>(target.size()))};
        auto network = handle.value ? ConfigureNetwork(handle.value, config, target.size()) : nullptr;
        if (!network)
            throw std::runtime_error("Could not create the path integral simulator for a noisy probability.");
        auto sim = network->GetSimulator();
        double m2 = 0.;
        for (size_t r = 0; r < out.policy.realizations; ++r)
        {
            auto noisy = shared ? shared : first ? std::exchange(first, nullptr) : InjectPlanned(layer, plan, rng, false);
            if (plan.RegisterWidth() > target.size())
                noisy = noise::drop_outside_spectator_phases(noisy, target.size());
            draws.See(noisy);
            const auto execution = draws ? RuntimeNoiseExecutionConfig(config, noise_seed, r) : NoiseExecutionConfig(config, noise_seed, r);
            sim->Reset();
            if (execution.seed)
                sim->SetSeed(*execution.seed);
            Circuits::OperationState state(clean->GetMaxCbitIndex() + 1);
            noisy->ExecutePhysicalOperations(sim, state);
            const double sample = readout.Evaluate([&](const auto &bits) { return sim->ProbabilityBits(bits); });
            const double delta = sample - out.probability;
            out.probability += delta / static_cast<double>(r + 1);
            m2 += delta * (sample - out.probability);
        }
        if (out.policy.realizations > 1)
            out.std_error = std::sqrt(std::max(0., m2) / static_cast<double>(out.policy.realizations - 1) / static_cast<double>(out.policy.realizations));
        else if (!out.policy.stochastic_execution && !out.policy.stochastic_noise)
            out.std_error = 0.;
        out.realizations_evaluated = out.policy.realizations;
    }
    out.realizations = out.policy.realizations;
    out.time_taken = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return out;
}
} // namespace MaestroExecution
