#ifdef MAESTRO_TENSOR_TEST_MAIN
#define BOOST_TEST_MODULE Tensor contraction and planning
#include <boost/test/included/unit_test.hpp>
#else
// Combined test projects provide the Boost.Test runner in their main file.
#include <boost/test/unit_test.hpp>
#endif

#include <complex>
#include <deque>
#include <numeric>
#include <random>

#include "../TensorNetworks/ForestContractor.h"

namespace {
using Complex = std::complex<double>;
using Tensor = Utils::Tensor<>;
using Pairs = std::vector<std::pair<size_t, size_t>>;

// Deliberately simple definition of contraction: decode coordinates, substitute
// the summed indices, then compute offsets. It shares no optimized machinery.
template <class T, class Storage>
Utils::Tensor<T, Storage> Reference(const Utils::Tensor<T, Storage> &a,
                                    const Utils::Tensor<T, Storage> &b,
                                    const Pairs &pairs) {
  std::vector<bool> ca(a.GetRank()), cb(b.GetRank());
  size_t reduction = 1;
  for (auto [i, j] : pairs) {
    ca[i] = cb[j] = true;
    reduction *= a.GetDim(i);
  }
  std::vector<size_t> dims;
  for (size_t i = 0; i < a.GetRank(); ++i)
    if (!ca[i]) dims.push_back(a.GetDim(i));
  for (size_t i = 0; i < b.GetRank(); ++i)
    if (!cb[i]) dims.push_back(b.GetDim(i));
  if (dims.empty()) dims.push_back(1);
  Utils::Tensor<T, Storage> result(dims);
  std::vector<size_t> ai(a.GetRank()), bi(b.GetRank());
  auto offset = [](const auto &coordinates, const auto &shape) {
    size_t value = 0;
    for (size_t i = shape.size(); i-- > 0;)
      value = value * shape[i] + coordinates[i];
    return value;
  };
  for (size_t out = 0; out < result.GetSize(); ++out) {
    size_t x = out;
    for (size_t i = 0; i < a.GetRank(); ++i)
      if (!ca[i]) {
        ai[i] = x % a.GetDim(i);
        x /= a.GetDim(i);
      }
    for (size_t i = 0; i < b.GetRank(); ++i)
      if (!cb[i]) {
        bi[i] = x % b.GetDim(i);
        x /= b.GetDim(i);
      }
    T sum{};
    for (size_t term = 0; term < reduction; ++term) {
      x = term;
      for (auto [i, j] : pairs) {
        ai[i] = bi[j] = x % a.GetDim(i);
        x /= a.GetDim(i);
      }
      sum = sum + a[offset(ai, a.GetDims())] * b[offset(bi, b.GetDims())];
    }
    result[out] = sum;
  }
  return result;
}

template <class T = Complex, class Storage = std::valarray<T>>
Utils::Tensor<T, Storage> RandomTensor(const std::vector<size_t> &dims,
                                       std::mt19937_64 &rng) {
  Utils::Tensor<T, Storage> tensor(dims);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  for (size_t i = 0; i < tensor.GetSize(); ++i) {
    if constexpr (std::is_same_v<T, Complex> ||
                  std::is_same_v<T, std::complex<float>>)
      tensor[i] = T(static_cast<typename T::value_type>(dist(rng)),
                    static_cast<typename T::value_type>(dist(rng)));
    else if constexpr (std::is_integral_v<T>)
      tensor[i] = T(int(rng() % 7) - 3);
    else
      tensor[i] = T(dist(rng));
  }
  return tensor;
}

template <class T, class Storage>
void CheckEqual(const Utils::Tensor<T, Storage> &a,
                const Utils::Tensor<T, Storage> &b, double tolerance = 1e-11) {
  BOOST_REQUIRE(a.GetDims() == b.GetDims());
  double error = 0, scale = 1;
  for (size_t i = 0; i < a.GetSize(); ++i) {
    error = std::max(error, double(std::abs(a[i] - b[i])));
    scale = std::max(scale, double(std::abs(a[i])));
  }
  BOOST_CHECK_LE(error, tolerance * scale);
}

double QubitZero(const QC::QubitRegister<> &state, size_t qubits,
                 size_t qubit) {
  double result = 0;
  for (size_t basis = 0; basis < (size_t(1) << qubits); ++basis)
    if (!(basis & (size_t(1) << qubit)))
      result += state.getBasisStateProbability(basis);
  return result;
}
}  // namespace

BOOST_AUTO_TEST_CASE(random_axis_pairings) {
  std::mt19937_64 rng(184792);
  for (size_t trial = 0; trial < 300; ++trial) {
    size_t ra = 1 + rng() % 5, rb = 1 + rng() % 5;
    std::vector<size_t> da(ra), db(rb), aa(ra), ab(rb);
    for (auto &d : da) d = 1 + rng() % 3;
    for (auto &d : db) d = 1 + rng() % 3;
    std::iota(aa.begin(), aa.end(), 0);
    std::iota(ab.begin(), ab.end(), 0);
    std::shuffle(aa.begin(), aa.end(), rng);
    std::shuffle(ab.begin(), ab.end(), rng);
    Pairs pairs;
    const size_t shared = rng() % (std::min(ra, rb) + 1);
    for (size_t i = 0; i < shared; ++i) {
      db[ab[i]] = da[aa[i]];
      pairs.emplace_back(aa[i], ab[i]);
    }
    const auto a = RandomTensor(da, rng), b = RandomTensor(db, rng);
    const auto expected = Reference(a, b, pairs);
    CheckEqual(expected, a.Contract(b, pairs, false));
    CheckEqual(expected, a.Contract(b, pairs, true));
    if (pairs.size() == 1)
      CheckEqual(expected,
                 a.Contract(b, pairs[0].first, pairs[0].second, false));
  }
}

BOOST_AUTO_TEST_CASE(matrix_tiles_and_packing) {
  std::mt19937_64 rng(2419);
  auto a = RandomTensor({8, 8, 16, 16}, rng),
       b = RandomTensor({16, 8, 16, 8}, rng);
  const Pairs pairs{{2, 0}, {1, 1}};
  const auto expected = Reference(a, b, pairs);
  const int eigenThreads = Eigen::nbThreads();
  CheckEqual(expected, a.Contract(b, pairs, false));
  CheckEqual(expected, a.Contract(b, pairs, true));
  BOOST_CHECK_EQUAL(Eigen::nbThreads(), eigenThreads);
  a = RandomTensor({3, 11, 9}, rng);
  b = RandomTensor({9, 5, 7}, rng);
  CheckEqual(Reference(a, b, {{2, 0}}), a.Contract(b, 2, 0, true));
}

BOOST_AUTO_TEST_CASE(small_gates_on_large_tensors) {
  std::mt19937_64 rng(9123);
  for (size_t rank : {5, 8, 12})
    for (size_t half : {1, 2})
      for (size_t trial = 0; trial < 12; ++trial) {
        std::vector<size_t> da(rank, 2), db(half * 2, 2), aa(rank),
            bb(half * 2);
        std::iota(aa.begin(), aa.end(), 0);
        std::iota(bb.begin(), bb.end(), 0);
        std::shuffle(aa.begin(), aa.end(), rng);
        std::shuffle(bb.begin(), bb.end(), rng);
        Pairs pairs, reversed;
        for (size_t i = 0; i < half; ++i) {
          pairs.emplace_back(aa[i], bb[i]);
          reversed.emplace_back(bb[i], aa[i]);
        }
        // Nonbinary free axes must also preserve the output layout.
        if (trial % 3 == 0) da[aa[half]] = 3;
        if (trial % 3 == 1) da[aa[half]] = 1;
        const auto a = RandomTensor(da, rng), b = RandomTensor(db, rng);
        const auto expected = Reference(a, b, pairs);
        const auto swapped = Reference(b, a, reversed);
        for (bool threaded : {false, true}) {
          CheckEqual(expected, a.Contract(b, pairs, threaded));
          CheckEqual(swapped, b.Contract(a, reversed, threaded));
        }
      }
  auto a = RandomTensor<double, std::vector<double>>({2, 3, 2, 2, 2}, rng);
  auto b = RandomTensor<double, std::vector<double>>({2, 2}, rng);
  CheckEqual(Reference(a, b, {{2, 1}}), a.Contract(b, 2, 1));
  auto c = RandomTensor<std::complex<float>>({2, 2, 2, 2, 2}, rng);
  auto d = RandomTensor<std::complex<float>>({2, 2, 2, 2}, rng);
  const Pairs pairs{{1, 3}, {4, 0}};
  CheckEqual(Reference(c, d, pairs), c.Contract(d, pairs), 1e-5);
  auto e = RandomTensor<int, std::vector<int>>({2, 3, 2, 2, 2}, rng);
  auto f = RandomTensor<int, std::vector<int>>({2, 2}, rng);
  CheckEqual(Reference(e, f, {{2, 1}}), e.Contract(f, 2, 1), 0);
  auto g = RandomTensor<Complex, std::deque<Complex>>({2, 3, 2, 2, 2}, rng);
  auto h = RandomTensor<Complex, std::deque<Complex>>({2, 2}, rng);
  CheckEqual(Reference(g, h, {{2, 1}}), g.Contract(h, 2, 1));
}

BOOST_AUTO_TEST_CASE(prepared_metadata_numerics_and_budget) {
  std::mt19937_64 rng(721);
  struct Case {
    std::vector<size_t> a, b;
    Pairs pairs;
  };
  const std::vector<Case> cases{
      {{2, 2, 2, 2}, {2, 2, 2, 2}, {{3, 0}, {2, 1}}},
      {{2, 3, 2, 2, 2}, {2, 2}, {{3, 1}}},
      {{2, 2, 2, 2}, {2, 3, 2, 2, 2}, {{3, 0}, {0, 4}}},
      {{35, 9}, {9, 131}, {{1, 0}}},
      {{3, 9, 4, 5}, {7, 3, 9, 5}, {{3, 3}, {1, 2}, {0, 1}}},
      {{2, 3, 4}, {4, 2, 3}, {{2, 0}, {0, 1}, {1, 2}}},
      {{1, 1}, {1}, {{1, 0}}},
      {{2, 3}, {4, 2}, {}}};
  for (const auto &item : cases) {
    auto a = RandomTensor(item.a, rng), b = RandomTensor(item.b, rng);
    Utils::detail::TensorContractionPlan plan;
    BOOST_REQUIRE(plan.Prepare(item.a, item.b, item.pairs, 1024 * 1024));
    Tensor result(plan.GetDims());
    for (size_t pass = 0; pass < 3; ++pass) {
      for (bool threaded : {false, true}) {
        plan.Execute(&a[size_t(0)], &b[size_t(0)], &result[size_t(0)],
                     threaded);
        CheckEqual(Reference(a, b, item.pairs), result);
      }
      b = RandomTensor(item.b, rng);
    }
  }
  Utils::detail::TensorContractionPlan plan;
  // An irregular 2^30-element offset table must be rejected before allocation.
  BOOST_CHECK(!plan.Prepare({size_t(1) << 20, 2, size_t(1) << 10}, {2, 2},
                            Pairs{{1, 0}}, 1024));
  // A huge irregular scalar reduction needs only rank-sized metadata.
  BOOST_CHECK(plan.Prepare({size_t(1) << 15, size_t(1) << 15},
                           {size_t(1) << 15, size_t(1) << 15},
                           Pairs{{0, 1}, {1, 0}}, 1024));
  BOOST_CHECK_LE(plan.ExtraBytes(), 1024);
  BOOST_CHECK_THROW(plan.Prepare({2, 2}, {2, 2}, Pairs{{0, 0}, {0, 1}}, 1024),
                    std::invalid_argument);
  BOOST_CHECK(!plan.Prepare({0, 2}, {2}, Pairs{{1, 0}}, 1024));
}

BOOST_AUTO_TEST_CASE(scalar_reductions_and_nonconjugating_product) {
  std::mt19937_64 rng(95);
  auto a = RandomTensor(std::vector<size_t>(18, 2), rng);
  auto b = RandomTensor(std::vector<size_t>(18, 2), rng);
  Pairs pairs;
  for (size_t i = 0; i < 18; ++i) pairs.emplace_back(i, i);
  auto expected = Reference(a, b, pairs);
  CheckEqual(expected, a.Contract(b, pairs, false));
  CheckEqual(expected, a.Contract(b, pairs, true));
  for (size_t i = 0; i < pairs.size(); ++i)
    pairs[i].second = pairs.size() - i - 1;
  expected = Reference(a, b, pairs);
  CheckEqual(expected, a.Contract(b, pairs, false));
  CheckEqual(expected, a.Contract(b, pairs, true));
  Tensor imaginary({2});
  imaginary[0] = imaginary[1] = Complex(0, 1);
  BOOST_CHECK_SMALL(
      std::abs(imaginary.Contract(imaginary, 0, 0)[size_t(0)] + 2.0), 1e-15);
}

BOOST_AUTO_TEST_CASE(dummy_high_rank_and_size_safety) {
  Tensor a(std::vector<size_t>(70, 2), true), b({2}, true);
  auto result = a.Contract(b, 0, 0);
  BOOST_CHECK(result.IsDummy());
  BOOST_CHECK_EQUAL(result.GetRank(), 69);
  if (sizeof(size_t) > 4) {
    Tensor big(std::vector<size_t>(40, 2), true);
    BOOST_CHECK_EQUAL(big.GetSize(), size_t(1ULL << 40));
  }
  BOOST_CHECK_THROW(a.GetSize(), std::length_error);
  std::mt19937_64 rng(29);
  std::vector<size_t> dimensions(40, 1);
  dimensions[2] = dimensions[35] = 2;
  auto high = RandomTensor(dimensions, rng), other = RandomTensor({2, 3}, rng);
  CheckEqual(Reference(high, other, {{35, 0}}), high.Contract(other, 35, 0));
  Tensor moved(std::move(other));
  BOOST_CHECK_EQUAL(moved.GetSize(), 6);
}

BOOST_AUTO_TEST_CASE(invalid_axes) {
  Tensor a({2, 3}), b({2, 4});
  BOOST_CHECK_THROW(a.Contract(b, 2, 0), std::invalid_argument);
  BOOST_CHECK_THROW(a.Contract(b, 1, 1), std::invalid_argument);
  const Pairs repeated{{0, 0}, {0, 0}};
  BOOST_CHECK_THROW(a.Contract(b, repeated), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(scalar_types_and_storage) {
  std::mt19937_64 rng(295);
  auto realA = RandomTensor<double, std::vector<double>>({16, 16}, rng);
  auto realB = RandomTensor<double, std::vector<double>>({16, 16}, rng);
  CheckEqual(Reference(realA, realB, {{1, 0}}), realA.Contract(realB, 1, 0));
  auto floatA = RandomTensor<std::complex<float>>({16, 16}, rng);
  auto floatB = RandomTensor<std::complex<float>>({16, 16}, rng);
  CheckEqual(Reference(floatA, floatB, {{1, 0}}), floatA.Contract(floatB, 1, 0),
             1e-5);
  auto integerA = RandomTensor<int, std::vector<int>>({8, 9, 3}, rng);
  auto integerB = RandomTensor<int, std::vector<int>>({9, 8}, rng);
  CheckEqual(Reference(integerA, integerB, {{1, 0}}),
             integerA.Contract(integerB, 1, 0), 0);
  auto dequeA = RandomTensor<Complex, std::deque<Complex>>({16, 16}, rng);
  auto dequeB = RandomTensor<Complex, std::deque<Complex>>({16, 16}, rng);
  CheckEqual(Reference(dequeA, dequeB, {{1, 0}}),
             dequeA.Contract(dequeB, 1, 0));
}

BOOST_AUTO_TEST_CASE(openmp_limits_and_nested_call) {
#ifdef _OPENMP
  const int saved = omp_get_max_threads();
  omp_set_num_threads(2);
  BOOST_CHECK_LE(Utils::detail::TensorContractionThreads(100, 1e9, true), 2);
  BOOST_CHECK_EQUAL(Utils::detail::TensorContractionThreads(100, 1e9, false),
                    1);
  std::mt19937_64 rng(98);
  const auto a = RandomTensor({64, 64}, rng), b = RandomTensor({64, 64}, rng);
  const auto expected = Reference(a, b, {{1, 0}});
  std::array<Tensor, 2> results;
  std::array<int, 2> nestedThreads{};
#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 2; ++i) {
    nestedThreads[i] = Utils::detail::TensorContractionThreads(100, 1e9, true);
    results[i] = a.Contract(b, 1, 0, true);
  }
  omp_set_num_threads(saved);
  for (size_t i = 0; i < results.size(); ++i) {
    BOOST_CHECK_EQUAL(nestedThreads[i], 1);
    CheckEqual(expected, results[i]);
  }
#endif
}

BOOST_AUTO_TEST_CASE(network_reference_and_repeated_conditioning) {
  constexpr size_t count = 6;
  TensorNetworks::TensorNetwork net(count);
  auto contractor = std::make_shared<TensorNetworks::ForestContractor>();
  net.SetContractor(contractor);
  net.SetSeed(181);
  QC::QubitRegister<> state(count);
  state.SetMultithreading(false);
  QC::Gates::HadamardGate<> h;
  QC::Gates::CNOTGate<> cx;
  net.AddGate(h, 0);
  state.ApplyGate(h, 0);
  for (size_t layer = 0; layer < 4; ++layer) {
    Eigen::MatrixXcd matrix(2, 2);
    const double angle = .17 + .09 * layer;
    matrix << std::cos(angle), -std::sin(angle), std::sin(angle),
        std::cos(angle);
    QC::Gates::SingleQubitGate<> rotation(matrix);
    for (size_t q = 0; q < count; ++q) {
      net.AddGate(rotation, q);
      state.ApplyGate(rotation, q);
    }
    for (size_t q = layer % 2; q + 1 < count; q += 2) {
      // TensorNetwork orders control then target; QubitRegister takes target
      // then control, as does the statevector simulator's low-level API.
      net.AddGate(cx, q, q + 1);
      state.ApplyGate(cx, q + 1, q);
    }
  }
  for (int pass = 0; pass < 2; ++pass)
    for (size_t q = 0; q < count; ++q) {
      const double p0 = QubitZero(state, count, q);
      BOOST_CHECK_SMALL(net.Probability(q) - p0, 1e-11);
      BOOST_CHECK_SMALL(net.Probability(q, false) - (1 - p0), 1e-11);
    }
  BOOST_CHECK_GE(contractor->GetPlanCacheHits(), count);
  for (size_t basis = 0; basis < (size_t(1) << count); basis += 7)
    BOOST_CHECK_SMALL(net.getBasisStateProbability(basis) -
                          state.getBasisStateProbability(basis),
                      1e-11);
  BOOST_CHECK_SMALL(
      net.ExpectationValue("Z") - (2 * QubitZero(state, count, 0) - 1), 1e-11);

  const auto original = state;
  net.SaveState();
  for (size_t shot = 0; shot < 5; ++shot) {
    state = original;
    for (size_t q = 0; q < count; ++q) {
      const double p0 = QubitZero(state, count, q);
      BOOST_CHECK_SMALL(net.Probability(q) - p0, 1e-10);
      const bool outcome = net.Measure(q);
      Eigen::MatrixXcd projector = Eigen::MatrixXcd::Zero(2, 2);
      projector(outcome ? 1 : 0, outcome ? 1 : 0) =
          1 / std::sqrt(outcome ? 1 - p0 : p0);
      QC::Gates::SingleQubitGate<> gate(projector);
      state.ApplyGate(gate, q);
    }
    net.RestoreState();
  }
}

BOOST_AUTO_TEST_CASE(cache_values_topology_limits_and_clone) {
  TensorNetworks::TensorNetwork net(3);
  auto contractor = std::make_shared<TensorNetworks::ForestContractor>();
  net.SetContractor(contractor);
  BOOST_CHECK_SMALL(net.Probability(0) - 1, 1e-15);
  const auto hits = contractor->GetPlanCacheHits();
  QC::Gates::PauliXGate<> x;
  net.AddGate(x, 0);
  BOOST_CHECK_SMALL(net.Probability(0), 1e-15);
  BOOST_CHECK_EQUAL(contractor->GetPlanCacheHits(), hits + 1);
  BOOST_CHECK_SMALL(net.Probability(1) - 1, 1e-15);
  QC::Gates::CNOTGate<> cx;
  net.AddGate(cx, 0, 1);
  BOOST_CHECK_SMALL(net.Probability(1), 1e-15);
  BOOST_CHECK_SMALL(net.Probability(0), 1e-15);
  BOOST_CHECK_SMALL(net.Probability(2) - 1, 1e-15);
  net.Clear();
  BOOST_CHECK_SMALL(net.Probability(0) - 1, 1e-15);
  contractor->SetPlanCacheByteLimit(1);
  BOOST_CHECK_EQUAL(contractor->GetCachedPlanCount(), 0);
  BOOST_CHECK_SMALL(net.Probability(0) - 1, 1e-15);
  BOOST_CHECK_EQUAL(contractor->GetCachedPlanCount(), 0);
  contractor->SetPlanCacheByteLimit(4096);
  net.Probability(0);
  auto clone = std::dynamic_pointer_cast<TensorNetworks::ForestContractor>(
      contractor->Clone());
  BOOST_REQUIRE(clone);
  BOOST_CHECK_EQUAL(clone->GetPlanCacheByteLimit(), 4096);
  BOOST_CHECK_EQUAL(clone->GetCachedPlanCount(), 0);
  contractor->SetPlanCacheByteLimit(0);
  BOOST_CHECK_EQUAL(contractor->GetCachedPlanCount(), 0);
  BOOST_CHECK_SMALL(net.Probability(0) - 1, 1e-15);
}

BOOST_AUTO_TEST_CASE(cache_evicts_old_plans) {
  TensorNetworks::TensorNetwork net(40);
  auto contractor = std::make_shared<TensorNetworks::ForestContractor>();
  contractor->SetPlanCacheByteLimit(2048);
  net.SetContractor(contractor);
  for (size_t q = 0; q < 40; ++q)
    BOOST_CHECK_SMALL(net.Probability(q) - 1, 1e-15);
  BOOST_CHECK_GT(contractor->GetCachedPlanCount(), 0);
  BOOST_CHECK_LT(contractor->GetCachedPlanCount(), 40);
  const auto hits = contractor->GetPlanCacheHits();
  net.Probability(39);
  BOOST_CHECK_EQUAL(contractor->GetPlanCacheHits(), hits + 1);
  net.Probability(0);
  BOOST_CHECK_EQUAL(contractor->GetPlanCacheHits(), hits + 1);
}

BOOST_AUTO_TEST_CASE(prepared_cache_workspace_limits_and_reuse) {
  TensorNetworks::TensorNetwork net(6);
  auto contractor = std::make_shared<TensorNetworks::ForestContractor>();
  net.SetContractor(contractor);
  QC::Gates::HadamardGate<> h;
  QC::Gates::CNOTGate<> cx;
  net.AddGate(h, 0);
  for (size_t q = 1; q < 6; ++q) net.AddGate(cx, (q - 1) / 2, q);
  BOOST_CHECK_SMALL(net.Probability(0) - .5, 1e-12);
  BOOST_CHECK_EQUAL(contractor->GetPreparedPlanCount(), 0);
  BOOST_CHECK_EQUAL(contractor->GetWorkspaceBytes(), 0);
  const size_t orderBytes = contractor->GetCachedPlanBytes();
  // Keep the order cached, with too little space to prepare its metadata.
  contractor->SetPlanCacheByteLimit(orderBytes);
  BOOST_CHECK_SMALL(net.Probability(0) - .5, 1e-12);
  BOOST_CHECK_EQUAL(contractor->GetPreparedPlanCount(), 0);
  BOOST_CHECK_EQUAL(contractor->GetCachedPlanBytes(), orderBytes);
  contractor->SetPlanCacheByteLimit(1024 * 1024);
  BOOST_CHECK_SMALL(net.Probability(0) - .5, 1e-12);
  BOOST_CHECK_EQUAL(contractor->GetPreparedPlanCount(), 1);
  BOOST_CHECK_GT(contractor->GetWorkspaceBytes(), 0);
  BOOST_CHECK_LE(contractor->GetCachedPlanBytes(),
                 contractor->GetPlanCacheByteLimit());
  BOOST_CHECK_LE(contractor->GetWorkspaceBytes(),
                 contractor->GetWorkspaceByteLimit());
  // Alternate queries/plans and overwrite input values while reusing storage.
  for (size_t q = 0; q < 6; ++q)
    for (size_t pass = 0; pass < 3; ++pass)
      BOOST_CHECK_SMALL(net.Probability(q, pass % 2 == 0) - .5, 1e-12);
  const auto workspaceBytes = contractor->GetWorkspaceBytes();
  contractor->SetWorkspaceByteLimit(workspaceBytes);
  BOOST_CHECK_SMALL(net.Probability(0) - .5, 1e-12);
  BOOST_CHECK_EQUAL(contractor->GetWorkspaceBytes(), workspaceBytes);
  contractor->SetWorkspaceByteLimit(1);
  BOOST_CHECK_EQUAL(contractor->GetWorkspaceBytes(), 0);
  for (size_t pass = 0; pass < 3; ++pass)
    BOOST_CHECK_SMALL(net.Probability(0) - .5, 1e-12);
  BOOST_CHECK_EQUAL(contractor->GetWorkspaceBytes(), 0);
  contractor->SetWorkspaceByteLimit(0);
  BOOST_CHECK_SMALL(net.Probability(1) - .5, 1e-12);
  contractor->SetWorkspaceByteLimit(4096);
  BOOST_CHECK_SMALL(net.Probability(0) - .5, 1e-12);
  BOOST_CHECK_GT(contractor->GetWorkspaceBytes(), 0);
  auto clone = std::dynamic_pointer_cast<TensorNetworks::ForestContractor>(
      contractor->Clone());
  BOOST_REQUIRE(clone);
  BOOST_CHECK_EQUAL(clone->GetWorkspaceByteLimit(), 4096);
  BOOST_CHECK_EQUAL(clone->GetWorkspaceBytes(), 0);
  BOOST_CHECK_EQUAL(clone->GetPreparedPlanCount(), 0);
  net.Clear();
  BOOST_CHECK_SMALL(net.Probability(0) - 1, 1e-15);
  BOOST_CHECK_SMALL(net.Probability(0) - 1, 1e-15);
  QC::Gates::PauliXGate<> x;
  net.AddGate(x, 0);
  BOOST_CHECK_SMALL(net.Probability(0), 1e-15);
  BOOST_CHECK_SMALL(net.Probability(0, false) - 1, 1e-15);
  contractor->ClearPlanCache();
  BOOST_CHECK_EQUAL(contractor->GetCachedPlanBytes(), 0);
  BOOST_CHECK_EQUAL(contractor->GetWorkspaceBytes(), 0);
  BOOST_CHECK_EQUAL(contractor->GetPreparedPlanCount(), 0);
}

BOOST_AUTO_TEST_CASE(prepared_metadata_preserves_order_cache_under_pressure) {
  TensorNetworks::TensorNetwork net(8);
  auto contractor = std::make_shared<TensorNetworks::ForestContractor>();
  net.SetContractor(contractor);
  QC::Gates::HadamardGate<> h;
  QC::Gates::CNOTGate<> cx;
  net.AddGate(h, 0);
  for (size_t q = 1; q < 8; ++q) net.AddGate(cx, (q - 1) / 2, q);
  for (size_t q = 0; q < 8; ++q)
    BOOST_CHECK_SMALL(net.Probability(q) - .5, 1e-12);
  const auto orderBytes = contractor->GetCachedPlanBytes();
  BOOST_CHECK_EQUAL(contractor->GetCachedPlanCount(), 8);
  net.Probability(0);
  const auto onePrepared = contractor->GetCachedPlanBytes() - orderBytes;
  BOOST_REQUIRE_GT(onePrepared, 0);
  contractor->SetPlanCacheByteLimit(orderBytes + onePrepared);
  auto hits = contractor->GetPlanCacheHits();
  for (size_t pass = 0; pass < 3; ++pass)
    for (size_t q = 0; q < 8; ++q) {
      BOOST_CHECK_SMALL(net.Probability(q) - .5, 1e-12);
      BOOST_CHECK_EQUAL(contractor->GetPlanCacheHits(), ++hits);
      BOOST_CHECK_EQUAL(contractor->GetCachedPlanCount(), 8);
      BOOST_CHECK_LE(contractor->GetCachedPlanBytes(),
                     contractor->GetPlanCacheByteLimit());
    }
  contractor->SetPlanCacheByteLimit(orderBytes);
  BOOST_CHECK_EQUAL(contractor->GetPreparedPlanCount(), 0);
  BOOST_CHECK_EQUAL(contractor->GetCachedPlanCount(), 8);
  BOOST_CHECK_EQUAL(contractor->GetCachedPlanBytes(), orderBytes);
  for (size_t q = 0; q < 8; ++q) net.Probability(q);
  BOOST_CHECK_EQUAL(contractor->GetPlanCacheHits(), hits + 8);
}

BOOST_AUTO_TEST_CASE(random_nonlocal_complex_networks) {
  std::mt19937_64 rng(90122);
  std::uniform_real_distribution<double> angle(-2, 2);
  QC::Gates::CNOTGate<> cx;
  for (size_t trial = 0; trial < 12; ++trial) {
    const size_t count = 4 + trial % 4;
    TensorNetworks::TensorNetwork net(count);
    auto contractor = std::make_shared<TensorNetworks::ForestContractor>();
    net.SetContractor(contractor);
    net.SetMultithreading(trial % 2 != 0);
    QC::QubitRegister<> state(count);
    state.SetMultithreading(false);
    for (size_t gate = 0; gate < 40; ++gate) {
      const size_t a = rng() % count;
      if (gate % 3 == 2) {
        const size_t b = (a + 1 + rng() % (count - 1)) % count;
        net.AddGate(cx, a, b);
        state.ApplyGate(cx, b, a);
      } else {
        const double theta = angle(rng), phi = angle(rng);
        const Complex phase = std::polar(1.0, phi);
        Eigen::MatrixXcd matrix(2, 2);
        matrix << std::cos(theta), -std::sin(theta) * std::conj(phase),
            std::sin(theta) * phase, std::cos(theta);
        QC::Gates::SingleQubitGate<> rotation(matrix);
        net.AddGate(rotation, a);
        state.ApplyGate(rotation, a);
      }
    }
    for (size_t pass = 0; pass < 2; ++pass)
      for (size_t q = 0; q < count; ++q)
        BOOST_CHECK_SMALL(net.Probability(q) - QubitZero(state, count, q),
                          1e-10);
  }
}
