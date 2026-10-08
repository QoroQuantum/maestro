#include "../maestrolib/Interface.h"
#include "../maestrolib/Maestro.h"

#include <iostream>

#if MAESTRO_EXPECT_COMPOSER != defined(MAESTRO_ENABLE_COMPOSER)
#error Composer integration settings did not propagate from the Maestro target
#endif

int main()
{
    auto *maestro = static_cast<Maestro *>(GetMaestroObjectWithMute());
    const auto handle = CreateSimpleSimulator(2);
    const auto network = maestro->GetSimpleSimulator(handle);
#ifdef MAESTRO_ENABLE_COMPOSER
    const bool correctNetwork = dynamic_cast<Network::SimpleNetwork<> *>(network.get()) != nullptr;
#else
    const bool correctNetwork = dynamic_cast<Network::SimpleDisconnectedNetwork<> *>(network.get()) != nullptr;
#endif
    DestroySimpleSimulator(handle);
    if (!correctNetwork)
    {
        std::cerr << "The Maestro library created the wrong network for its Composer integration setting\n";
        return 1;
    }
    return 0;
}
