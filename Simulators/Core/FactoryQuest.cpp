/**
 * @file FactoryQuest.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Simulator factory part for the quest backend, including the loading of the
 * quest library.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "FactoryBackends.h"

#define INCLUDED_BY_FACTORY
#include "../Quest/QuestSimulator.h"

namespace Simulators
{

std::shared_ptr<QuestLibSim> SimulatorsFactory::questLibrary = nullptr;
std::atomic_bool SimulatorsFactory::firstTimeQuest = true;

bool SimulatorsFactory::InitQuestLibrary()
{
    if (!questLibrary)
    {
        questLibrary = std::make_shared<QuestLibSim>();
        if (!firstTimeQuest.exchange(false))
            questLibrary->SetMute(true);
        if (questLibrary->Init(
#ifdef _WIN32
                "maestroquest.dll"
#elif defined(__APPLE__)
                "libmaestroquest.dylib"
#else
                "libmaestroquest.so"
#endif
                ))
            return true;
        else
            questLibrary = nullptr;
    }
    return false;
}

bool SimulatorsFactory::IsQuestLibraryAvailable()
{
    return questLibrary && questLibrary->IsValid();
}

std::shared_ptr<QuestLibSim> SimulatorsFactory::GetQuestLibrary()
{
    if (!questLibrary || !questLibrary->IsValid())
        return nullptr;
    return questLibrary;
}

bool SimulatorsFactory::InitQuestLibraryWithMute()
{
    if (!questLibrary)
    {
        questLibrary = std::make_shared<QuestLibSim>();
        firstTimeQuest = false;
        questLibrary->SetMute(true);
        if (questLibrary->Init(
#ifdef _WIN32
                "maestroquest.dll"
#elif defined(__APPLE__)
                "libmaestroquest.dylib"
#else
                "libmaestroquest.so"
#endif
                ))
            return true;
        else
            questLibrary = nullptr;
    }
    return false;
}

namespace FactoryBackends
{

std::unique_ptr<ISimulator> CreateQuestSimulator()
{
    return std::make_unique<Private::QuestSimulator>();
}

} // namespace FactoryBackends
} // namespace Simulators
