//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import HealthKit
import Spezi
import SpeziHealthKit
import SpeziLLM
import SpeziLLMLocal
import SwiftUI


class PocketTSLMAppDelegate: SpeziAppDelegate {
    override var configuration: Configuration {
        Configuration(standard: HealthyStandard()) {
            LLMRunner {
                LLMLocalPlatform()
            }
            HealthDataInterpreter()
            OpenTSLMInferenceService()
            HealthKitECGFetcher()

            if HKHealthStore.isHealthDataAvailable() {
                HealthKit()
            }
        }
    }
}

actor HealthyStandard: Standard, HealthKitConstraint {
    func handleNewSamples<Sample>(
        _ addedSamples: some Collection<Sample>,
        ofType sampleType: SampleType<Sample>
    ) async { }

    func handleDeletedObjects<Sample>(
        _ deletedObjects: some Collection<HKDeletedObject>,
        ofType sampleType: SampleType<Sample>
    ) async { }
}
