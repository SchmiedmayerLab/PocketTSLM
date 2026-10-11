//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation
import HealthKit
import Spezi
import SpeziHealthKit

/// One Apple Watch ECG recording, as read from HealthKit.
///
/// Apple Watch records a single lead (`appleWatchSimilarToLeadI`) at ~512 Hz, whereas
/// OpenTSLM-SP's ECG checkpoint was trained on 12-lead PTB-XL at 100 Hz. PocketTSLM's
/// `ECG.twelveLeads` resamples the lead and copies it to all 12 leads.
struct ElectrocardiogramData {
    let startDate: Date
    let endDate: Date
    let classification: String
    let symptomsStatus: String
    let averageHeartRate: Double?
    let samplingFrequency: Double?
    let voltages: [Double]
}

/// Reads Apple Watch ECG recordings from HealthKit so they can be fed to the on-device
/// model. Restored from the pre-`f7d5d5e` HealthKit path, trimmed to the ECG type only.
class HealthKitECGFetcher: DefaultInitializable, Module, EnvironmentAccessible {
    required init() { }

    /// Requests read access to the electrocardiogram type. No write access is requested,
    /// and no other HealthKit type is read.
    func askForAuthorization(_ healthKit: HealthKit) async throws {
        try await healthKit.askForAuthorization(
            for: .init(read: [HKObjectType.electrocardiogramType()], write: [])
        )
    }

    /// Most recent ECG recording with its voltage series, or `nil` if none exists.
    func fetchLatestElectrocardiogram(_ healthKit: HealthKit) async throws -> ElectrocardiogramData? {
        guard let sample = try await fetchLatestSample(healthKit) else {
            return nil
        }
        let voltages = try await fetchVoltageMeasurements(for: sample, healthKit: healthKit)
        return ElectrocardiogramData(
            startDate: sample.startDate,
            endDate: sample.endDate,
            classification: Self.describe(sample.classification),
            symptomsStatus: String(describing: sample.symptomsStatus),
            averageHeartRate: sample.averageHeartRate?.doubleValue(for: .count().unitDivided(by: .minute())),
            samplingFrequency: sample.samplingFrequency?.doubleValue(for: .hertz()),
            voltages: voltages
        )
    }

    /// `String(describing:)` on `HKElectrocardiogram.Classification` yields
    /// "HKElectrocardiogramClassification(rawValue: 1)", which is unreadable in the UI and
    /// in the prompt's ground-truth summary. Map to Apple's own names instead.
    private static func describe(_ classification: HKElectrocardiogram.Classification) -> String {
        switch classification {
        case .notSet: return "not set"
        case .sinusRhythm: return "sinus rhythm"
        case .atrialFibrillation: return "atrial fibrillation"
        case .inconclusiveLowHeartRate: return "inconclusive (low heart rate)"
        case .inconclusiveHighHeartRate: return "inconclusive (high heart rate)"
        case .inconclusivePoorReading: return "inconclusive (poor reading)"
        case .inconclusiveOther: return "inconclusive"
        case .unrecognized: return "unrecognized"
        @unknown default: return "unknown"
        }
    }

    private func fetchLatestSample(_ healthKit: HealthKit) async throws -> HKElectrocardiogram? {
        try await withCheckedThrowingContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: HKObjectType.electrocardiogramType(),
                predicate: nil,
                limit: 1,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]
            ) { _, samples, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume(returning: (samples as? [HKElectrocardiogram])?.first)
            }
            healthKit.healthStore.execute(query)
        }
    }

    private func fetchVoltageMeasurements(
        for sample: HKElectrocardiogram,
        healthKit: HealthKit
    ) async throws -> [Double] {
        try await withCheckedThrowingContinuation { continuation in
            var voltages: [Double] = []
            var didResume = false

            let query = HKElectrocardiogramQuery(electrocardiogram: sample) { _, measurement, done, error in
                if let error, !didResume {
                    didResume = true
                    continuation.resume(throwing: error)
                    return
                }
                if let measurement,
                   let voltageQuantity = measurement.quantity(for: .appleWatchSimilarToLeadI) {
                    voltages.append(voltageQuantity.doubleValue(for: .volt()))
                }
                if done, !didResume {
                    didResume = true
                    continuation.resume(returning: voltages)
                }
            }
            healthKit.healthStore.execute(query)
        }
    }
}
