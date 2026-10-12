//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation
import MLX

final class OpenTSLMSPPipeline {

    let encoder: TransformerCNNEncoder
    let projector: MLPProjector
    let patchSize: Int

    /// The checkpoint's encoder and projector, with their weights.
    init(checkpoint: TSLMCheckpoint) throws {
        let configuration = checkpoint.configuration
        patchSize = configuration.encoder.patchSize
        encoder = TransformerCNNEncoder(
            transformerInputDim: configuration.encoder.embedDim,
            patchSize: patchSize,
            maxPatches: configuration.encoder.maxPatches
        )
        projector = MLPProjector(inputDim: configuration.encoder.embedDim, outputDim: configuration.baseModel.hiddenSize)
        try encoder.loadWeights(from: checkpoint.encoderURL)
        try projector.loadWeights(from: checkpoint.projectorURL)
    }

    func projectTimeSeries(_ seriesBatch: [[Float]]) -> [MLXArray] {
        guard !seriesBatch.isEmpty else { return [] }

        let maxLength = paddedLength(for: seriesBatch)
        var paddedRows: [[Float]] = []
        paddedRows.reserveCapacity(seriesBatch.count)

        for series in seriesBatch {
            if series.count == maxLength {
                paddedRows.append(series)
                continue
            }
            var row = Array<Float>(repeating: 0, count: maxLength)
            row.replaceSubrange(0 ..< series.count, with: series)
            paddedRows.append(row)
        }

        let flat = paddedRows.flatMap { $0 }.map(Double.init)
        let input = MLXArray(converting: flat, [seriesBatch.count, maxLength])
        let encoded = encoder(input)
        let projected = projector(encoded)
        eval(projected)

        var result: [MLXArray] = []
        result.reserveCapacity(seriesBatch.count)
        for i in 0 ..< seriesBatch.count {
            result.append(projected[i])
        }
        return result
    }

    private func paddedLength(for seriesBatch: [[Float]]) -> Int {
        let currentMax = seriesBatch.map(\.count).max() ?? 0
        let remainder = currentMax % patchSize
        if remainder == 0 {
            return currentMax
        }
        return currentMax + (patchSize - remainder)
    }
}
