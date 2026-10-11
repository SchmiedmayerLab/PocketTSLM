import XCTest
import MLX
@testable import OpenTSLMKit

final class SoftPromptInterleaverTests: XCTestCase {

    override class func setUp() {
        super.setUp()
        Device.setDefault(device: .cpu)
    }

    func testPadAndInterleaveBatch() {
        Device.withDefaultDevice(.cpu) {
            let sample1 = SoftPromptSample(
                prePrompt: SoftPromptSegment(
                    embeddings: MLXArray(converting: [Double(1), Double(2), Double(3), Double(4), Double(5), Double(6)], [3, 2]),
                    attentionMask: MLXArray(converting: [Double(1), Double(1), Double(0)], [3])
                ),
                timeSeriesText: [
                    SoftPromptSegment(
                        embeddings: MLXArray(converting: [Double(7), Double(8), Double(9), Double(10)], [2, 2]),
                        attentionMask: MLXArray(converting: [Double(1), Double(0)], [2])
                    )
                ],
                timeSeriesEmbeddings: [
                    MLXArray(converting: [Double(11), Double(12), Double(13), Double(14)], [2, 2])
                ],
                postPrompt: SoftPromptSegment(
                    embeddings: MLXArray(converting: [Double(15), Double(16), Double(17), Double(18)], [2, 2]),
                    attentionMask: MLXArray(converting: [Double(1), Double(1)], [2])
                )
            )

            let sample2 = SoftPromptSample(
                prePrompt: SoftPromptSegment(
                    embeddings: MLXArray(converting: [Double(21), Double(22)], [1, 2]),
                    attentionMask: MLXArray(converting: [Double(1)], [1])
                ),
                timeSeriesText: [
                    SoftPromptSegment(
                        embeddings: MLXArray(converting: [Double(23), Double(24)], [1, 2]),
                        attentionMask: MLXArray(converting: [Double(1)], [1])
                    )
                ],
                timeSeriesEmbeddings: [
                    MLXArray(converting: [Double(25), Double(26)], [1, 2])
                ],
                postPrompt: SoftPromptSegment(
                    embeddings: MLXArray(converting: [Double(27), Double(28)], [1, 2]),
                    attentionMask: MLXArray(converting: [Double(1)], [1])
                )
            )

            let batch = SoftPromptInterleaver.padAndInterleaveBatch([sample1, sample2])
            eval(batch.inputsEmbeds)
            eval(batch.attentionMask)

            XCTAssertEqual(batch.inputsEmbeds.shape, [2, 7, 2])
            XCTAssertEqual(batch.attentionMask.shape, [2, 7])

            let expected1 = MLXArray(converting: [
                Double(1), Double(2),
                Double(3), Double(4),
                Double(7), Double(8),
                Double(11), Double(12),
                Double(13), Double(14),
                Double(15), Double(16),
                Double(17), Double(18)
            ], [7, 2])
            let expected2 = MLXArray(converting: [
                Double(21), Double(22),
                Double(23), Double(24),
                Double(25), Double(26),
                Double(27), Double(28)
            ], [4, 2])
            let expectedEmbeds = stacked([expected1, concatenated([expected2, MLXArray.zeros([3, 2])], axis: 0)], axis: 0)

            let expectedMask1 = MLXArray(converting: [Double(1), Double(1), Double(1), Double(1), Double(1), Double(1), Double(1)], [7])
            let expectedMask2 = MLXArray(converting: [Double(1), Double(1), Double(1), Double(1)], [4])
            let expectedMasks = stacked([expectedMask1, concatenated([expectedMask2, MLXArray.zeros([3])], axis: 0)], axis: 0)

            let embedDiff = (batch.inputsEmbeds - expectedEmbeds).abs().max(keepDims: false)
            let maskDiff = (batch.attentionMask - expectedMasks).abs().max(keepDims: false)
            eval(embedDiff)
            eval(maskDiff)

            XCTAssertEqual(embedDiff.item(Float.self), 0.0)
            XCTAssertEqual(maskDiff.item(Float.self), 0.0)
        }
    }
}