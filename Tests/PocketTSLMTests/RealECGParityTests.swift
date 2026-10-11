import XCTest
import MLX
@testable import OpenTSLMKit

/// Per-component Python↔Swift parity on the REAL on-device ECG weights and the REAL
/// ECG-QA CoT sample (test idx 0) — the pipeline that diverged between the iOS Swift
/// port ("no") and the pytorch/mlx-python reference ("yes").
///
/// Fixtures are produced by `generate_real_parity_fixtures.py` (run it first). Each test
/// loads the same weights + input the reference used and asserts the Swift output matches,
/// localizing the divergence: a component that fails here is where the port drifts; if all
/// pass, the divergence is downstream in the LLM-from-embeddings path (not in this package).
final class RealECGParityTests: XCTestCase {

    override class func setUp() {
        super.setUp()
        Device.setDefault(device: .cpu)
    }

    private var fixturesURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
    }

    /// Bundled ECG encoder pos_embed is [1, 1024, 128].
    private let maxPatches = 1024

    private func fixture(_ name: String) throws -> URL {
        let url = fixturesURL.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Fixture \(name) not found — run generate_real_parity_fixtures.py first")
        }
        return url
    }

    private func assertParity(_ swiftOut: MLXArray, _ refOut: MLXArray, _ label: String,
                              maxTol: Float = 1e-4, meanTol: Float = 1e-5) {
        eval(swiftOut)
        XCTAssertEqual(swiftOut.shape, refOut.shape, "\(label): shape mismatch")
        let absDiff = (swiftOut - refOut).abs()
        eval(absDiff)
        let maxDiff = absDiff.max(keepDims: false).item(Float.self)
        let meanDiff = absDiff.mean(keepDims: false).item(Float.self)
        print("\(label) parity: max_diff=\(maxDiff) mean_diff=\(meanDiff)")
        XCTAssertLessThan(maxDiff, maxTol, "\(label): max abs diff \(maxDiff) exceeds \(maxTol)")
        XCTAssertLessThan(meanDiff, meanTol, "\(label): mean abs diff \(meanDiff) exceeds \(meanTol)")
    }

    func testEncoderRealWeightsParity() throws {
        let weightsURL = try fixture("real_encoder_weights.safetensors")
        let ioURL = try fixture("real_encoder_io.safetensors")
        try Device.withDefaultDevice(.cpu) {
            let encoder = TransformerCNNEncoder(maxPatches: maxPatches)
            try encoder.loadWeights(from: weightsURL)
            let io = try loadArrays(url: ioURL)
            assertParity(encoder(io["input"]!), io["output"]!, "encoder")
        }
    }

    func testProjectorRealWeightsParity() throws {
        let weightsURL = try fixture("real_projector_weights.safetensors")
        let ioURL = try fixture("real_projector_io.safetensors")
        try Device.withDefaultDevice(.cpu) {
            let projector = MLPProjector(inputDim: 128, outputDim: 2048)
            try projector.loadWeights(from: weightsURL)
            let io = try loadArrays(url: ioURL)
            assertParity(projector(io["input"]!), io["output"]!, "projector")
        }
    }

    func testPipelineRealWeightsParity() throws {
        let encURL = try fixture("real_encoder_weights.safetensors")
        let projURL = try fixture("real_projector_weights.safetensors")
        let ioURL = try fixture("real_pipeline_io.safetensors")
        try Device.withDefaultDevice(.cpu) {
            let pipeline = OpenTSLMSPPipeline(hiddenSize: 2048, patchSize: 4, maxPatches: maxPatches)
            try pipeline.loadWeights(encoderURL: encURL, projectorURL: projURL)

            let io = try loadArrays(url: ioURL)
            let input = io["input"]!          // [12, 1000]
            let leadCount = input.dim(0)
            let length = input.dim(1)
            let flat = input.asArray(Float.self)
            let rows: [[Float]] = (0 ..< leadCount).map { r in
                Array(flat[(r * length) ..< ((r + 1) * length)])
            }

            let projected = pipeline.projectTimeSeries(rows)   // [MLXArray] of [250, 2048]
            let stacked = stacked(projected, axis: 0)          // [12, 250, 2048]
            assertParity(stacked, io["output"]!, "pipeline(encoder→projector)")
        }
    }

    /// SoftPromptInterleaver assembly parity: reconstruct the sample structure from the
    /// fixture (a 12-segment sample + a shorter one, with intra-segment padding) and check
    /// padAndInterleaveBatch reproduces the Python reference layout/padding exactly.
    func testInterleaverParity() throws {
        let ioURL = try fixture("interleaver_io.safetensors")
        try Device.withDefaultDevice(.cpu) {
            let a = try loadArrays(url: ioURL)
            let tsCounts = a["ts_counts"]!.asArray(Int32.self).map(Int.init)

            var samples: [SoftPromptSample] = []
            for (si, k) in tsCounts.enumerated() {
                let pre = SoftPromptSegment(embeddings: a["s\(si)_pre_emb"]!,
                                            attentionMask: a["s\(si)_pre_mask"]!)
                var tstext: [SoftPromptSegment] = []
                var tsemb: [MLXArray] = []
                for i in 0 ..< k {
                    tstext.append(SoftPromptSegment(embeddings: a["s\(si)_tt\(i)_emb"]!,
                                                    attentionMask: a["s\(si)_tt\(i)_mask"]!))
                    tsemb.append(a["s\(si)_te\(i)"]!)
                }
                let post = SoftPromptSegment(embeddings: a["s\(si)_post_emb"]!,
                                             attentionMask: a["s\(si)_post_mask"]!)
                samples.append(SoftPromptSample(prePrompt: pre, timeSeriesText: tstext,
                                                timeSeriesEmbeddings: tsemb, postPrompt: post))
            }

            let batch = SoftPromptInterleaver.padAndInterleaveBatch(samples)
            assertParity(batch.inputsEmbeds, a["out_embeds"]!, "interleaver.embeds")
            assertParity(batch.attentionMask, a["out_mask"]!, "interleaver.mask")
        }
    }
}
