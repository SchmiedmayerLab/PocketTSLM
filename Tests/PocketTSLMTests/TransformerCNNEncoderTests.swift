import XCTest
import MLX
import MLXRandom
@testable import PocketTSLM

final class TransformerCNNEncoderTests: XCTestCase {

    // Runs once before any test instance is created — set CPU before the
    // @TaskLocal default device is first evaluated (MLX's GPU init would crash
    // in the SPM test runner which has no Metal bundle).
    override class func setUp() {
        super.setUp()
        Device.setDefault(device: .cpu)
    }

    // Fixtures directory sits next to this file.
    private var fixturesURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
    }

    // MARK: - Shape tests

    func testOutputShape_singleSample() {
        Device.withDefaultDevice(.cpu) {
            let encoder = TransformerCNNEncoder()
            let x = MLXArray.zeros([1, 400])   // [B=1, L=400]
            let out = encoder(x)
            eval(out)
            // N = 400 / 4 = 100 patches, dim = 128
            XCTAssertEqual(out.shape, [1, 100, 128])
        }
    }

    func testOutputShape_batch() {
        Device.withDefaultDevice(.cpu) {
            let encoder = TransformerCNNEncoder()
            let x = MLXArray.zeros([4, 800])   // [B=4, L=800]
            let out = encoder(x)
            eval(out)
            // N = 800 / 4 = 200 patches
            XCTAssertEqual(out.shape, [4, 200, 128])
        }
    }

    func testOutputShape_longerSequence() {
        Device.withDefaultDevice(.cpu) {
            let encoder = TransformerCNNEncoder()
            let x = MLXArray.zeros([2, 2400])  // [B=2, L=2400] → N=600
            let out = encoder(x)
            eval(out)
            XCTAssertEqual(out.shape, [2, 600, 128])
        }
    }

    func testOutputShape_customPatchSize() {
        Device.withDefaultDevice(.cpu) {
            let encoder = TransformerCNNEncoder(patchSize: 8)
            let x = MLXArray.zeros([2, 400])  // N = 400 / 8 = 50
            let out = encoder(x)
            eval(out)
            XCTAssertEqual(out.shape, [2, 50, 128])
        }
    }

    // MARK: - Determinism

    func testDeterminism() {
        Device.withDefaultDevice(.cpu) {
            let encoder = TransformerCNNEncoder()
            // Non-trivial input: ramp from 0 to 1
            let vals = (0 ..< 800).map { Float($0) / 800.0 }
            let x = MLXArray(vals).reshaped(2, 400)

            let out1 = encoder(x)
            eval(out1)
            let out2 = encoder(x)
            eval(out2)

            let diff = (out1 - out2).abs().max(keepDims: false)
            eval(diff)
            XCTAssertEqual(diff.item(Float.self), 0.0,
                "Two identical forward passes must produce identical output")
        }
    }

    // MARK: - Finite values

    func testOutputIsFinite() {
        Device.withDefaultDevice(.cpu) {
            let encoder = TransformerCNNEncoder()
            // Deterministic non-trivial input
            let vals = (0 ..< 400).map { Float($0 % 17) / 16.0 - 0.5 }
            let x = MLXArray(vals).reshaped(1, 400)
            let out = encoder(x)
            eval(out)

            let hasNaN = isNaN(out).any().item(Bool.self)
            let hasInf = isInf(out).any().item(Bool.self)
            XCTAssertFalse(hasNaN, "Output must not contain NaN")
            XCTAssertFalse(hasInf, "Output must not contain Inf")
        }
    }

    // MARK: - Parity with the Python reference
    //
    // `generate_parity_fixtures.py tiny` writes a small encoder with seeded random weights, an
    // input and the output of the MLX Python reference.

    func testNumericalParityWithPython() throws {
        try Device.withDefaultDevice(.cpu) {
            let encoder = TransformerCNNEncoder(
                outputDim: 16, transformerInputDim: 16, numHeads: 2, numLayers: 2, patchSize: 4, ffDim: 32, maxPatches: 16
            )
            try encoder.loadWeights(from: fixturesURL.appendingPathComponent("tiny_encoder.safetensors"))
            let io = try loadArrays(url: fixturesURL.appendingPathComponent("tiny_io.safetensors"))
            assertParity(encoder(try XCTUnwrap(io["input"])), try XCTUnwrap(io["encoded"]), "encoder")
        }
    }
}
