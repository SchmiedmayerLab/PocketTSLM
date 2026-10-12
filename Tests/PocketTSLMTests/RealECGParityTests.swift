import Foundation
import Hub
import MLX
@testable import PocketTSLM
import XCTest

/// Python↔Swift parity on the real ECG checkpoint and the first ECG-QA test row, the input on
/// which the iOS port once diverged from the Python reference.
///
/// `generate_parity_fixtures.py real` writes the reference: the input leads and a strided subsample
/// of the MLX Python reference's encoder and pipeline outputs. The real-weight tests run only with
/// `POCKETTSLM_INTEGRATION=1`; through Xcode, pass `TEST_RUNNER_POCKETTSLM_INTEGRATION=1` to
/// `xcodebuild test`. They read the checkpoint from `POCKETTSLM_ECG_CHECKPOINT`, else from the demo
/// app's copy that `scripts/fetch_assets.sh` fills, else download it from Hugging Face.
final class RealECGParityTests: XCTestCase {

    private var fixturesURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
    }

    private func fixture(_ name: String) throws -> URL {
        let url = fixturesURL.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Fixture \(name) not found — run generate_parity_fixtures.py first")
        }
        return url
    }

    /// The ECG checkpoint and the reference arrays.
    private func integrationInputs() async throws -> (TSLMCheckpoint, [String: MLXArray]) {
        let environment = ProcessInfo.processInfo.environment
        guard environment["POCKETTSLM_INTEGRATION"] == "1" else {
            throw XCTSkip("Set POCKETTSLM_INTEGRATION=1 to run the real-weight parity tests")
        }
        let reference = try loadArrays(url: fixture("real_ecg_parity.safetensors"))
        if let path = environment["POCKETTSLM_ECG_CHECKPOINT"] {
            return (try TSLMCheckpoint(directory: URL(fileURLWithPath: path)), reference)
        }
        let fetched = fixturesURL.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("UITests/Checkpoints/llama-3.2-1b-ecg-sp-mlx")
        if FileManager.default.fileExists(atPath: fetched.appendingPathComponent("config.json").path) {
            return (try TSLMCheckpoint(directory: fetched), reference)
        }
        let downloaded = try await HubApi().snapshot(
            from: "OpenTSLM/llama-3.2-1b-ecg-sp-mlx",
            matching: ["config.json", "encoder.safetensors", "projector.safetensors"]
        )
        return (try TSLMCheckpoint(directory: downloaded), reference)
    }

    func testEncoderRealWeightsParity() async throws {
        let (checkpoint, reference) = try await integrationInputs()
        try Device.withDefaultDevice(.cpu) {
            let encoder = TransformerCNNEncoder(maxPatches: checkpoint.configuration.encoder.maxPatches)
            try encoder.loadWeights(from: checkpoint.encoderURL)
            let encoded = encoder(try XCTUnwrap(reference["input"]))
            // The strides of `generate_parity_fixtures.py`.
            assertParity(encoded[.stride(by: 1), .stride(by: 10), .stride(by: 8)], try XCTUnwrap(reference["encoded"]), "encoder")
        }
    }

    func testPipelineRealWeightsParity() async throws {
        let (checkpoint, reference) = try await integrationInputs()
        try Device.withDefaultDevice(.cpu) {
            let configuration = checkpoint.configuration
            let pipeline = OpenTSLMSPPipeline(
                hiddenSize: configuration.baseModel.hiddenSize,
                patchSize: configuration.encoder.patchSize,
                maxPatches: configuration.encoder.maxPatches
            )
            try pipeline.loadWeights(encoderURL: checkpoint.encoderURL, projectorURL: checkpoint.projectorURL)
            let input = try XCTUnwrap(reference["input"])
            let leads = (0 ..< input.dim(0)).map { input[$0].asArray(Float.self) }
            let projected = stacked(pipeline.projectTimeSeries(leads), axis: 0)
            assertParity(projected[.stride(by: 1), .stride(by: 10), .stride(by: 64)], try XCTUnwrap(reference["projected"]), "pipeline")
        }
    }
}
