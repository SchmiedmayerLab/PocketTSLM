import XCTest
import MLX
import MLXFast
@testable import OpenTSLMKit

/// Checks whether `MLXFast.RoPE` actually applies a custom `freqs` array (the llama3-scaled
/// frequencies) the way mlx-python's `mx.fast.rope` does. On iOS the `ropeType=llama3` fix
/// went live but the generated text didn't change from the default-RoPE output — suggesting
/// the custom-`freqs` RoPE path (never exercised before the fix) isn't taking effect.
///
/// `rope_io.safetensors` holds a fixed input, the llama3 freqs, and the Python rope outputs
/// (llama3 via freqs, and default via base) at offset 3500 (where the two differ by ~7.8).
final class RoPEParityTests: XCTestCase {

    override class func setUp() {
        super.setUp()
        Device.setDefault(device: .cpu)
    }

    private var fixturesURL: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
    }

    func testFastRoPEAppliesCustomFreqs() throws {
        let ioURL = fixturesURL.appendingPathComponent("rope_io.safetensors")
        guard FileManager.default.fileExists(atPath: ioURL.path) else {
            throw XCTSkip("rope_io.safetensors missing — run the Python rope generator first")
        }
        try Device.withDefaultDevice(.cpu) {
            let a = try loadArrays(url: ioURL)
            let x = a["x"]!
            let llama3Freqs = a["llama3_freqs"]!
            let meta = a["meta"]!.asArray(Int32.self)
            let dims = Int(meta[0]); let offset = Int(meta[1])

            // Mirror EmbeddingLlamaModel's DynamicNTKScalingRoPE: base=nil + custom freqs.
            let swiftLlama3 = MLXFast.RoPE(
                x, dimensions: dims, traditional: false, base: nil, scale: 1.0,
                offset: offset, freqs: llama3Freqs)
            // Default path: base set, no freqs.
            let swiftDefault = MLXFast.RoPE(
                x, dimensions: dims, traditional: false, base: 500000.0, scale: 1.0,
                offset: offset, freqs: nil)
            eval(swiftLlama3, swiftDefault)

            func maxDiff(_ p: MLXArray, _ q: MLXArray) -> Float {
                let d = (p - q).abs().max(keepDims: false); eval(d); return d.item(Float.self)
            }

            let llama3VsPy = maxDiff(swiftLlama3, a["out_llama3"]!)
            let defaultVsPy = maxDiff(swiftDefault, a["out_default"]!)
            let llama3VsDefault = maxDiff(swiftLlama3, swiftDefault)
            print("RoPE: swiftLlama3-vs-python=\(llama3VsPy)  swiftDefault-vs-python=\(defaultVsPy)  swiftLlama3-vs-swiftDefault=\(llama3VsDefault)")

            // If the custom-freqs path works, swiftLlama3 matches Python's llama3 rope…
            XCTAssertLessThan(llama3VsPy, 1e-3, "MLXFast.RoPE custom freqs ≠ Python mx.fast.rope llama3")
            XCTAssertLessThan(defaultVsPy, 1e-3, "MLXFast.RoPE default ≠ Python default")
            // …and differs from the default path (proves freqs aren't being ignored).
            XCTAssertGreaterThan(llama3VsDefault, 1.0, "MLXFast.RoPE ignored custom freqs (== default)")
        }
    }
}
