//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import XCTest
import MLX
import MLXFast
@testable import PocketTSLM

/// Checks whether `MLXFast.RoPE` actually applies a custom `freqs` array (the llama3-scaled
/// frequencies) the way mlx-python's `mx.fast.rope` does. On iOS the `ropeType=llama3` fix
/// went live but the generated text didn't change from the default-RoPE output — suggesting
/// the custom-`freqs` RoPE path (never exercised before the fix) isn't taking effect.
///
/// `rope_io.safetensors` (`generate_parity_fixtures.py rope`) holds a fixed input, the llama3 freqs,
/// and the exact RoPE outputs, computed in float64, for the llama3 freqs and for the default base,
/// at offset 3500, where the two differ by about 7.8. float32 RoPE stays within 7e-4 of exact there,
/// though MLX versions differ from each other by up to 1e-3.
final class RoPEParityTests: XCTestCase {
    override func setUpWithError() throws {
        try skipWithoutMetalGPU()
    }


    private var fixturesURL: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
    }

    func testFastRoPEAppliesCustomFreqs() throws {
        let ioURL = fixturesURL.appendingPathComponent("rope_io.safetensors")
        guard FileManager.default.fileExists(atPath: ioURL.path) else {
            throw XCTSkip("rope_io.safetensors missing — run generate_parity_fixtures.py rope")
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

            let llama3VsExact = maxDiff(swiftLlama3, a["out_llama3"]!)
            let defaultVsExact = maxDiff(swiftDefault, a["out_default"]!)
            let llama3VsDefault = maxDiff(swiftLlama3, swiftDefault)
            print("RoPE: swiftLlama3-vs-exact=\(llama3VsExact)  swiftDefault-vs-exact=\(defaultVsExact)  swiftLlama3-vs-swiftDefault=\(llama3VsDefault)")

            // If the custom-freqs path works, swiftLlama3 matches the exact llama3 rope…
            XCTAssertLessThan(llama3VsExact, 1e-3, "MLXFast.RoPE with custom freqs ≠ exact llama3 RoPE")
            XCTAssertLessThan(defaultVsExact, 1e-3, "MLXFast.RoPE ≠ exact default RoPE")
            // …and differs from the default path (proves freqs aren't being ignored).
            XCTAssertGreaterThan(llama3VsDefault, 1.0, "MLXFast.RoPE ignored custom freqs (== default)")
        }
    }
}
