//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation
import MLX
import MLXLLM
import MLXRandom
@testable import PocketTSLM
import XCTest

/// Checks the patched Llama against the stock `MLXLLM.LlamaModel` of the same mlx-swift-examples
/// revision, on a tiny model with default RoPE, where the RoPE patch does not apply.
final class EmbeddingLlamaModelTests: XCTestCase {
    static let configuration = Data("""
        {
          "model_type": "llama",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "rms_norm_eps": 1e-5,
          "vocab_size": 256,
          "rope_theta": 10000,
          "tie_word_embeddings": true
        }
        """.utf8)

    static let tokens = MLXArray([1, 17, 42, 255, 3, 99, 128, 7] as [Int32], [1, 8])

    /// The stock model with seeded random weights, and the patched model with the same weights.
    func models() throws -> (stock: LlamaModel, patched: EmbeddingLlamaModel) {
        MLXRandom.seed(42)
        let stock = LlamaModel(try JSONDecoder().decode(LlamaConfiguration.self, from: Self.configuration))
        let patched = EmbeddingLlamaModel(try JSONDecoder().decode(EmbeddingLlamaConfiguration.self, from: Self.configuration))
        _ = try patched.update(parameters: stock.parameters(), verify: .all)
        eval(stock, patched)
        return (stock, patched)
    }

    func testTokenPathMatchesStockLlama() throws {
        let (stock, patched) = try models()
        let expected = stock(Self.tokens, cache: nil)
        let logits = patched(Self.tokens, cache: nil)
        XCTAssertEqual(logits.shape, [1, 8, 256])
        XCTAssertTrue(arrayEqual(logits, expected).item(Bool.self))
    }

    func testEmbeddingPathMatchesTokenPath() throws {
        let (_, patched) = try models()
        let expected = patched(Self.tokens, cache: patched.makeCache())
        let logits = try patched(nil, cache: patched.makeCache(), inputEmbedding: patched.tokenEmbeddings(Self.tokens))
        XCTAssertTrue(arrayEqual(logits, expected).item(Bool.self))
    }
}
