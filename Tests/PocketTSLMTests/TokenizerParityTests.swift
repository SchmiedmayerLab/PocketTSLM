import XCTest
import Tokenizers
@testable import PocketTSLM

/// Python↔Swift tokenizer parity. The app embeds each prompt segment with
/// `tokenizer.encode(text:, addSpecialTokens: true)` (swift-transformers, via MLXLLM).
/// Encoder and projector matched the reference, so a divergence from the very
/// first generated token would most plausibly come from the text tokens (IDs / BOS).
///
/// `tokenizer_ref.json` holds the HF (Python) token IDs for each real ECG prompt segment.
/// We load the same Llama-3.2-1B tokenizer with swift-transformers and assert identical IDs.
final class TokenizerParityTests: XCTestCase {

    private struct Segment: Decodable { let name: String; let text: String; let ids: [Int] }

    private var fixturesURL: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
    }

    /// Folder with Meta's Llama-3.2-1B tokenizer files: `POCKETTSLM_TOKENIZER_DIR`, or else the
    /// Hugging Face cache's snapshot of the base-model revision the checkpoints pin.
    private var modelFolder: URL {
        if let path = ProcessInfo.processInfo.environment["POCKETTSLM_TOKENIZER_DIR"] {
            return URL(fileURLWithPath: path)
        }
        guard let baseModel = try? TSLMCheckpoint(directory: fixturesURL.appendingPathComponent("Checkpoints/sleep-edf"))
            .configuration.baseModel else {
            return fixturesURL
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
            ".cache/huggingface/hub/models--\(baseModel.id.replacingOccurrences(of: "/", with: "--"))/snapshots/\(baseModel.revision ?? "main")"
        )
    }

    func testTokenizerParityWithPython() async throws {
        let refURL = fixturesURL.appendingPathComponent("tokenizer_ref.json")
        guard FileManager.default.fileExists(atPath: refURL.path) else {
            throw XCTSkip("tokenizer_ref.json missing — run the Python dump first")
        }
        guard FileManager.default.fileExists(atPath: modelFolder.appendingPathComponent("tokenizer.json").path) else {
            throw XCTSkip("tokenizer files not found at \(modelFolder.path)")
        }

        let segments = try JSONDecoder().decode([Segment].self, from: Data(contentsOf: refURL))
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)

        var mismatches: [String] = []
        for seg in segments {
            let swiftIDs = tokenizer.encode(text: seg.text, addSpecialTokens: true)
            if swiftIDs != seg.ids {
                mismatches.append(seg.name)
                let n = min(swiftIDs.count, seg.ids.count)
                let firstDiff = (0 ..< n).first { swiftIDs[$0] != seg.ids[$0] }
                print("MISMATCH \(seg.name): swift n=\(swiftIDs.count) py n=\(seg.ids.count) "
                    + "firstDiff@\(firstDiff.map(String.init) ?? "len") "
                    + "swift5=\(Array(swiftIDs.prefix(5))) py5=\(Array(seg.ids.prefix(5)))")
            } else {
                print("OK \(seg.name): \(seg.ids.count) tokens")
            }
        }
        XCTAssertTrue(mismatches.isEmpty, "Tokenizer ID mismatch in segments: \(mismatches)")
    }
}
