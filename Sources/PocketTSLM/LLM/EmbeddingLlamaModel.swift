//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2024 Apple Inc.
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

// A copy of mlx-swift-examples `Libraries/MLXLLM/Models/Llama.swift` at 2.29.1
// (9bff95ca5f0b9e8c021acc4d71a2bbe4a7441631). Upstream's building blocks (`Attention`, `MLP`,
// `TransformerBlock`, `LlamaModelInner`) are private, so only a copy can change the forward pass.
// The code is upstream's, with the types renamed `EmbeddingLlama*` so they do not collide with
// `MLXLLM.LlamaModel`, apart from these functional changes, which `// PATCH(pockettslm):` marks:
//  1. Embedding input: `EmbeddingLlamaModelInner` uses a given `inputEmbedding` in place of the
//     token lookup, and `EmbeddingLlamaModel` exposes it and the token-embedding lookup.
//  2. RoPE: the rope type is read from `rope_scaling["rope_type"]` as well as `["type"]`, and
//     integer values are accepted, so Llama 3.x configs get the llama3 frequency scaling that
//     upstream silently skips.
// `registerModelTypes()` at the end is PocketTSLM's own.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

// port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/llama.py

func computeBaseFrequency(
    base: Float, dims: Int, ropeType: String, ropeScaling: [String: StringOrNumber]?
)
    -> Float
{
    if ropeType != "llama3" {
        return base
    }

    guard let ropeScaling = ropeScaling else {
        return base
    }

    guard case .float(let factor) = ropeScaling["factor"],
        case .float(let lowFreqFactor) = ropeScaling["low_freq_factor"] ?? .float(1.0),
        case .float(let highFreqFactor) = ropeScaling["high_freq_factor"] ?? .float(4.0),
        case .float(let oldContextLen) = ropeScaling["original_max_position_embeddings"]
            ?? .float(8192)
    else {
        return base
    }

    let lowFreqWavelen = oldContextLen / lowFreqFactor
    let highFreqWavelen = oldContextLen / highFreqFactor

    let freqs = (0 ..< dims).compactMap { index -> Float? in
        if index % 2 == 0 {
            return pow(base, Float(index) / Float(dims))
        }
        return nil
    }

    let newBaseFreqs = freqs.map { freq -> Float in
        let wavelen = 2 * .pi / freq
        let smooth = max(
            0, min(1, (wavelen - highFreqWavelen) / (lowFreqWavelen - highFreqWavelen)))
        return freq * ((1 - smooth) * factor + smooth)
    }

    return newBaseFreqs.reduce(0, +) / Float(newBaseFreqs.count)
}

private class DynamicNTKScalingRoPE: Module {
    let dims: Int
    let maxPositionEmbeddings: Int
    let traditional: Bool
    var base: Float?
    let scale: Float
    let ropeType: String
    let ropeScaling: [String: StringOrNumber]?
    var freqs: MLXArray?

    init(
        dims: Int,
        maxPositionEmbeddings: Int?,
        traditional: Bool = false,
        base: Float = 10000,
        scale: Float = 1.0,
        ropeType: String = "default",
        ropeScaling: [String: StringOrNumber]? = nil
    ) {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings ?? 2048
        self.traditional = traditional
        self.base = base
        self.scale = scale
        self.ropeType = ropeType
        self.ropeScaling = ropeScaling
        super.init()
        computeFreqs()
    }

    private func computeFreqs() {
        if ropeType != "llama3" {
            freqs = nil
            return
        }

        // PATCH(pockettslm): MLXLMCommon's StringOrNumber decodes integers (e.g.
        // `original_max_position_embeddings: 8192`) as `.int`, not `.float`. The upstream
        // `case .float = …` pattern match therefore fails on that field, the whole guard fails,
        // and it silently falls back to default RoPE — skipping llama3 scaling even when
        // ropeType == "llama3". Accept both int and float.
        func number(_ value: StringOrNumber?) -> Float? {
            switch value {
            case .float(let f): return f
            case .int(let i): return Float(i)
            default: return nil
            }
        }

        guard let ropeScaling = ropeScaling,
            let factor = number(ropeScaling["factor"]),
            let base
        else {
            freqs = nil
            return
        }
        let lowFreqFactor = number(ropeScaling["low_freq_factor"]) ?? 1.0
        let highFreqFactor = number(ropeScaling["high_freq_factor"]) ?? 4.0
        let oldContextLen = number(ropeScaling["original_max_position_embeddings"]) ?? 8192

        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor

        let indices = MLXArray(stride(from: 0, to: dims, by: 2))
        var frequencies = MLX.pow(base, indices / Float(dims))
        let wavelens = 2 * Float.pi * frequencies

        frequencies = MLX.where(
            wavelens .> MLXArray(lowFreqWavelen), frequencies * factor, frequencies)
        let isMediumFreq = MLX.logicalAnd(
            wavelens .> MLXArray(highFreqWavelen),
            wavelens .< MLXArray(lowFreqWavelen)
        )
        let smoothFactors =
            (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
        let smoothFreqs = frequencies / ((1 - smoothFactors) / factor + smoothFactors)

        freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
        self.base = nil
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: base,
            scale: scale,
            offset: offset,
            freqs: freqs
        )
    }
}

private class Attention: Module {

    let args: EmbeddingLlamaConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: DynamicNTKScalingRoPE

    init(_ args: EmbeddingLlamaConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)

        self.rope = DynamicNTKScalingRoPE(
            dims: headDim,
            maxPositionEmbeddings: args.maxPositionEmbeddings,
            traditional: args.ropeTraditional,
            base: args.ropeTheta,
            scale: 1.0,
            ropeType: {
                // PATCH(pockettslm): HF Llama-3.x configs key this as `rope_type` (e.g.
                // "llama3"); older configs used `type`. Upstream mlx-swift only checks `type`,
                // so for a Llama-3.2 `rope_scaling` it silently falls back to "default" and
                // skips the llama3 frequency scaling — diverging from mlx_lm / transformers.
                let ropeScalingType = args.ropeScaling?["rope_type"] ?? args.ropeScaling?["type"]
                if case .string(let value) = ropeScalingType {
                    return value
                } else {
                    return "default"
                }
            }(),
            ropeScaling: args.ropeScaling)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // Prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

private class MLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: EmbeddingLlamaConfiguration) {
        self._gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        self._down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
        self._up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activation = silu(gate(x))
        return down(activation * up(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "mlp") var mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: EmbeddingLlamaConfiguration) {
        self._attention.wrappedValue = Attention(args)
        self._mlp.wrappedValue = MLP(args)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

private class EmbeddingLlamaModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [TransformerBlock]
    let norm: RMSNorm

    init(_ args: EmbeddingLlamaConfiguration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { _ in TransformerBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    // PATCH(pockettslm): an optional `inputEmbedding` replaces the token-embedding lookup. Llama,
    // unlike Gemma, does not scale its embeddings, so there is nothing else to reconcile.
    func callAsFunction(
        _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbedding {
            h = inputEmbedding
        } else if let inputs {
            h = embedTokens(inputs)
        } else {
            fatalError("EmbeddingLlamaModelInner requires either inputs or inputEmbedding")
        }

        let mask = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

/// Model for Llama and Mistral model types.
public class EmbeddingLlamaModel: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: EmbeddingLlamaModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: EmbeddingLlamaConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = EmbeddingLlamaModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(inputs, cache: cache, inputEmbedding: nil)
    }

    // PATCH(pockettslm): one forward pass for token and embedding inputs.
    private func forward(_ inputs: MLXArray?, cache: [KVCache]?, inputEmbedding: MLXArray?) -> MLXArray {
        let out = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Remove unused precomputed rotary frequencies
        weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        // some models allow the system role and some do not -- this is enforced
        // by the chat template (code).
        do {
            let probe = [
                [
                    "role": "system",
                    "content": "test",
                ]
            ]
            _ = try tokenizer.applyChatTemplate(messages: probe)
            return DefaultMessageGenerator()
        } catch {
            return NoSystemMessageGenerator()
        }
    }

    // PATCH(pockettslm): the token-embedding lookup, for building embedding inputs.
    /// Token-embedding lookup, exposed so callers can build interleaved
    /// soft-prompt sequences (text token embeddings + projected time-series
    /// embeddings) before priming the decoder via ``inputEmbedding``.
    /// `ids` is an integer index array, e.g. shape `[B, T]`; result is `[B, T, hidden]`.
    public func tokenEmbeddings(_ ids: MLXArray) -> MLXArray {
        model.embedTokens(ids)
    }
}

public struct EmbeddingLlamaConfiguration: Codable, Sendable {

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int?
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool = true
    var attentionBias: Bool = false
    var mlpBias: Bool = false

    public init(
        hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int, attentionHeads: Int,
        headDimensions: Int? = nil, rmsNormEps: Float, vocabularySize: Int, kvHeads: Int,
        maxPositionEmbeddings: Int? = nil, ropeTheta: Float = 10_000, ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil, tieWordEmbeddings: Bool = true,
        attentionBias: Bool = false, mlpBias: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
    }

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        if let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            self.ropeTheta = ropeTheta
        }
        if let ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional)
        {
            self.ropeTraditional = ropeTraditional
        }
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        if let tieWordEmbeddings = try container.decodeIfPresent(
            Bool.self, forKey: .tieWordEmbeddings)
        {
            self.tieWordEmbeddings = tieWordEmbeddings
        }
        if let attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) {
            self.attentionBias = attentionBias
        }
        if let mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) {
            self.mlpBias = mlpBias
        }

        if let ropeScaling {
            if ropeScaling["factor"] == nil {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling, in: container,
                    debugDescription: "rope_scaling must contain 'factor'")
            }
            if let ropeType = ropeScaling["type"] ?? ropeScaling["rope_type"] {
                if case .string = ropeType {
                    let options = [
                        StringOrNumber.string("linear"), StringOrNumber.string("dynamic"),
                        StringOrNumber.string("llama3"),
                    ]
                    if !options.contains(ropeType) {
                        throw DecodingError.dataCorruptedError(
                            forKey: .ropeScaling, in: container,
                            debugDescription:
                                "rope_scaling 'type' currently only supports 'linear', 'dynamic', or 'llama3'"
                        )
                    }
                }
            } else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling, in: container,
                    debugDescription: "rope_scaling must contain either 'type' or 'rope_type'")
            }
        }
    }
}

// MARK: - LoRA

extension EmbeddingLlamaModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}

// PATCH(pockettslm): a forward pass from tokens or embeddings, for decoding primed with a soft prompt.
extension EmbeddingLlamaModel {
    func callAsFunction(_ inputs: MLXArray?, cache: [KVCache], inputEmbedding: MLXArray?) -> MLXArray {
        forward(inputs, cache: cache, inputEmbedding: inputEmbedding)
    }
}

// MARK: - Factory registration

/// Overrides the `"llama"` (and `"mistral"`) entries in the shared MLX model-type
/// registry so `LLMModelFactory.shared.loadContainer(...)` instantiates
/// ``EmbeddingLlamaModel`` instead of the stock `LlamaModel`, which cannot take
/// soft-prompt embeddings.
///
/// Call once, *before* the first model load. The closure mirrors the upstream private
/// `create(_:_:)` helper: the URL it receives is the `config.json` file itself (not the
/// directory), and it only builds the bare model — the factory loads/quantizes weights afterwards.
public func registerModelTypes() {
    let creator: @Sendable (URL) throws -> any LanguageModel = { url in
        let config = try JSONDecoder().decode(
            EmbeddingLlamaConfiguration.self, from: Data(contentsOf: url))
        return EmbeddingLlamaModel(config)
    }
    LLMModelFactory.shared.typeRegistry.registerModelType("llama", creator: creator)
    LLMModelFactory.shared.typeRegistry.registerModelType("mistral", creator: creator)
}
