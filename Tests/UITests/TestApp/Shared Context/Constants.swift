//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation

enum Constants {
    // You can override this at runtime for debugging by setting the env var
    // POCKETTSLM_MODEL_ID in the scheme.
    private static let defaultLLMModelName = "meta-llama/Llama-3.2-1B"
    static let llmModelName = ProcessInfo.processInfo.environment["POCKETTSLM_MODEL_ID"] ?? defaultLLMModelName

    // Local model directory copied into Spezi/Hub cache before LLM setup.
    // Override with POCKETTSLM_LOCAL_MODEL_PATH in the scheme if needed.
    static let localModelSourcePathOverride = ProcessInfo.processInfo.environment["POCKETTSLM_LOCAL_MODEL_PATH"]
    static let localModelBundleSubdirectory = "LocalLLM"
    static let hostHuggingFaceCacheRoot = ProcessInfo.processInfo.environment["POCKETTSLM_HOST_HF_CACHE_ROOT"]
        ?? "~/.cache/huggingface/hub"
    // Where SpeziLLMLocal / HubApi actually reads and writes the model:
    // <Documents>/huggingface/models/<modelID>  (matches HubApi.shared.localRepoLocation)
    static let llmLocalModelDirectory: URL = {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Documents", isDirectory: true)
        return documents
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(llmModelName, isDirectory: true)
    }()

    /// Cap main chat generation length (avoids long repetition loops in console and UI).
    static let llmDefaultMaxOutputLength = Int(
        ProcessInfo.processInfo.environment["POCKETTSLM_LLM_MAX_OUTPUT"] ?? "128"
    ) ?? 128

    // swiftlint:disable:next line_length
    static let llmModelChatTemplate = "{% set loop_messages = messages %}{% if not messages[0]['role'] == 'system' %}{% set dummy = loop_messages.insert(0, {'role': 'system', 'content': 'You are a helpful assistant'}) %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'system' %}{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    /// If "1", pass `Constants.llmModelChatTemplate` as the chat template to LLMLocalParameters;
    /// otherwise pass nil and let MLX use the tokenizer's built-in chat template from
    /// `tokenizer_config.json`. Defaults to "0", also outside Xcode (home screen, devicectl),
    /// because the custom Jinja template triggers `LLMLocalError.illegalContext`.
    static let useCustomChatTemplate = (ProcessInfo.processInfo.environment["POCKETTSLM_USE_CUSTOM_CHAT_TEMPLATE"] ?? "0") == "1"
}
