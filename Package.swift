// swift-tools-version: 6.0
//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import PackageDescription

let package = Package(
    name: "PocketTSLM",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "PocketTSLM", targets: ["PocketTSLM"])
    ],
    dependencies: [
        // Exactly the MLX stack the demo app resolves through SpeziLLM `feature/add-tools`, because an app
        // links one MLX version. The revision pin only works while apps add this package by local path.
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.21.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples", revision: "0c50f71e1a3fcaaf09d1f93721a55e4f291bc442"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "0.1.20")
    ],
    targets: [
        .target(
            name: "PocketTSLM",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers")
            ]
        ),
        .executableTarget(
            name: "pockettslm-cli",
            dependencies: [
                "PocketTSLM",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples")
            ]
        ),
        .testTarget(
            name: "PocketTSLMTests",
            dependencies: [
                "PocketTSLM",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers")
            ],
            // The tests read their fixtures relative to their own source files.
            exclude: ["Fixtures", "generate_interleaver_fixtures.py", "generate_parity_fixtures.py", "generate_real_parity_fixtures.py"]
        )
    ],
    swiftLanguageModes: [.v5]
)
