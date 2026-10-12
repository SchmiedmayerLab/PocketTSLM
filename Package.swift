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
        // Grove's MLX stack, because an app links one MLX version. mlx-swift-examples releases after
        // 2.29 may lack MLXLLM, which moved to mlx-swift-lm.
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.29.1")),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples", .upToNextMinor(from: "2.29.1")),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0")
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
            exclude: ["Fixtures", "generate_parity_fixtures.py"]
        )
    ],
    swiftLanguageModes: [.v5]
)
