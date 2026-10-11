// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "OpenTSLMKit",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "OpenTSLMKit", targets: ["OpenTSLMKit"]),
        .executable(name: "OpenTSLMRunner", targets: ["OpenTSLMRunner"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.2"),
        // Same tokenizer the app uses (via MLXLLM) — pinned to the app's resolved version
        // so the tokenizer-parity test exercises the exact implementation.
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "0.1.20"),
    ],
    targets: [
        .target(
            name: "OpenTSLMKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "OpenTSLMRunner",
            dependencies: [
                "OpenTSLMKit",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "OpenTSLMKitTests",
            dependencies: [
                "OpenTSLMKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            resources: [
                .copy("Fixtures/encoder_io.npz"),
                .copy("Fixtures/encoder_io.safetensors"),
            ]
        ),
    ]
)
