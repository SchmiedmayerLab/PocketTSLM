//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import MLX

extension Device {
    /// mlx-swift 0.21 spells this `using(device:fn:)`. Remove the shim once the package moves to
    /// mlx-swift 0.29, which has `Device.withDefaultDevice` itself.
    static func withDefaultDevice<R>(_ device: Device, _ body: () throws -> R) rethrows -> R {
        try using(device: device, fn: body)
    }
}
