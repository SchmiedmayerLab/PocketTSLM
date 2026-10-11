//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import SwiftUI

@main
struct PocketTSLMApplication: App {
    @UIApplicationDelegateAdaptor(PocketTSLMAppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            PocketTSLMView()
                .spezi(appDelegate)
        }
    }
}
