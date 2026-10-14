//
// This source file is part of the PocketTSLM open-source project
//
// SPDX-FileCopyrightText: 2026 Schmiedmayer Lab and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation

/// The CSV reading that the dataset readers share.
enum CSV {
    /// The records in the first 20 MB of a CSV file, header first. The Sleep-EDF export is 350 MB,
    /// and the readers only need its first rows.
    static func records(at url: URL) throws -> [[String]] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        // `String(decoding:)` on purpose: the 20 MB cut can split a UTF-8 sequence, which the failable
        // initialiser would reject, and the cut tail is dropped below anyway.
        // swiftlint:disable:next optional_data_string_conversion
        let text = String(decoding: try handle.read(upToCount: 20 * 1024 * 1024) ?? Data(), as: UTF8.self)
        return parse(String(text.prefix(upTo: text.lastIndex(of: "\n") ?? text.endIndex)))
    }

    private static func parse(_ csv: String) -> [[String]] {
        var records: [[String]] = []
        var row: [String] = []
        var field = ""
        var insideQuotes = false

        var index = csv.startIndex
        while index < csv.endIndex {
            let character = csv[index]
            if character == "\"" {
                let nextIndex = csv.index(after: index)
                if insideQuotes && nextIndex < csv.endIndex && csv[nextIndex] == "\"" {
                    field.append("\"")
                    index = csv.index(after: nextIndex)
                    continue
                }
                insideQuotes.toggle()
                index = nextIndex
                continue
            }

            if character == "," && !insideQuotes {
                row.append(field)
                field.removeAll(keepingCapacity: true)
                index = csv.index(after: index)
                continue
            }

            // NB: Swift treats "\r\n" as a single Character (grapheme cluster), so a CRLF file
            // matches neither "\n" nor "\r" and the whole file collapses into one record.
            if (character == "\n" || character == "\r" || character == "\r\n") && !insideQuotes {
                row.append(field)
                field.removeAll(keepingCapacity: true)
                if !row.isEmpty {
                    records.append(row)
                }
                row.removeAll(keepingCapacity: true)

                if character == "\r" {
                    let nextIndex = csv.index(after: index)
                    if nextIndex < csv.endIndex && csv[nextIndex] == "\n" {
                        index = csv.index(after: nextIndex)
                        continue
                    }
                }

                index = csv.index(after: index)
                continue
            }

            field.append(character)
            index = csv.index(after: index)
        }

        if !field.isEmpty || !row.isEmpty {
            row.append(field)
            records.append(row)
        }

        return records
    }
}
