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
        let text = String(decoding: try handle.read(upToCount: 20 * 1024 * 1024) ?? Data(), as: UTF8.self)
        return parse(String(text.prefix(upTo: text.lastIndex(of: "\n") ?? text.endIndex)))
    }

    private static func parse(_ csv: String) -> [[String]] {
        var records: [[String]] = []
        var row: [String] = []
        var field = ""
        var insideQuotes = false

        var i = csv.startIndex
        while i < csv.endIndex {
            let ch = csv[i]
            if ch == "\"" {
                let nextIndex = csv.index(after: i)
                if insideQuotes && nextIndex < csv.endIndex && csv[nextIndex] == "\"" {
                    field.append("\"")
                    i = csv.index(after: nextIndex)
                    continue
                }
                insideQuotes.toggle()
                i = nextIndex
                continue
            }

            if ch == "," && !insideQuotes {
                row.append(field)
                field.removeAll(keepingCapacity: true)
                i = csv.index(after: i)
                continue
            }

            // NB: Swift treats "\r\n" as a single Character (grapheme cluster), so a CRLF file
            // matches neither "\n" nor "\r" and the whole file collapses into one record.
            if (ch == "\n" || ch == "\r" || ch == "\r\n") && !insideQuotes {
                row.append(field)
                field.removeAll(keepingCapacity: true)
                if !row.isEmpty {
                    records.append(row)
                }
                row.removeAll(keepingCapacity: true)

                if ch == "\r" {
                    let nextIndex = csv.index(after: i)
                    if nextIndex < csv.endIndex && csv[nextIndex] == "\n" {
                        i = csv.index(after: nextIndex)
                        continue
                    }
                }

                i = csv.index(after: i)
                continue
            }

            field.append(ch)
            i = csv.index(after: i)
        }

        if !field.isEmpty || !row.isEmpty {
            row.append(field)
            records.append(row)
        }

        return records
    }
}
