// Copyright (c) 2026 Torin Etheridge. MIT License (see LICENSE).
// Author: Torin Etheridge · 2026-10-09

/// GPT-2 byte ↔ printable-unicode mapping used by byte-level BPE vocabularies.
///
/// Printable Latin-1 bytes (33–126, 161–172, 174–255) map to themselves; the remaining 68 bytes map to
/// U+0100, U+0101, … in byte order. This is the table every GPT-2-style vocabulary string is written in.
enum ByteLevel {
    static let byteToScalar: [UInt8: Unicode.Scalar] = {
        var printable = Array(33...126) + Array(161...172) + Array(174...255)
        var table: [UInt8: Unicode.Scalar] = [:]
        for b in printable { table[UInt8(b)] = Unicode.Scalar(UInt32(b))! }
        var extra: UInt32 = 0
        for b in 0...255 where !printable.contains(b) {
            table[UInt8(b)] = Unicode.Scalar(256 + extra)!
            extra += 1
        }
        printable.removeAll()
        return table
    }()

    static let scalarToByte: [Unicode.Scalar: UInt8] = {
        Dictionary(uniqueKeysWithValues: byteToScalar.map { ($1, $0) })
    }()
}
