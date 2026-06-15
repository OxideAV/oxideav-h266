//! VVC SEI manifest SEI message parser (Annex D.8.1 / D.8.2).
//!
//! The SEI manifest SEI message (`payloadType == 200`) conveys, for the
//! coded video sequence (CVS) that contains it, a list of SEI message
//! types together with an indication of whether each type is expected
//! (i.e., likely) to be present and — when expected — the degree of
//! necessity of interpreting those messages. It is one of the few
//! `sei_payload()` bodies specified directly in this Specification
//! (most other payload types defer to Rec. ITU-T H.274 |
//! ISO/IEC 23002-7), so it is parseable from its own payload bytes with
//! no external buffering-period / HRD context.
//!
//! Syntax (§D.8.1):
//!
//! ```text
//!   sei_manifest( payloadSize ) {                              Descriptor
//!       manifest_num_sei_msg_types                                 u(16)
//!       for( i = 0; i < manifest_num_sei_msg_types; i++ ) {
//!           manifest_sei_payload_type[ i ]                         u(16)
//!           manifest_sei_description[ i ]                          u(8)
//!       }
//!   }
//! ```
//!
//! §D.8.2 semantics modelled here:
//!
//! * `manifest_num_sei_msg_types` — the number of SEI message types for
//!   which information is provided.
//! * `manifest_sei_payload_type[ i ]` — the `payloadType` value of the
//!   i-th described SEI message type. §D.8.2 requires these values to
//!   be pairwise distinct across `i`; the parser surfaces a duplicate
//!   as a parse error.
//! * `manifest_sei_description[ i ]` — interpretation per Table D.2:
//!     - `0` → no message of this type expected to be present,
//!     - `1` → messages expected, considered *necessary*,
//!     - `2` → messages expected, considered *unnecessary*,
//!     - `3` → messages expected, necessity *undetermined*,
//!     - `4..=255` → reserved for future use. §D.8.2 requires decoders
//!       to *allow* reserved values to appear in the syntax (and ignore
//!       the associated information) rather than reject them, so the raw
//!       value is preserved as [`ManifestDescription::Reserved`].
//!
//! The payload is byte-aligned and a whole number of bytes (a `u(16)`
//! header followed by `3` bytes per entry), matching §7.4.6's "the
//! derived `payloadSize` ... shall be equal to the number of RBSP bytes
//! in the SEI message payload". The parser cross-checks the consumed
//! length against `payloadSize` so a truncated or over-long entry run
//! surfaces as a parse error instead of silently desynchronising the
//! enclosing `sei_rbsp()` walk.
//!
//! Spec reference: ITU-T H.266 | ISO/IEC 23090-3 (V4, 01/2026),
//! §D.8.1, §D.8.2, Table D.2, §7.4.6.
//!
//! No third-party VVC decoder source was consulted; the implementation
//! is spec-only and reads the payload through the crate's own
//! [`BitReader`].

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// `payloadType` value that selects the SEI manifest body in §D.2.1.
pub const SEI_MANIFEST_PAYLOAD_TYPE: u32 = 200;

/// One `manifest_sei_description[ i ]` value, interpreted per Table D.2.
///
/// Conforming bitstreams use `0..=3`; `4..=255` is reserved and
/// preserved as [`ManifestDescription::Reserved`] so callers can
/// inspect the raw byte while honouring §D.8.2's "decoders shall also
/// allow ... and shall ignore" handling rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifestDescription {
    /// `0` — no SEI message with the associated `payloadType` is
    /// expected to be present in the CVS.
    NotExpected,
    /// `1` — messages expected to be present and considered *necessary*.
    ExpectedNecessary,
    /// `2` — messages expected to be present and considered
    /// *unnecessary*.
    ExpectedUnnecessary,
    /// `3` — messages expected to be present, necessity *undetermined*.
    ExpectedUndetermined,
    /// `4..=255` — reserved for future use; raw value preserved.
    Reserved(u8),
}

impl ManifestDescription {
    /// Map the raw `u(8)` `manifest_sei_description[ i ]` byte to the
    /// Table D.2 enum.
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            0 => ManifestDescription::NotExpected,
            1 => ManifestDescription::ExpectedNecessary,
            2 => ManifestDescription::ExpectedUnnecessary,
            3 => ManifestDescription::ExpectedUndetermined,
            other => ManifestDescription::Reserved(other),
        }
    }

    /// The raw `u(8)` value that produced this description.
    pub fn raw(self) -> u8 {
        match self {
            ManifestDescription::NotExpected => 0,
            ManifestDescription::ExpectedNecessary => 1,
            ManifestDescription::ExpectedUnnecessary => 2,
            ManifestDescription::ExpectedUndetermined => 3,
            ManifestDescription::Reserved(raw) => raw,
        }
    }

    /// `true` when the value is in the conforming `0..=3` range of
    /// Table D.2 (i.e. not [`ManifestDescription::Reserved`]).
    pub fn is_conforming(self) -> bool {
        !matches!(self, ManifestDescription::Reserved(_))
    }

    /// `true` when the description indicates messages of the associated
    /// type are expected to be present (Table D.2 values `1`, `2`, or
    /// `3`). `false` for `NotExpected` and reserved values (which
    /// §D.8.2 instructs decoders to ignore).
    pub fn is_expected_present(self) -> bool {
        matches!(
            self,
            ManifestDescription::ExpectedNecessary
                | ManifestDescription::ExpectedUnnecessary
                | ManifestDescription::ExpectedUndetermined
        )
    }
}

/// One `(manifest_sei_payload_type[ i ], manifest_sei_description[ i ])`
/// entry of the SEI manifest (§D.8.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ManifestEntry {
    /// `manifest_sei_payload_type[ i ]` — `payloadType` of the i-th
    /// described SEI message type.
    pub payload_type: u16,
    /// `manifest_sei_description[ i ]` interpreted per Table D.2.
    pub description: ManifestDescription,
}

/// A parsed SEI manifest SEI message (§D.8.1).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SeiManifest {
    /// The `(payload_type, description)` entries, in stream order.
    /// `entries.len()` equals `manifest_num_sei_msg_types`.
    pub entries: Vec<ManifestEntry>,
}

impl SeiManifest {
    /// `manifest_num_sei_msg_types` — the number of described SEI
    /// message types.
    pub fn num_msg_types(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no SEI message types are described.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry describing `payload_type`, if the manifest lists it.
    pub fn entry_for(&self, payload_type: u16) -> Option<&ManifestEntry> {
        self.entries.iter().find(|e| e.payload_type == payload_type)
    }
}

/// Parse a `sei_manifest( payloadSize )` body (§D.8.1) from the raw SEI
/// payload bytes carried by a `payloadType == 200` `sei_message()`.
///
/// `payload` is the `sei_payload()` argument region of the SEI message
/// — the `payloadSize` bytes that follow the `sei_message()` header,
/// with emulation-prevention bytes already removed. The body is
/// byte-aligned and occupies `2 + 3 * manifest_num_sei_msg_types`
/// bytes; the parser requires `payload.len()` to match that exactly so
/// a framing error does not desynchronise the enclosing `sei_rbsp()`
/// walk.
///
/// Errors:
/// * a `payload` shorter than the `manifest_num_sei_msg_types` header
///   or than the entry run it announces is rejected as truncated;
/// * a `payload` longer than the announced entry run (extra trailing
///   bytes beyond the §D.8.1 structure) is rejected;
/// * a duplicate `manifest_sei_payload_type[ i ]` violates the §D.8.2
///   "shall not be identical when m is not equal to n" constraint and
///   is rejected.
pub fn parse_sei_manifest(payload: &[u8]) -> Result<SeiManifest> {
    // Header is a single u(16); each entry is u(16) + u(8) == 3 bytes.
    if payload.len() < 2 {
        return Err(Error::invalid(
            "h266 sei_manifest: payload too short for manifest_num_sei_msg_types (§D.8.1)",
        ));
    }

    let mut reader = BitReader::new(payload);
    let num_types = reader.u(16)? as usize;

    // §7.4.6: payloadSize is exactly the body length. A SEI manifest is
    // a 2-byte count plus 3 bytes per entry, so cross-check before
    // reading the entry run to reject truncated / over-long payloads.
    let expected_len = 2 + num_types * 3;
    if payload.len() != expected_len {
        return Err(Error::invalid(
            "h266 sei_manifest: payloadSize does not match 2 + 3 * manifest_num_sei_msg_types \
             (§D.8.1 / §7.4.6)",
        ));
    }

    let mut entries = Vec::with_capacity(num_types);
    for _ in 0..num_types {
        let payload_type = reader.u(16)? as u16;
        let description = ManifestDescription::from_raw(reader.u(8)? as u8);
        // §D.8.2: manifest_sei_payload_type values shall be pairwise
        // distinct.
        if entries
            .iter()
            .any(|e: &ManifestEntry| e.payload_type == payload_type)
        {
            return Err(Error::invalid(
                "h266 sei_manifest: duplicate manifest_sei_payload_type (§D.8.2)",
            ));
        }
        entries.push(ManifestEntry {
            payload_type,
            description,
        });
    }

    Ok(SeiManifest { entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty manifest (`manifest_num_sei_msg_types == 0`) is the bare
    /// 2-byte header with no entries.
    #[test]
    fn empty_manifest() {
        let payload = [0x00u8, 0x00];
        let m = parse_sei_manifest(&payload).unwrap();
        assert_eq!(m.num_msg_types(), 0);
        assert!(m.is_empty());
    }

    /// A single entry: type 168 (frame_field_info), description 1
    /// (expected / necessary).
    #[test]
    fn single_entry_necessary() {
        // count = 1, then payload_type = 168 (0x00A8), description = 1.
        let payload = [0x00u8, 0x01, 0x00, 0xA8, 0x01];
        let m = parse_sei_manifest(&payload).unwrap();
        assert_eq!(m.num_msg_types(), 1);
        let e = &m.entries[0];
        assert_eq!(e.payload_type, 168);
        assert_eq!(e.description, ManifestDescription::ExpectedNecessary);
        assert!(e.description.is_conforming());
        assert!(e.description.is_expected_present());
        assert_eq!(m.entry_for(168), Some(e));
        assert_eq!(m.entry_for(45), None);
    }

    /// Multiple entries with the four conforming description values.
    #[test]
    fn multiple_entries_all_descriptions() {
        let payload = [
            0x00, 0x04, // count = 4
            0x00, 0x00, 0x00, // type 0, desc 0 (NotExpected)
            0x00, 0x2D, 0x01, // type 45, desc 1 (ExpectedNecessary)
            0x00, 0x96, 0x02, // type 150, desc 2 (ExpectedUnnecessary)
            0x00, 0xCB, 0x03, // type 203, desc 3 (ExpectedUndetermined)
        ];
        let m = parse_sei_manifest(&payload).unwrap();
        assert_eq!(m.num_msg_types(), 4);
        assert_eq!(m.entries[0].payload_type, 0);
        assert_eq!(m.entries[0].description, ManifestDescription::NotExpected);
        assert!(!m.entries[0].description.is_expected_present());
        assert_eq!(m.entries[1].payload_type, 45);
        assert_eq!(
            m.entries[1].description,
            ManifestDescription::ExpectedNecessary
        );
        assert_eq!(m.entries[2].payload_type, 150);
        assert_eq!(
            m.entries[2].description,
            ManifestDescription::ExpectedUnnecessary
        );
        assert_eq!(m.entries[3].payload_type, 203);
        assert_eq!(
            m.entries[3].description,
            ManifestDescription::ExpectedUndetermined
        );
    }

    /// A reserved description value (`4..=255`) is preserved verbatim
    /// and flagged non-conforming, per §D.8.2's "decoders shall also
    /// allow ... and shall ignore" rule (it is NOT rejected).
    #[test]
    fn reserved_description_preserved() {
        let payload = [0x00u8, 0x01, 0x00, 0x05, 0xFF];
        let m = parse_sei_manifest(&payload).unwrap();
        let d = m.entries[0].description;
        assert_eq!(d, ManifestDescription::Reserved(0xFF));
        assert!(!d.is_conforming());
        assert!(!d.is_expected_present());
        assert_eq!(d.raw(), 0xFF);
    }

    /// Round-trip every raw description byte through `from_raw`/`raw`.
    #[test]
    fn description_raw_roundtrip() {
        for raw in 0u8..=255 {
            assert_eq!(ManifestDescription::from_raw(raw).raw(), raw);
        }
    }

    /// A duplicate `manifest_sei_payload_type` violates §D.8.2 and is
    /// rejected.
    #[test]
    fn duplicate_payload_type_rejected() {
        let payload = [
            0x00, 0x02, // count = 2
            0x00, 0x2D, 0x01, // type 45
            0x00, 0x2D, 0x02, // type 45 again -> duplicate
        ];
        assert!(parse_sei_manifest(&payload).is_err());
    }

    /// A payload shorter than the 2-byte header is rejected.
    #[test]
    fn truncated_header_rejected() {
        assert!(parse_sei_manifest(&[0x00]).is_err());
        assert!(parse_sei_manifest(&[]).is_err());
    }

    /// A payload that announces more entries than its byte length can
    /// hold is rejected.
    #[test]
    fn truncated_entry_run_rejected() {
        // count = 2 needs 2 + 6 = 8 bytes, only 5 supplied.
        let payload = [0x00u8, 0x02, 0x00, 0x2D, 0x01];
        assert!(parse_sei_manifest(&payload).is_err());
    }

    /// A payload with extra trailing bytes beyond the §D.8.1 structure
    /// is rejected (would otherwise desync the sei_rbsp() walk).
    #[test]
    fn over_long_payload_rejected() {
        // count = 1 needs 2 + 3 = 5 bytes, 6 supplied.
        let payload = [0x00u8, 0x01, 0x00, 0x2D, 0x01, 0xAB];
        assert!(parse_sei_manifest(&payload).is_err());
    }

    /// `SEI_MANIFEST_PAYLOAD_TYPE` matches the §D.2.1 dispatch value.
    #[test]
    fn payload_type_constant() {
        assert_eq!(SEI_MANIFEST_PAYLOAD_TYPE, 200);
    }
}
