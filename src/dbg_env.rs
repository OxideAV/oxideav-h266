//! Triage switches (`H266_DBG_*`, `H266_DUMP_*`, `H266_ENC_*`
//! environment variables) behind one cached gate.
//!
//! The decode and encode paths consult these switches per CU, per
//! transform block and per filter edge. An environment lookup takes
//! the process environment lock and scans every variable, so on a
//! normal run — no switch set — the lookups were pure overhead on the
//! hottest paths. [`var_os`] / [`var`] answer "not set" from one
//! atomic load unless at least one variable with a switch prefix
//! ([`SWITCH_PREFIXES`]) existed when the first switch was consulted;
//! only then do they read the environment. Other `H266_` variables
//! (corpus locations, bench settings) do not open the gate.
//!
//! The gate is sampled once per process: a switch has to be in the
//! environment before the first picture is coded.

use std::ffi::OsString;
use std::sync::OnceLock;

/// The name prefixes of every switch this crate reads.
const SWITCH_PREFIXES: [&str; 3] = ["H266_DBG_", "H266_DUMP_", "H266_ENC_"];

fn is_switch(name: &str) -> bool {
    SWITCH_PREFIXES.iter().any(|p| name.starts_with(p))
}

/// `true` when the process environment held any switch at first use.
fn any_switch_set() -> bool {
    static ANY: OnceLock<bool> = OnceLock::new();
    *ANY.get_or_init(|| std::env::vars_os().any(|(k, _)| is_switch(&k.to_string_lossy())))
}

/// `std::env::var_os` for a switch.
#[inline]
pub(crate) fn var_os(name: &str) -> Option<OsString> {
    debug_assert!(is_switch(name));
    if any_switch_set() {
        std::env::var_os(name)
    } else {
        None
    }
}

/// `std::env::var` for a switch.
#[inline]
pub(crate) fn var(name: &str) -> Result<String, std::env::VarError> {
    debug_assert!(is_switch(name));
    if any_switch_set() {
        std::env::var(name)
    } else {
        Err(std::env::VarError::NotPresent)
    }
}
