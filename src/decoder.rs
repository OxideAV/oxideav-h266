//! Registry glue for the VVC foundation crate.
//!
//! The foundation crate does not perform any pixel reconstruction. The
//! registered decoder factory just returns a placeholder that rejects every
//! packet with `Error::Unsupported`. The useful surface of this crate at
//! the foundation stage lives in the parser modules (`nal`, `bitreader`,
//! and — as they land in subsequent increments — `vps`, `sps`, `pps`,
//! `aps`, `picture_header`, `slice_header`).

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::CODEC_ID_STR;

/// Build the placeholder foundation decoder for the registry.
pub fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(FoundationDecoder {
        codec_id: CodecId::new(CODEC_ID_STR),
    }))
}

pub struct FoundationDecoder {
    codec_id: CodecId,
}

impl Decoder for FoundationDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, _packet: &Packet) -> Result<()> {
        Err(Error::unsupported(
            "h266: foundation scaffold does not reconstruct pictures yet",
        ))
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(Error::Eof)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}
