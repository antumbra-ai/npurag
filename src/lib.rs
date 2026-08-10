//! npurag — on-device semantic search and RAG over a local directory.
//!
//! The crate is split into a library and a thin binary so that the integration
//! tests in `tests/` can drive the same code paths the CLI uses. Everything that
//! touches an inference server goes through the [`backend::Backend`] trait, which
//! keeps the whole pipeline runnable against [`backend::MockBackend`] with no
//! hardware and no running server.

pub mod backend;
pub mod chunk;
pub mod config;
pub mod extract;
pub mod index;
pub mod store;
pub mod walk;
