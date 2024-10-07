use thiserror::Error;


/// The kind of errors that can happen in this program
#[derive(Debug, Error)]
pub enum Error {
    #[error("not a boolean")]
    NotABoolean,
    #[error("not an integer")]
    NotAnInt,
    #[error("not a list")]
    NotAList,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Custom result type to map the possible errors
pub type Result<T> = std::result::Result<T, Error>;