use std::time::Instant;

use tinybert_rs::{data::read_parquet, error::Result};

fn main() -> Result<()>{
    let now = Instant::now();
    let x = read_parquet("../tinybert_hf/augmented/augmented-0000.parquet")?;
    let dur = Instant::now() - now;
    println!("SEQ {} -- {}", dur.as_secs_f32(), x.len());

    Ok(())
}
