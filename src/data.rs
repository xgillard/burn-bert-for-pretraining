use burn::data::dataset::InMemDataset;
use rand::{random, seq::SliceRandom, thread_rng};
use tokenizers::{Encoding, Tokenizer, TruncationParams};

use crate::bert::BertPreTrainingInputItem;
use rayon::prelude::*;



#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct AllTexts {
    pub title: String,
    pub full_text: String,
}

impl AllTexts {
    pub fn all_text(self) -> String {
        format!("{}\n{}", self.title, self.full_text)
    }
}

fn encoding_to_vec(mut e: Encoding, tok: &Tokenizer) -> Vec<String> {
    let mut out = vec![];
    out.push(tok.decode(e.get_ids(), true).unwrap());
    for i in e.take_overflowing() {
        let x = tok.decode(i.get_ids(), true).unwrap();
        out.push(x);
    }
    out
}


pub fn data_from_csv(fname: &str, tok: &mut Tokenizer, stride: usize) -> (InMemDataset<BertPreTrainingInputItem>, InMemDataset<BertPreTrainingInputItem>){
    tok.with_truncation(Some(TruncationParams{stride, ..Default::default()})).unwrap();

    let mut rdr = csv::ReaderBuilder::new()
        .from_path(fname)
        .unwrap();

    let data = rdr.deserialize()
        .par_bridge()
        .filter(|x: &Result<AllTexts, csv::Error>| x.is_ok())
        .flatten()
        .map(|x| x.all_text())
        .map(|x| tok.encode(x, true).unwrap())
        .map(|x| encoding_to_vec(x, tok))
        .collect::<Vec<_>>();

    let mut dset = vec![];
    tok.with_truncation(None).unwrap();
    let n = data.len();
    for (i, xs) in data.iter().enumerate() {
        for (j, x) in xs.iter().enumerate() {
            let mut nsp: bool = random() && j < xs.len() -1;
            let next = if nsp {
                &xs[j+1]
            } else {
                let nx = f32::floor(rand::random::<f32>() * n as f32) as usize;
                nsp = nx == i && j == 0;
                &data[nx][0]
            };

            dset.push(
                BertPreTrainingInputItem {
                    encoding: tok.encode((x.as_str(), next.as_str()), true).unwrap(),
                    nsp_label: if nsp { 0 } else { 1 },
                }
            );
        }
    }

    dset.shuffle(&mut thread_rng());
    let (train, valid) = dset.split_at_mut(data.len() * 8 / 10);

    (InMemDataset::new(train.to_vec()), InMemDataset::new(valid.to_vec()))
}