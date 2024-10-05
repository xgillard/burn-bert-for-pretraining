use burn::{backend::{wgpu::WgpuDevice, Wgpu}, optim::AdamWConfig, serde::de};
use tinybert_rs::{bert::BertForPreTrainingConfig, train::{self, TrainingConfig}};
use tokenizers::Tokenizer;

fn main() {
    let device = WgpuDevice::IntegratedGpu(0);

    let tok = Tokenizer::from_pretrained("xaviergillard/parti-pris-v2", None).unwrap();

    let mask_id = tok.get_added_vocabulary().get_vocab()["[MASK]"];
    let pad_id=  tok.get_added_vocabulary().get_vocab()["[PAD]"];
    let vocab_size = tok.get_vocabs_size(false);

    let model = BertForPreTrainingConfig::new()
        .with_max_position_embeddings(1024)
        .with_type_vocab_size(2)
        .with_vocab_size(vocab_size)
        .with_pad_token_id(pad_id);
    
    let optimizer = AdamWConfig::new();
    let config = TrainingConfig::new(model, optimizer)
        .with_mask_token_id(mask_id);

    train("./artifacts", config, device);
}
