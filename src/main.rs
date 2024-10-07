use burn::{backend::{wgpu::WgpuDevice, Autodiff, Wgpu}, optim::AdamWConfig};
use tinybert_rs::{bert::BertForPreTrainingConfig, data::data_from_csv, train::{train, TrainingConfig}};
use tokenizers::Tokenizer;

fn main() {
    type Backend = Wgpu;
    let device = WgpuDevice::BestAvailable;
    
    let mut tok = Tokenizer::from_pretrained("xaviergillard/parti-pris-v2", None).unwrap();
    let mask_id = tok.get_added_vocabulary().get_vocab()["[MASK]"] as usize;
    let pad_id=  tok.get_added_vocabulary().get_vocab()["[PAD]"] as usize;
    let vocab_size = tok.get_vocab_size(false);

    let model = BertForPreTrainingConfig::new()
        .with_max_position_embeddings(1024)
        .with_type_vocab_size(2)
        .with_vocab_size(vocab_size)
        .with_pad_token_id(pad_id);
    
    let optimizer = AdamWConfig::new();
    let config = TrainingConfig::new(model, optimizer, "./artifacts/".to_string())
        .with_mask_token_id(mask_id)
        .with_batch_size(32)
        .with_num_epochs(60);


    let (trainset, validset) = data_from_csv("resources/corpus_partipris_v2.csv", &mut tok, 20);
    train::<Autodiff<Backend>>(config, trainset, validset, device);
}
