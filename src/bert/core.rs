use burn::{config::Config, module::Module, nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput}, prelude::Backend, tensor::{Bool, Int, Tensor}};
use derive_builder::Builder;

use super::{BertEmbedding, BertEmbeddingConfig};


/// Configuration to initialize a bert model
#[derive(Debug, Config)]
pub struct BertModelConfig {
    // ---- embedding ---------
    /// size of the word token vocabulary
    #[config(default="30522")]
    pub vocab_size: usize,
    /// identifier of the pad token
    #[config(default="0")]
    pub pad_token_id: usize,
    /// size of the segment vocabulary (default 2)
    #[config(default="2")]
    pub type_vocab_size: usize,
    /// max length of any processable sequence
    #[config(default="512")]
    pub max_position_embeddings: usize,
    /// 'hidden' size of the embeddings
    #[config(default="768")]
    pub hidden_size: usize,
    /// probability that an embedding neuron be deactivated during a training step
    #[config(default="0.1")]
    pub hidden_dropout_prob: f64,
    /// small value whose role is to prevent division by zero in layer norm
    #[config(default="1e-12")]
    pub layer_norm_eps: f64,
    // ---- encoder
    /// probability that a neuron from the hidden layers in the encoder be deactivated during training
    #[config(default="0.1")]
    pub attention_probs_dropout_prob: f64,
    /// std deviation when initializing the encoder weights
    #[config(default="0.02")]
    pub initializer_range: f64,
    /// output size of the hidden layers
    #[config(default="3072")]
    pub intermediate_size: usize,
    /// number of self attention heads in the encoder
    #[config(default="12")]
    pub num_attention_heads: usize,
    /// number of encoder layers
    #[config(default="12")]
    pub num_hidden_layers: usize,
}

/// The input of a bert model.
/// 
/// The only mandatory input is the 'input_ids' field.
#[derive(Debug, Clone, Builder)]
pub struct BertInputBatch<B: Backend> {
    /// The identifier of the encoded tokens 
    /// Shape: [batch, sequence]
    pub input_ids: Tensor<B, 2, Int>,
    /// The segment identifiers 
    /// Shape: [batch, sequence]
    pub token_type_ids: Option<Tensor<B, 2, Int>>,
    /// The mask to hide the padding stuff 
    /// Shape: [batch, sequence]
    pub padding_mask: Option<Tensor<B, 2, Bool>>,
}

/// The core bert model consisting of a bert embedding layer alongside with an encoder layer
#[derive(Debug, Module)]
pub struct BertModel<B: Backend> {
    pub hidden_size: usize,
    pub embedding: BertEmbedding<B>,
    pub encoder: TransformerEncoder<B>
}

impl BertModelConfig {
    /// Initializes the model
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertModel<B> {
        let embedding = self.embedding_config().init(device);
        let encoder = self.encoder_config().init(device);
        
        BertModel { hidden_size: self.hidden_size, embedding, encoder }
    }
    /// Creates an embedding configuration
    fn embedding_config(&self) -> BertEmbeddingConfig {
        BertEmbeddingConfig::new()
            .with_vocab_size(self.vocab_size)
            .with_type_vocab_size(self.type_vocab_size)
            .with_pad_token_id(self.pad_token_id)
            .with_max_position_embeddings(self.max_position_embeddings)
            .with_layer_norm_eps(self.layer_norm_eps)
            .with_hidden_size(self.hidden_size)
            .with_hidden_dropout_prob(self.hidden_dropout_prob)
    }
    /// Creates an encoder configuration
    fn encoder_config(&self) -> TransformerEncoderConfig {
        TransformerEncoderConfig::new(
            self.hidden_size, 
            self.intermediate_size, 
            self.num_attention_heads, 
            self.num_hidden_layers)
            .with_dropout(self.attention_probs_dropout_prob)
            .with_initializer(burn::nn::Initializer::Normal { mean: 0.0, std: self.initializer_range })
            .with_norm_first(false)
            .with_quiet_softmax(false)
    }
}

impl <B: Backend> BertModel<B> {
    pub fn forward(&self, x: BertInputBatch<B>) -> Tensor<B, 3>{
        let y = self.embedding.forward(x.input_ids, x.token_type_ids);

        let mut e_input = TransformerEncoderInput::new(y);
        if let Some(pad_mask) = x.padding_mask {
            e_input = e_input.mask_pad(pad_mask);
        }
        self.encoder.forward(e_input)
    }
}