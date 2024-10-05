use std::default::Default;
use burn::{config::Config, module::Module, nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig}, prelude::Backend, tensor::{Float, Int, Tensor}};

/// Configuration of a Bert embedding
#[derive(Debug, Copy, Config)]
pub struct BertEmbeddingConfig {
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
    /// probability that a neuron be deactivated during a training step
    #[config(default="0.1")]
    pub hidden_dropout_prob: f64,
    /// small value whose role is to prevent division by zero in layer norm
    #[config(default="1e-12")]
    pub layer_norm_eps: f64,
}

/// An actual bert embedding module
#[derive(Debug, Module)]
pub struct BertEmbedding<B: Backend> {
    /// identifier of the pad token
    pub pad_token_id: usize,
    /// max length of any processable sequence
    pub max_position_embeddings: usize,
    /// token input embeddings
    word_embedding: Embedding<B>,
    /// segment embedding (token_type_id)
    segment_embedding: Embedding<B>,
    /// position embedding (absolute)
    position_embedding: Embedding<B>,
    /// normalization
    layer_norm: LayerNorm<B>,
    /// dropout
    dropout: Dropout,
}

impl Default for BertEmbeddingConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl BertEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertEmbedding<B> {
        BertEmbedding {
            pad_token_id:            self.pad_token_id,
            max_position_embeddings: self.max_position_embeddings,
            word_embedding:          EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            segment_embedding:       EmbeddingConfig::new(self.type_vocab_size, self.hidden_size).init(device),
            position_embedding:      EmbeddingConfig::new(self.max_position_embeddings, self.hidden_size).init(device),
            layer_norm:              LayerNormConfig::new(self.hidden_size).with_epsilon(self.layer_norm_eps).init(device),
            dropout:                 DropoutConfig::new(self.hidden_dropout_prob).init(),
        }
    }
}

impl <B: Backend> BertEmbedding<B> {
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>, token_type_ids: Option<Tensor<B, 2, Int>>) -> Tensor<B, 3, Float> {
        let shape = input_ids.shape();
        let device = input_ids.device();

        let seq_len = shape.dims[1];

        let words = self.word_embedding.forward(input_ids);

        let segment = token_type_ids.unwrap_or_else(|| Tensor::<B, 2, Int>::zeros(shape, &device));
        let segment = self.segment_embedding.forward(segment);

        let position = Tensor::arange(0..seq_len as i64, &device).unsqueeze();
        let position = self.position_embedding.forward(position);

        let embedding = words + segment + position;
        let embedding = self.layer_norm.forward(embedding);
        //
        self.dropout.forward(embedding)
    }
}


#[cfg(test)]
mod tests {
    use super::BertEmbeddingConfig;

    #[test]
    fn test_default() {
        let embedding_config: BertEmbeddingConfig = Default::default();
        assert_eq!(embedding_config.type_vocab_size, 2);
    }
}