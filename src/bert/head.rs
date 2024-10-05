use burn::{config::Config, module::Module, nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig}, prelude::Backend, tensor::{activation::gelu, Tensor}};

#[derive(Debug, Config)]
pub struct BertPredictionTransformConfig {
    /// 'hidden' size of the embeddings
    #[config(default="768")]
    pub hidden_size: usize,
    /// small value whose role is to prevent division by zero in layer norm
    #[config(default="1e-12")]
    pub layer_norm_eps: f64,
}

impl BertPredictionTransformConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertPredictionTransform<B> {
        let dense = LinearConfig::new(self.hidden_size, self.hidden_size).init(device);
        let layer_norm = LayerNormConfig::new(self.hidden_size).with_epsilon(self.layer_norm_eps).init(device);

        BertPredictionTransform { dense, layer_norm }
    }
}

#[derive(Debug, Module)]
pub struct BertPredictionTransform<B: Backend> {
    dense: Linear<B>,
    layer_norm: LayerNorm<B>
}
impl <B: Backend> BertPredictionTransform<B> {
    pub fn forward(&self, mut hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        hidden = self.dense.forward(hidden);
        hidden = gelu(hidden);
        self.layer_norm.forward(hidden)
    }
}



#[derive(Debug, Config)]
pub struct BertLMPredictionHeadConfig {
    /// 'hidden' size of the embeddings
    #[config(default="768")]
    pub hidden_size: usize,
    /// small value whose role is to prevent division by zero in layer norm
    #[config(default="1e-12")]
    pub layer_norm_eps: f64,
    /// size of the word token vocabulary
    #[config(default="30522")]
    pub vocab_size: usize,
    /// std deviation when initializing the weights
    #[config(default="0.02")]
    pub initializer_range: f64,
}
impl BertLMPredictionHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertLMPredictionHead<B> {
        let transform = BertPredictionTransformConfig::new()
            .with_hidden_size(self.hidden_size)
            .with_layer_norm_eps(self.layer_norm_eps)
            .init(device);
        let decoder = LinearConfig::new(self.hidden_size, self.vocab_size)
            .with_bias(true)
            .with_initializer(burn::nn::Initializer::Normal { mean: 0.0, std: self.initializer_range })
            .init(device);

        BertLMPredictionHead { transform, decoder }
    }
}

#[derive(Debug, Module)]
pub struct BertLMPredictionHead<B: Backend> {
    transform: BertPredictionTransform<B>,
    decoder: Linear<B>
}

impl <B: Backend> BertLMPredictionHead<B> {
    pub fn forward(&self, mut hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        hidden = self.transform.forward(hidden);
        self.decoder.forward(hidden)
    }
}

pub type BertMLMHeadConfig  = BertLMPredictionHeadConfig;
pub type BertMLMHead<B>     = BertLMPredictionHead<B>;