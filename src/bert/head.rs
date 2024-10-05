use burn::{config::Config, module::Module, nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig}, prelude::Backend, tensor::{activation::gelu, Tensor}};

/// After the encoder, we pass the output tensor inside of a shallow network before
/// producing the actual output. This transformation is used to normalize the logits
/// before they are output.
/// 
/// This struct provides a config to initialize this shallow network
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
    /// Creates a prediction transform module from this config
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertPredictionTransform<B> {
        let dense = LinearConfig::new(self.hidden_size, self.hidden_size).init(device);
        let layer_norm = LayerNormConfig::new(self.hidden_size).with_epsilon(self.layer_norm_eps).init(device);

        BertPredictionTransform { dense, layer_norm }
    }
}

/// After the encoder, we pass the output tensor inside of a shallow network before
/// producing the actual output. This transformation is used to normalize the logits
/// before they are output.
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


/// A basic language modeling head configuration
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
    /// creates an associated module
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


/// A basic language modeling head
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

/// Masked language modeling head configuration
pub type BertMLMHeadConfig = BertLMPredictionHeadConfig;
/// Masked language modeling (mlm) head
pub type BertMLMHead<B> = BertLMPredictionHead<B>;

/// Configuration for the next sentence prediction head
#[derive(Debug, Config)]
pub struct BertTokenClassificationHeadConfig {
    /// 'hidden' size of the embeddings
    #[config(default="768")]
    pub hidden_size: usize,
    /// the number of possible classes
    #[config(default="2")]
    pub num_classes: usize,
    /// probability that an embedding neuron be deactivated during a training step
    #[config(default="0.1")]
    pub hidden_dropout_prob: f64,
    /// std deviation when initializing the encoder weights
    #[config(default="0.02")]
    pub initializer_range: f64,
}

/// Next Sentence prediction head
#[derive(Debug, Module)]
pub struct BertTokenClassificationHead<B: Backend> {
    pub dropout: Dropout,
    pub classifier: Linear<B>
}
impl BertTokenClassificationHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertTokenClassificationHead<B> {
        let dropout = DropoutConfig::new(self.hidden_dropout_prob).init();
        let classifier = LinearConfig::new(self.hidden_size, self.num_classes)
            .with_initializer(burn::nn::Initializer::Normal { mean: 0.0, std: self.initializer_range })
            .init(device);

        BertTokenClassificationHead { dropout, classifier }
    }
}
impl <B: Backend> BertTokenClassificationHead<B> {
    pub fn forward(&self, mut hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        hidden = self.dropout.forward(hidden);
        self.classifier.forward(hidden)
    }
}