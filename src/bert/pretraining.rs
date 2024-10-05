use burn::{config::Config, data::dataloader::batcher::Batcher, module::Module, nn::{attention::{generate_padding_mask, GeneratePaddingMask}, loss::CrossEntropyLossConfig}, prelude::Backend, tensor::{backend::AutodiffBackend, Distribution, Int, Shape, Tensor, TensorData}, train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep}};
use derive_builder::Builder;
use tokenizers::Encoding;

use super::{BertInputBatch, BertMLMHead, BertMLMHeadConfig, BertModel, BertModelConfig, BertTokenClassificationHead, BertTokenClassificationHeadConfig};

/// One single input item for the bert pretraining. 
/// It comprises both a tokenized sentence pair and one nsp label
#[derive(Debug, Clone, Builder)]
pub struct BertPreTrainingInputItem {
    /// the tokenized sentence pair
    pub encoding: Encoding,
    /// the next sequence prediction label
    pub nsp_label: u32,
}

/// The input of a bert model.
/// 
/// The only mandatory input is the 'input_ids' field.
#[derive(Debug, Clone, Builder)]
pub struct BertPreTrainingLabelBatch<B: Backend> {
    /// The labels for the MLM task
    /// Shape: [batch, sequence]
    pub mlm_labels: Tensor<B, 2, Int>,
    /// The labels for the NSP task 
    /// Shape: [batch, sequence]
    pub nsp_labels: Tensor<B, 1, Int>,
}

/// The input for a bert pretraining step.
/// 
/// The only mandatory input is the 'input_ids' field.
#[derive(Debug, Clone, Builder)]
pub struct BertPreTrainingInputBatch<B: Backend> {
    pub input: BertInputBatch<B>,
    pub labels: BertPreTrainingLabelBatch<B>,
}

/// Creates a BertInputBatch from a series of bert pretraining inputs 
/// (that is, from a series of tokenized sequence pairs alongside with an nsp label)
#[derive(Debug, Clone, Builder)]
pub struct BertPreTrainingBatcher<B: Backend> {
    /// probability that a token be replaced by this processor
    #[builder(default="0.15")]
    pub mlm_proba: f64,
    /// the identifier of the [MASK] token in the vocabulary
    /// => for bert-base-uncased, it would be 103
    pub mask_token_id: u32,
    /// the identifier of the [PAD] token in the vocabulary
    /// => for bert-base-uncased, it would be 0
    pub pad_token_id: u32,
    /// maximum sequence length for the tokenized text
    pub max_seq_length: u32,
    /// the size of the vocabulary
    /// => for bert-base-uncased, it would be 30522
    pub voc_size: usize,
    /// the device where the tensors must be created
    pub device: B::Device
}

/// Configuration for the pretraining head
#[derive(Debug, Config)]
pub struct BertPreTrainingHeadConfig {
    /// 'hidden' size of the embeddings
    #[config(default="768")]
    pub hidden_size: usize,
    /// small value whose role is to prevent division by zero in layer norm
    #[config(default="1e-12")]
    pub layer_norm_eps: f64,
    /// size of the word token vocabulary
    #[config(default="30522")]
    pub vocab_size: usize,
    /// size of the segment vocabulary (default 2)
    #[config(default="2")]
    pub type_vocab_size: usize,
    /// std deviation when initializing the weights
    #[config(default="0.02")]
    pub initializer_range: f64,
    /// probability that an embedding neuron be deactivated during a training step
    #[config(default="0.1")]
    pub hidden_dropout_prob: f64,
}

/// The actual pretraining head stacked on top of a bert model when pretraining
#[derive(Debug, Module)]
pub struct BertPreTrainingHead<B: Backend> {
    mlm: BertMLMHead<B>,
    nsp: BertTokenClassificationHead<B>
}

/// Pretraining head output
#[derive(Debug)]
pub struct BertPreTrainingOutput<B: Backend> {
    pub mlm: Tensor<B, 3>,
    pub nsp: Tensor<B, 2>
}

#[derive(Debug, Config)]
pub struct BertForPreTrainingConfig {
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

/// The actual bert for pre training model
#[derive(Debug, Module)]
pub struct BertForPreTraining<B: Backend> {
    model: BertModel<B>,
    head: BertPreTrainingHead<B>
}

impl <B: Backend> BertPreTrainingBatcher<B> {
    /// Automatically applies MLM for the BERT pretraining.
    pub fn mask_tokens(&self, mut input_ids: Tensor<B, 2, Int>, special_tokens: Tensor<B, 2, Int>) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>){
        let shape = input_ids.shape();
        let device = input_ids.device();

        // this creates a boolean tensor of the same shape (but so far, it is considered to be a float tensor)
        let replaced: Tensor<B, 2, Int> = Tensor::random(shape.clone(), Distribution::Bernoulli(self.mlm_proba), &device);
        let replaced = replaced.mask_fill(special_tokens.clone().bool(), 0.0);
        let labels = input_ids.clone().mask_fill(replaced.clone().bool().bool_not(), -100);

        // 80% of the replaced tokens will be replaced by [MASK]
        let masked: Tensor<B, 2, Int> = Tensor::random(shape.clone(), Distribution::Bernoulli(0.8), &device);
        let masked = masked.min_pair(replaced.clone());
        input_ids = input_ids.mask_fill(masked.clone().bool(), self.mask_token_id);

        // 10% of the replaced tokens will be replaced by some random token
        let randtoken: Tensor<B, 2, Int> = Tensor::random(shape.clone(), Distribution::Bernoulli(0.5), &device);
        let randtoken = randtoken.min_pair(replaced.clone());
        let randtoken = randtoken.min_pair(masked.clone().bool().bool_not().int());
        let rnd_words: Tensor<B, 2, Int> = Tensor::random(shape.clone(), Distribution::Uniform(0.0, self.voc_size as f64), &device);
        input_ids = input_ids.mask_where(randtoken.clone().bool(), rnd_words);

        // 10% remaining masked tokens will actually be left unchanged
        (input_ids, labels)
    }

    /// Given an iterator over slices of integers, this method will create a 2d tensor of the given shape.
    /// Whenever a sequence from the iterator is shorter than shape[1], padding will be used.
    fn to_tensor_of_shape<'a, I>(&self, shape: Shape<2>, iter: I) -> Tensor<B, 2, Int>
        where I: Iterator<Item = &'a [u32]> + 'a
    {
        let mut tensor: Tensor<B, 2, Int> = Tensor::full(shape, self.pad_token_id, &self.device);
        
        for (i, item) in iter.enumerate() {
            tensor = tensor.slice_assign(
                #[allow(clippy::single_range_in_vec_init)]
                [i..i+1], 
                Tensor::from_data(
                    TensorData::new(item.to_vec(), [1, item.len()]), 
                    &self.device));
        }

        tensor
    }
}

impl <B: Backend> Batcher<BertPreTrainingInputItem, BertPreTrainingInputBatch<B>> for BertPreTrainingBatcher<B> {
    fn batch(&self, items: Vec<BertPreTrainingInputItem>) -> BertPreTrainingInputBatch<B> {
        let tokens_list = items.iter()
            .map(|i| i.encoding.get_ids().iter().copied().map(|i| i as usize).collect())
            .collect();

        // let us first ensure that all sequences are padded to a common length
        let GeneratePaddingMask{tensor, mask} = generate_padding_mask::<B>(
            self.pad_token_id as usize, 
            tokens_list, 
            Some(self.max_seq_length as usize), 
            &self.device);

        let input_ids = tensor;
        let padding_mask = mask;
        let shape = input_ids.shape();

        let token_type_ids = self.to_tensor_of_shape(shape.clone(), items.iter().map(|i| i.encoding.get_type_ids()));
        let special_tokens = self.to_tensor_of_shape(shape.clone(), items.iter().map(|i| i.encoding.get_special_tokens_mask()));

        let nsp_labels = items.iter().map(|i| i.nsp_label).collect::<Vec<_>>();
        let nsp_labels_len = nsp_labels.len();
        let nsp_labels = Tensor::from_data(TensorData::new(nsp_labels, [nsp_labels_len]), &self.device);

        let (input_ids, mlm_labels) = self.mask_tokens(input_ids, special_tokens);

        BertPreTrainingInputBatch {
            input: BertInputBatch {
                input_ids,
                token_type_ids: Some(token_type_ids),
                padding_mask: Some(padding_mask),
            },
            labels: BertPreTrainingLabelBatch { 
                mlm_labels,
                nsp_labels,
            },
        }
    }
}

impl BertPreTrainingHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertPreTrainingHead<B> {
        let mlm = BertMLMHeadConfig::new()
            .with_hidden_size(self.hidden_size)
            .with_initializer_range(self.initializer_range)
            .with_layer_norm_eps(self.layer_norm_eps)
            .with_vocab_size(self.vocab_size)
            .init(device);
        let nsp = BertTokenClassificationHeadConfig::new()
            .with_hidden_dropout_prob(self.hidden_dropout_prob)
            .with_hidden_size(self.hidden_size)
            .with_initializer_range(self.initializer_range)
            .with_num_classes(2)
            .init(device);

        BertPreTrainingHead { mlm, nsp }
    }
}

impl <B: Backend> BertPreTrainingHead<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> BertPreTrainingOutput<B> {
        let b = hidden.shape().dims[0];
        let mlm = self.mlm.forward(hidden.clone());
        let nsp = self.nsp.forward(hidden.slice([0..b, 0..1])).squeeze(1);

        BertPreTrainingOutput{ mlm, nsp }
    }
}

impl BertForPreTrainingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertForPreTraining<B> {
        let model = self.bert_config().init(device);
        let head = self.head_config().init(device);

        BertForPreTraining { model, head }
    }
    fn bert_config(&self) -> BertModelConfig {
        BertModelConfig::new()
            .with_attention_probs_dropout_prob(self.attention_probs_dropout_prob)
            .with_hidden_dropout_prob(self.hidden_dropout_prob)
            .with_hidden_size(self.hidden_size)
            .with_initializer_range(self.initializer_range)
            .with_intermediate_size(self.intermediate_size)
            .with_layer_norm_eps(self.layer_norm_eps)
            .with_max_position_embeddings(self.max_position_embeddings)
            .with_num_attention_heads(self.num_attention_heads)
            .with_num_hidden_layers(self.num_hidden_layers)
            .with_pad_token_id(self.pad_token_id)
            .with_type_vocab_size(self.type_vocab_size)
            .with_vocab_size(self.vocab_size)
    }
    fn head_config(&self) -> BertPreTrainingHeadConfig {
        BertPreTrainingHeadConfig::new()
            .with_hidden_dropout_prob(self.hidden_dropout_prob)
            .with_hidden_size(self.hidden_size)
            .with_initializer_range(self.initializer_range)
            .with_layer_norm_eps(self.layer_norm_eps)
            .with_type_vocab_size(self.type_vocab_size)
    }
}

impl <B: Backend> BertForPreTraining<B> {
    pub fn forward(&self, input: BertInputBatch<B>) -> BertPreTrainingOutput<B> {
        let hidden = self.model.forward(input);
        self.head.forward(hidden)
    }

    pub fn foward_classification(&self, batch: BertPreTrainingInputBatch<B>) -> ClassificationOutput<B> {
        let y_hat = self.forward(batch.input);
        
        let criterion = CrossEntropyLossConfig::new().init(&y_hat.mlm.device());
        
        let h = y_hat.mlm.shape().dims[2] as i32;
        //
        let yh_mlm = y_hat.mlm.clone().reshape([-1, h]);
        let yh_nsp = y_hat.nsp.clone();
        let y_mlm = batch.labels.mlm_labels.clone().reshape([-1]);
        let y_nsp = batch.labels.nsp_labels.clone();
        //
        let mlm_loss = criterion.forward(yh_mlm, y_mlm);
        let nsp_loss = criterion.forward(yh_nsp, y_nsp);
        let tot_loss = mlm_loss + nsp_loss;
        //
        let o_mlm = y_hat.mlm.reshape([-1, h]);
        let t_mlm = batch.labels.mlm_labels.reshape([-1]);

        // SORRY: The classification output is imho too restrictive and will not allow bi-objective optimization
        ClassificationOutput::new(tot_loss, o_mlm, t_mlm)
    }
}

impl <B: AutodiffBackend> TrainStep<BertPreTrainingInputBatch<B>, ClassificationOutput<B>> for BertForPreTraining<B> {
    fn step(&self, batch: BertPreTrainingInputBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let classification = self.foward_classification(batch);
        TrainOutput::new(self, classification.loss.backward(), classification)
    }
}
impl <B: Backend> ValidStep<BertPreTrainingInputBatch<B>, ClassificationOutput<B>> for BertForPreTraining<B> {
    fn step(&self, batch: BertPreTrainingInputBatch<B>) -> ClassificationOutput<B> {
        self.foward_classification(batch)
    }
}

#[cfg(test)]
pub mod test {
    use burn::{backend::{wgpu::WgpuDevice, Wgpu}, data::dataloader::batcher::Batcher};
    use tokenizers::Tokenizer;

    use crate::bert::BertForPreTrainingConfig;

    use super::{BertPreTrainingBatcherBuilder, BertPreTrainingInputItemBuilder};

    #[test]
    fn test_tokenizer() {
        let device = WgpuDevice::IntegratedGpu(0);
        let tok = Tokenizer::from_pretrained("xaviergillard/parti-pris-v2", None).unwrap();

        let mask_id = tok.get_added_vocabulary().get_vocab()["[MASK]"];
        let pad_id=  tok.get_added_vocabulary().get_vocab()["[PAD]"];
        
        let batcher = BertPreTrainingBatcherBuilder::<Wgpu>::default()
            .pad_token_id(pad_id)
            .mask_token_id(mask_id)
            .max_seq_length(15)
            .voc_size(tok.get_vocab_size(false))
            .device(device.clone())
            .build()
            .unwrap();

        let text1 = "Bonjour Xavier, comment vas-tu aujourd'hui ?";
        let encoded1 = tok.encode(text1, true).unwrap();
        let item1 = BertPreTrainingInputItemBuilder::default()
            .encoding(encoded1)
            .nsp_label(0)
            .build()
            .unwrap();

        let text2 = "Une phrase courte";
        let encoded2 = tok.encode(text2, true).unwrap();
        let item2 = BertPreTrainingInputItemBuilder::default()
            .encoding(encoded2)
            .nsp_label(0)
            .build()
            .unwrap();


        let batch = batcher.batch(vec![item1, item2]);

        let model = BertForPreTrainingConfig::new().init::<Wgpu>(&device);
        let output = model.foward_classification(batch);

        println!("{:?}", output.loss.into_scalar());
        println!("{:?}", output.output.shape().dims);
        println!("{:?}", output.targets.shape().dims);

    }
}