use std::collections::{HashMap, HashSet};

use ndarray::{Array, Array2, Array3, ArrayBase, Dim, OwnedRepr};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use crate::read_harry_potters;

//struct Embedding {
//    e: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64>,
//    dim: usize,
//}
//
//const VOCAB_SIZE: usize = 20000;
//impl Embedding {
//    pub fn new(dim: usize) -> Self {
//        let a = Array::random((VOCAB_SIZE, dim), Uniform::new(0.0, 1.0).unwrap());
//
//        Self { e: a, dim }
//    }
//}

pub fn get_vocab(corpus: &[String]) -> HashMap<String, usize> {
    let mut v: HashMap<String, usize> = HashMap::new();

    for (i, line) in corpus.iter().enumerate() {
        let words: Vec<&str> = line.split_whitespace().collect();

        for (j, word) in words.iter().enumerate() {
            let word: String = word
                .chars()
                .filter(|c| !c.is_ascii_punctuation())
                .map(|c| c.to_ascii_lowercase())
                .collect();

            if let Some(occurances) = v.get(&word) {
                v.insert(word, *occurances + 1);
            } else {
                v.insert(word, 1);
            }
        }
    }

    v
}

fn filter_infrequent_words(vocab: &HashMap<String, usize>) -> HashMap<String, usize> {
    let mut new_v = HashMap::new();
    for word in vocab.iter() {
        if *word.1 > 5 {
            new_v.insert(word.0.to_string(), *word.1);
        }
    }

    new_v
}

fn create_vocab_maps(
    v: &HashMap<String, usize>,
) -> (HashMap<String, usize>, HashMap<usize, String>) {
    let mut w_to_i: HashMap<String, usize> = HashMap::new();
    let mut i_to_w: HashMap<usize, String> = HashMap::new();

    for (i, word) in v.keys().enumerate() {
        w_to_i.insert(word.to_string(), i);
        i_to_w.insert(i, word.to_string());
    }

    (w_to_i, i_to_w)
}

pub struct Model {
    output_e: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>, f32>,
    input_e: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>, f32>,
    corpus: Vec<String>,
    vocab: HashMap<String, usize>,  // With occurances
    w_to_i: HashMap<String, usize>, // Word to index
    i_to_w: HashMap<usize, String>, // Index to word
    dim: usize,
}

impl Model {
    pub fn new(corpus: &[String], dim: usize) -> Self {
        let vocab = get_vocab(corpus);
        let filtered_vocab = filter_infrequent_words(&vocab);
        let (w_to_i, i_to_w) = create_vocab_maps(&filtered_vocab);

        let vocab_size = vocab.len();
        Self {
            input_e: Array::random((vocab_size, dim), Uniform::new(-0.5, 0.5).unwrap()),
            output_e: Array::random((vocab_size, dim), Uniform::new(-0.5, 0.5).unwrap()),
            corpus: corpus.to_vec(),
            dim,
            vocab: filtered_vocab,
            w_to_i,
            i_to_w,
        }
    }
    pub fn embed(&self) {
        let vocab_size = self.vocab.len();

        //let mut a = Array::random((vocab_size, self.dim), Uniform::new(0.0, 1.0).unwrap());

        println!("Vocab: {:?}", self.w_to_i);

        println!("Vocab size: {}", vocab_size);

        let sliding_window = 1;

        for line in self.corpus.iter() {
            let words: Vec<&str> = line.split_whitespace().collect();

            for i in 0..words.len() {
                if i > 1 {
                    let word = tokenize_word(words[i]);

                    for w in 1..=sliding_window {
                        let word1 = tokenize_word(words[i - w]);
                        let word2 = tokenize_word(words[i + w]);
                        self.compute_dot_product(&word, &word1);
                        self.compute_dot_product(&word, &word2);
                    }
                }
            }
        }
    }

    pub fn train(&mut self, ephocs: usize, learning_rate: f32, window: usize, negatives: usize) {}

    fn compute_dot_product(&self, target: &str, context: &str) -> f32 {
        let target_idx = self.w_to_i.get(target).unwrap();
        let context_idx = self.w_to_i.get(context).unwrap();

        self.input_e
            .row(*target_idx)
            .dot(&self.output_e.row(*context_idx))
    }
    fn compute_positive_context(&self, word: &str) {}
}
fn tokenize_word(word: &str) -> String {
    let word: String = word
        .chars()
        .filter(|c| !c.is_ascii_punctuation())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    word
}
