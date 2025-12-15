use std::{collections::HashMap, fs, process};

use rand::Rng;

fn main() {
    let corpus = fs::read_to_string("./corpus/harry-potter-1").unwrap();

    let mut chain = Chain::new();

    chain.process(&corpus);

    //println!("Values: {:?}", chain.map);

    let text = chain.predict(200);

    println!("\n \n{}", text);
}

struct Chain {
    map: HashMap<String, Vec<String>>,
}

impl Chain {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn process(&mut self, text: &str) {
        let words: Vec<&str> = text.split_whitespace().collect();

        for i in 2..words.len() {
            let first_word = words[i - 2];
            let second_word = words[i - 1];
            let next_word = words[i];

            let key = format!("{} {}", first_word, second_word);

            let prev = self.map.get_mut(&key);

            let value = if let Some(prev) = prev {
                prev.push(next_word.to_string());
                prev.to_vec()
            } else {
                vec![next_word.to_string()]
            };

            self.map.insert(key, value);
        }
    }

    fn predict(&self, len: usize) -> String {
        let mut rng = rand::rng();

        let i = rng.random_range(0..self.map.len());

        let entries: Vec<_> = self.map.keys().collect();

        let mut last_words: String = entries[i].clone();
        let mut text = String::new();

        for _ in 0..len {
            if let Some(options) = self.map.get(&last_words) {
                let o = rng.random_range(0..options.len());

                let curr_words = last_words.split_once(" ").unwrap();

                let next_word = options[o].clone();
                text.push(' ');
                text.push_str(&next_word);

                last_words = format!("{} {}", curr_words.1, next_word);
            } else {
                eprintln!("Word not found in map: |{}|", last_words);
                break;
            };
        }

        text
    }
}
