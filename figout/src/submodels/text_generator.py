import tensorflow as tf
import numpy as np
import os
import time
from framework.model import Model
import random


class txtRNN(Model):
    save_frequency = 2
    send_to_cloud_frequency = 10

    def __init__(self, rnn_units, embedding_dim, batch_size):
        super().__init__()
        self.rnn_units = rnn_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.model = None
        self.vocab_size = None
        self.idx2char = None
        self.char2idx = None
        self.start_strings = []
        self.max_length = None

    @staticmethod
    def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])
        return model

    def compile_model(self):
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.model = self.build_model(self.vocab_size, self.embedding_dim, self.rnn_units, self.batch_size)
        self.is_compiled = True
        self.model.summary()

    def train_step(self, inputs, target):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = tf.reduce_mean(self.loss(target, predictions))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def train(self, dataset, epochs):
        if not self.is_compiled:
            self.log.e("ERROR! You must compile this model before training!")
            raise Exception()

        for epoch in range(epochs):
            start = time.time()
            self.epoch = epoch
            self.model.reset_states()
            for (inputs, target) in dataset:
                self.train_step(inputs, target)
            self.epoch_callback()

            self.log.i('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    def epoch_callback(self):
        self.store.save_texts(self.generate_text())

    def preprocess_dataset(self, dataset):
        self.max_length = len(max(dataset, key=len))
        vocab = [' ']
        for entry in dataset:
            self.start_strings.append(entry[:2])
            unique_chars = set(entry)
            vocab = self._flatten_list(vocab, unique_chars)
            vocab = sorted(set(vocab))
        self.char2idx = {u: i for i, u in enumerate(vocab)}
        self.idx2char = np.array(vocab)

        self.vocab_size = len(vocab)
        descr_as_int = np.zeros((len(dataset), self.max_length), dtype=int)
        for pos, entry in enumerate(dataset):
            descr_as_int[pos, :len(entry)] = np.array([self.char2idx[c] for c in entry], dtype=int)

        text_dataset = tf.data.Dataset.from_tensor_slices(descr_as_int)

        return text_dataset.map(self._split_input_target)

    @staticmethod
    def _split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    @staticmethod
    def _flatten_list(flat_list, list2):
        for thingy in list2:
            flat_list.append(thingy)
        return flat_list

    def generate_text(self):
        generated_text = random.sample(self.start_strings, self.batch_size)
        num_generate = self.max_length
        input_eval = []
        # Converting our start string to numbers (vectorizing)
        for string in generated_text:
            input_eval.append([self.char2idx[s] for s in string])
        input_eval = tf.convert_to_tensor(input_eval)

        predicted_characters = np.empty((self.batch_size, 1), dtype=int)

        # Low temperature results in more predictable text.
        # Higher temperature results in more surprising text.
        # Experiment to find the best setting.
        temperature = 0.5
        for i in range(num_generate):
            predictions = self.model(input_eval)
            for batchnum, prediction in enumerate(predictions):
                # using a categorical distribution to predict the character returned by the model
                prediction = prediction / temperature
                predicted_id = tf.random.categorical(prediction, num_samples=1)
                predicted_id = predicted_id[-1, 0]
                predicted_characters[batchnum, ] = predicted_id
                generated_text[batchnum] += self.idx2char[predicted_id.numpy()]

            input_eval = tf.concat((input_eval, predicted_characters), axis=1)

        return generated_text
