import numpy as np
import collections
import re
import pickle
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

vowels = 'aeiou'
# tracking unique letter counts rather than absolute
def unique_counts(dictionary):
    combined_counts = collections.Counter()
    for individual_word in dictionary:
        char_counter = collections.Counter(individual_word)
        for character in char_counter:
            char_counter[character] = 1
            combined_counts += char_counter
    return combined_counts
# list to hold words that match the given pattern, can also be substituted with greedy dynamic matching with regex '.*' 
def substring_dic(word_index, pattern):
    matched_words = []
    pattern_length = len(pattern)
    if pattern_length in word_index:
        for word in word_index[pattern_length]:
            if re.match(pattern, word):
                matched_words.append(word)
    return matched_words
def build_dictionary(dictionary_file_location):
    text_file = open(dictionary_file_location, "r")
    full_dictionary = text_file.read().splitlines()
    text_file.close()
    return full_dictionary

def vowel_count(word):
    return (sum(1 for letter in word if letter in vowels))

class HangmanGame:
    def __init__(self, model_path, dictionary_path):
        self.model = load_model(model_path)
        self.tokenizer, self.max_seq_length = self.load_tokenizer_and_length()
        self.guessed_letters = set()
        self.tries_left = 6
        self.word_to_guess = None
        self.full_dictionary = build_dictionary(dictionary_path)

    def load_tokenizer_and_length(self):
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('max_seq_length.pickle', 'rb') as handle:
            max_seq_length = pickle.load(handle)
        return tokenizer, max_seq_length
    
    def make_a_guess(self):
        masked_word = ''.join([letter if letter in self.guessed_letters else '_' for letter in self.word_to_guess])
        input_word = masked_word.replace('_', '.')
        unguessed_count = input_word.count(".")
        unguessed_percentage = (unguessed_count / len(input_word)) * 100

        # want model to predict only at the beginning of games to leave end game scenarios to linguistic patterns/substring extraction, as model demonstrated poor performance in end-game scenarios despite targeted/weighted training
        if unguessed_percentage > 60 and self.tries_left > 4:
            word_seq = self.tokenizer.texts_to_sequences([input_word])
            word_seq = pad_sequences(word_seq, maxlen=self.max_seq_length, padding='pre', value=0)
            predictions = self.model.predict(np.array(word_seq))[0]

            predictions = {self.tokenizer.index_word[i]: predictions[i] for i in range(1, len(predictions)) if self.tokenizer.index_word[i] not in self.guessed_letters}
            if predictions:
                next_letter = max(predictions, key=predictions.get)
                if next_letter not in self.guessed_letters:
                    return next_letter
        new_dictionary = []
        for dict_word in full_dictionary:
            if len(dict_word) != len(self.word_to_guess):
                continue
            if re.match(input_word,dict_word):
                new_dictionary.append(dict_word)
        self.current_dictionary = new_dictionary      
        guess_letter = '!'
        if self.tries_left > 2:
        # second phase of guessing, relying on linguistic probability/rules and substring regex matching
        # most common letters from current dictionary with filtered length counts
            unique_common = unique_counts(self.current_dictionary).most_common()
            for letter,_ in unique_common:
                if letter not in self.guessed_letters:
                    # added vowel check due to lower likelihoods of vowels after 50% of word is vowels (rounded up from training set's 40% )
                    if letter in vowels and (vowel_count(input_word) / len(input_word)) > 0.50:
                        self.guessed_letters.append(letter)
                        continue
                    guess_letter = letter
                    break
        
        # if above fails, begin utilizing substring dictionary with the original dictionary (not filtered by length)
        if guess_letter == '!' and self.tries_left > 2:
            new_dictionary = substring_dic(word_length_index, input_word)
            c = unique_counts(new_dictionary)
            unique_common = c.most_common()
            for letter,_ in unique_common:
                if letter not in self.guessed_letters:
                    if letter in vowels and (vowel_count(input_word) / len(input_word)) > 0.50:
                        self.guessed_letters.append(letter)
                        continue
                    guess_letter = letter
                    break

        # begin splitting word further to specifically target variations of an existing word in training set e.g. 'swimming' and 'swimmingly'
        if guess_letter == '!':
            subtring_length = int(len(input_word) / 2)
            if subtring_length>=3:
                c = collections.Counter()
                for i in range(len(input_word) - subtring_length + 1):
                    s = input_word[i:i+subtring_length]
                    new_dictionary = substring_dic(word_length_index, s)
                    temp = unique_counts(new_dictionary)
                    c = c+temp
                unique_common = c.most_common()
                for letter, _ in unique_common:
                    if letter not in self.guessed_letters:
                        guess_letter = letter
                        break    

        # further split word (backup example)           
        if guess_letter == '!':
            subtring_length = int(len(input_word) / 3)
            if subtring_length>=3:
                c = collections.Counter()
                for i in range(len(input_word)-subtring_length+1):
                    s = input_word[i:i+subtring_length]
                    new_dictionary = substring_dic(word_length_index, s)
                    temp = unique_counts(new_dictionary)
                    c = c+temp
                unique_common = c.most_common()
                for letter, _ in unique_common:
                    if letter not in self.guessed_letters:
                        guess_letter = letter
                        break
        # fallback to full dictionary order if no matches
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter,_ in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break            
        
        return guess_letter

    def start_new_game(self, word):
        self.word_to_guess = word.lower()
        self.guessed_letters.clear()
        self.tries_left = 6
        print(f"New game started. The word has {len(self.word_to_guess)} letters.")
        while self.tries_left > 0 and '_' in ''.join([letter if letter in self.guessed_letters else '_' for letter in self.word_to_guess]):
            guessed_letter = self.make_a_guess()
            if guessed_letter not in self.word_to_guess:
                self.tries_left -= 1
            self.guessed_letters.add(guessed_letter)
            print(f"Outputted Guess: {guessed_letter}")
            masked_word = ''.join([letter if letter in self.guessed_letters else '_' for letter in self.word_to_guess])
            print("Current word:", ' '.join(masked_word))
            print(f"Tries Remaining: {self.tries_left}")
            if '_' not in ''.join([letter if letter in self.guessed_letters else '_' for letter in self.word_to_guess]):
                print(f"Game Ended: Model guessed the word {self.word_to_guess}.")
            if self.tries_left == 0:
                print("Model failed to guess the word")
                break
            time.sleep(2)


if __name__ == "__main__":
    print("Welcome to Brandon's Hangman Solver! This solver utilizes a LSTM-RNN model to predict the user's word.")
    dictionary_path = "words_250000_train.txt"
    full_dictionary = build_dictionary(dictionary_path)
    ## Language Rules/Substring Helper Functions
    max_length = 0
    for words in full_dictionary:
        if(len(words)>max_length):
            max_length = len(words)

    # dictionary storing lists of substrings indexed by length
    word_length_index = {length: [] for length in range(3, 30)}
    current_length = 3
    while current_length <= max_length:
        for word in full_dictionary:
            if len(word) >= current_length:
                # substring initialization
                for index in range(len(word) - current_length + 1):
                    substring = word[index:index+current_length]
                    word_length_index[current_length].append(substring)
        current_length += 1
    model_path = 'lstm_model.keras'
    game = HangmanGame(model_path,dictionary_path)

    while True:
        word = input("Enter a word to guess, or 'quit' to exit: ")
        if word.lower() == 'quit':
            break
        game.start_new_game(word)