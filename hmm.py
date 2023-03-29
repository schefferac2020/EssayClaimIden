from collections import defaultdict
import itertools
from scipy.special import logsumexp

import numpy as np
from tqdm import tqdm
import math


def load_data(filename):
    ner_tags = []

    #################################################################################
    # TODO: load data in ner_tags
    #################################################################################
    with open(filename) as file:
        sentences = file.read().split("\n\n")
        if (sentences[-1] == ""):
            sentences.pop()
        
        for sentence in sentences:
            sentence_list = []
            for pair in sentence.split("\n"):
                sentence_list.append(pair.split(" "))
                
            sentence_list.append(["</s>", '<end>'])
            ner_tags.append(sentence_list)
    #################################################################################

    return ner_tags


class HMMNER:
    """HMM for NER tagging."""

    def __init__(self):
        self.initial_count = None
        self.emission_count = None
        self.transition_count = None
        self.ner_tags = None
        self.observations = None
        self.tag_to_index = None
        self.observation_to_index = None
        self.initial_prob = None
        self.transition_prob = None
        self.emission_prob = None

    def get_counts(self, train_data):

        #################################################################################
        # TODO: store counts to self.initial_count, self.emission_count, self.transition_count
        # initial count
        #################################################################################
        #1. Replace infrequent words with UNK
        #Get raw word counts
        self.raw_word_cnts = defaultdict(int)
        for sentence in train_data:
            for pair in sentence:
                word = pair[0]
                self.raw_word_cnts[word] += 1
        
        #Replace words that appear <= 1 times with UNK
        for i, sentence in enumerate(train_data):
            for j, pair in enumerate(sentence):
                word = pair[0]
                if self.raw_word_cnts[word] == 1:
                    train_data[i][j][0] = "UNK"
                    self.raw_word_cnts["UNK"] += 1
                    self.raw_word_cnts.pop(word)
        
        #2. Get the initial count
        self.initial_count = defaultdict(int)
        for sentence in train_data:
            first_tag = sentence[0][1]
            self.initial_count[first_tag]+=1
        
        #3. Get the emission count
        self.emission_count = defaultdict(int)
        for sentence in train_data:
            for pair in sentence:
                pair_tuple = tuple(pair)
                self.emission_count[pair_tuple] += 1
        
        #4. Get the transition count
        self.transition_count = defaultdict(int)
        for sentence in train_data:
            last_tag = sentence[0][1]
            for j in range(1, len(sentence)):
                current_tag = sentence[j][1]
                self.transition_count[(last_tag, current_tag)] += 1
                last_tag = current_tag
        #################################################################################
    
    def get_lists(self):

        #################################################################################
        # TODO: store ner tags and vocabulary to self, store their maps to index
        #################################################################################
        tags_dict = defaultdict(int)
        observations_dict = defaultdict(int)
        
        for (observation, tag) in self.emission_count.keys():
            observations_dict[observation] += 1
            tags_dict[tag] += 1
            
        self.ner_tags = sorted(tags_dict.keys())
        self.observations = sorted(observations_dict.keys())
        
        self.tag_to_index = defaultdict(int)
        for i, tag in enumerate(self.ner_tags):
            self.tag_to_index[tag] = i
        
        self.observation_to_index = defaultdict(int)
        for i, observation in enumerate(self.observations):
            self.observation_to_index[observation] = i
        
       
        #################################################################################
    
    def get_probabilities(self, initial_k, transition_k, emission_k):
        
        #################################################################################
        # TODO: store probabilities in self.initial_prob, self.transition_prob, 
        # and self.emission_prob
        #################################################################################
        num_tags = len(self.ner_tags)
        num_unique_words = len(self.observations)
        
        # 1. Calculate initial_prob (prob that a tag is the start of sentence)
        bos_tag_counts = []
        for tag in self.ner_tags:
            tag_cnt = self.initial_count[tag]
            bos_tag_counts.append(tag_cnt)
        #k smoothing where N=num_initial_tags
        self.initial_prob = (np.array(bos_tag_counts)+initial_k) / (sum(bos_tag_counts) + initial_k*num_tags)
        
        # 2. Calculate self.transition_prob
        self.transition_prob = np.zeros((num_tags, num_tags))
        for i, tag_i in enumerate(self.ner_tags):
            # Get C(tag_i, *)
            tag_i_total_counts = 0
            for j, tag_j in enumerate(self.ner_tags):
                tag_i_total_counts += self.transition_count[(tag_i, tag_j)]
            
            for j, tag_j in enumerate(self.ner_tags):
                num = (self.transition_count[(tag_i, tag_j)] + transition_k)
                denom = tag_i_total_counts + transition_k * num_tags
                self.transition_prob[i, j] = num/denom
         
        # 3. Calculate self.emission_prob
        self.emission_prob = np.zeros((num_tags, num_unique_words))
        for i, tag_i in enumerate(self.ner_tags):
            
            # Get C(tag_i)
            tag_i_total_counts = 0
            for j, observation_j in enumerate(self.observations):
                tag_i_total_counts += self.emission_count[(observation_j, tag_i)]
            
            for j, observation_j in enumerate(self.observations):
                numer = self.emission_count[(observation_j, tag_i)] + emission_k
                denom = tag_i_total_counts + emission_k * num_unique_words
                self.emission_prob[i, j] = numer / denom

        #################################################################################

    def beam_search(self, observations, beam_width, should_print=False):
        ner_tags = []

        #################################################################################
        # TODO: predict NER tags, you can assume observations are already tokenized
        #################################################################################
        # 1. Replace words with UNK
        pp_obs = []
        for obs in observations:
            if obs in self.raw_word_cnts.keys():
                pp_obs.append(obs)
            else:
                pp_obs.append("UNK")
        #pp_obs.append("</s>")
        
        
        tags = np.zeros((beam_width, len(pp_obs)), dtype=int)
        predictions = np.zeros((beam_width, len(pp_obs)))
        
        backtrace = np.zeros((beam_width, len(pp_obs)))
        
        # 2. Do the first word/col (using INITIAL probabilties) -- There is no previous tag in this round
        tag_to_probs_dict = defaultdict(float)
        first_word = self.observation_to_index[pp_obs[0]]
        for next_tag in range(len(self.ner_tags)):
            inital_prob = self.initial_prob[next_tag]
            emmision_prob = self.emission_prob[next_tag, first_word]
            score = math.log(1) + math.log(inital_prob) + math.log(emmision_prob)
            tag_to_probs_dict[next_tag] = score
        
        #sort the top beam_width inds 
        for i, tag_index in enumerate(sorted(tag_to_probs_dict, key=tag_to_probs_dict.get, reverse=True)):
            if (i >= beam_width):
                break
            tags[i, 0] = tag_index
            predictions[i, 0] = tag_to_probs_dict[tag_index]
            #print(w, tag_to_probs_dict[w])
        
        # 3. Do the next words based on the TRANSITION probabilities
        for i in range(1, len(pp_obs)):
            tag_to_probs_dict = defaultdict(float)
            tag_to_previous_tag_row = defaultdict(int)
            
            word = self.observation_to_index[pp_obs[i]] #word here is a number
            
            active_tags = tags[:, i-1]
            for prev_tag_ind, prev_tag in enumerate(active_tags): #prev_tag here is a number

                for next_tag in range(len(self.ner_tags)): #next_tag here is a number
                    best_last_score = predictions[prev_tag_ind, i-1]
                    transition_prob = self.transition_prob[prev_tag, next_tag]
                    emmision_prob = self.emission_prob[next_tag, word]
                    score = best_last_score + math.log(transition_prob) + math.log(emmision_prob)
                                        
                    
                    #Update the best score if it is necessary
                    if next_tag not in tag_to_probs_dict.keys() or score > tag_to_probs_dict[next_tag]:
                        tag_to_probs_dict[next_tag] = score
                        tag_to_previous_tag_row[next_tag] = prev_tag_ind
                
                #sort the top beam_width inds 
                for j, tag in enumerate(sorted(tag_to_probs_dict, key=tag_to_probs_dict.get, reverse=True)):
                    if (j >= beam_width):
                        break
                    tags[j, i] = tag
                    predictions[j, i] = tag_to_probs_dict[tag]
                    backtrace[j, i] = tag_to_previous_tag_row[tag]
        
        # 4. Perform a backtrace through the table
        reversed_tag_inds = []
        #find the maximum likelihood of the last column
        
        current_col = tags.shape[1]-1
        
        
        while (current_col >= 0):
            prediction_row = np.argmax(predictions[:, -1])
            cooresponding_tag = tags[prediction_row, current_col]
            reversed_tag_inds.append(cooresponding_tag)
            
            prediction_row = backtrace[prediction_row, current_col]
            current_col -= 1
        
        reversed_tag_inds.reverse()
        
        for ind in reversed_tag_inds:
            ner_tags.append(self.ner_tags[ind])
            
        #ner_tags.pop()
        #################################################################################
        
        if should_print:
            #print("NER Tags:\n", self.ner_tags)
            print('Tag Index Matrix:\n', tags.astype(int))
            print('Backtrace Matrix:\n', backtrace.astype(int))
            #print('Predictions Matrix:\n', predictions.astype(float))

        return ner_tags

    
    def predict_ner_all(self, sentences, beam_width):
        # sentences is a list of sentences (each sentence is a list of tokens)
        results = []

        #################################################################################
        # TODO: append ner tags for each sentence to results
        #################################################################################
        for sentence in sentences:
            ner_tags = self.beam_search(sentence, beam_width)
            results.append(ner_tags)
            
        #################################################################################
        
        return results
    
    def search_k(self, val, beam_width):
        initial_k, transition_k, emission_k = 0, 0, 0
        best_acc = 0

        #################################################################################
        # TODO: search for the best combination of k values
        #################################################################################
        
        
        #Split val into sentences and labels
        sentences = []
        true_tag_sequences = []
        for sentence_with_tags in val:
            sentence = []
            true_tag_sequence = []
            for word_pair in sentence_with_tags:
                sentence.append(word_pair[0])
                true_tag_sequence.append(word_pair[1])
            sentence.append("</s>")
            true_tag_sequence.append("<end>")
            sentences.append(sentence)
            true_tag_sequences.append(true_tag_sequence)
            
            
        
        #Update hyperparameters
        initial_k, transition_k, emission_k = 0.03, 0.01, 0.2
        self.get_probabilities(initial_k, transition_k, emission_k)
        
        predicted_tag_sequences = self.predict_ner_all(sentences, beam_width)
        
        ## Search for the k values
        initial_k_poss = [0.01, 0.05, 0.1]
        transition_k_poss = [0.01, 0.05, 0.1]
        emission_k_poss = [0.01, 0.05, 0.1]
        best_acc = 0
        
        for tmp_init_k in initial_k_poss:
            for tmp_trans_k in transition_k_poss:
                for tmp_emiss_k in emission_k_poss:
                    self.get_probabilities(tmp_init_k, tmp_trans_k, tmp_emiss_k)
                    predicted_tag_sequences = self.predict_ner_all(sentences, beam_width)
                    
        
                    acc = get_accuracy(predicted_tag_sequences, true_tag_sequences)
                    if acc > best_acc:
                        best_acc = acc
                        initial_k, transition_k, emission_k = tmp_init_k, tmp_trans_k, tmp_emiss_k
                    #print(tmp_init_k, tmp_trans_k, tmp_emiss_k, best_acc)
        
        #################################################################################
        
        print(f"Best accuracy: {best_acc}")

        return initial_k, transition_k, emission_k

    def search_beam_width(self, initial_k, transition_k, emission_k, beam_widths, val):
        best_beam_width = -1
        accuracies = []
        
        #################################################################################
        # TODO: search for the best beam width
        #################################################################################
        #Split val into sentences and labels
        sentences = []
        true_tag_sequences = []
        for sentence_with_tags in val:
            sentence = []
            true_tag_sequence = []
            for word_pair in sentence_with_tags:
                sentence.append(word_pair[0])
                true_tag_sequence.append(word_pair[1])
            sentence.append("</s>")
            true_tag_sequence.append("<end>")
            sentences.append(sentence)
            true_tag_sequences.append(true_tag_sequence)
        
        self.get_probabilities(initial_k, transition_k, emission_k)
        
        best_acc = 0
        for beam_width in beam_widths:
            predicted_tag_sequences = self.predict_ner_all(sentences, beam_width)
            acc = get_accuracy(predicted_tag_sequences, true_tag_sequences)
            accuracies.append(acc)
            
            if acc > best_acc:
                best_acc = acc
                best_beam_width = beam_width
        #################################################################################

        for i in range(len(beam_widths)):
            print(f"Beamwidth = {beam_widths[i]}; Accuracy = {accuracies[i]}")

        return best_beam_width

    def test(self, initial_k, transition_k, emission_k, beam_width, test):
        accuracy = 0

        #################################################################################
        # TODO: get accuracy on the test set
        #################################################################################
        
        #Split test into sentences and labels
        sentences = []
        true_tag_sequences = []
        for sentence_with_tags in test:
            sentence = []
            true_tag_sequence = []
            for word_pair in sentence_with_tags:
                sentence.append(word_pair[0])
                true_tag_sequence.append(word_pair[1])
            sentence.append("</s>")
            true_tag_sequence.append("<end>")
            sentences.append(sentence)
            true_tag_sequences.append(true_tag_sequence)
            
        #Get Probabilities using k smoothing and predict tags
        self.get_probabilities(initial_k, transition_k, emission_k)
        
        predicted_tag_sequences = self.predict_ner_all(sentences, beam_width)
        
        #Compute the accuracy
        accuracy = get_accuracy(predicted_tag_sequences, true_tag_sequences)
        
        #################################################################################

        return accuracy

    def forward_algorithm(self, sentence):
        prob = 0

        #################################################################################
        # TODO: return the probability of sentence given the HMM you have created
        #################################################################################
        # 1. Preprocess the observations (make some UNK)
        observations = []
        for obs in sentence:
            if obs in self.raw_word_cnts.keys():
                observations.append(obs)
            else:
                observations.append("UNK")
        
        
        # 2. Initialize our matrix
        num_tags = len(self.ner_tags)
        num_observations = len(observations)
        
        probs = np.zeros((num_tags, num_observations)) 
        
        # 3. Base Cases
        first_word = sentence[0]
        first_word_ind = self.observation_to_index[first_word]
        for tag_ind, tag in enumerate(self.ner_tags):
            initial = self.initial_prob[tag_ind]
            emission = self.emission_prob[tag_ind, first_word_ind]
            probs[tag_ind, 0] = math.log(initial) + math.log(emission)
        
        # 4. Recursive Step
        for obs_ind in range(1, num_observations):
            word_string = observations[obs_ind]
            word_ind = self.observation_to_index[word_string]
            for tag_ind, tag in enumerate(self.ner_tags):
                prev_array = []
                for prev_tag_ind, prev_tag in enumerate(self.ner_tags):
                    last_log_prob = probs[prev_tag_ind, obs_ind-1]
                    transition_prob = self.transition_prob[prev_tag_ind, tag_ind]
                    emission_prob = self.emission_prob[tag_ind, word_ind]
                    
                    curr_log_prob = last_log_prob + math.log(transition_prob) + math.log(emission_prob)
                    prev_array.append(curr_log_prob)
                    
                probs[tag_ind, obs_ind] = logsumexp(prev_array)
                
        # 5. Get the final probability
        prob = logsumexp(probs[:, -1])
        
        #################################################################################

        print('Log Probability Matrix:\n', probs.astype(int))

        return prob

def get_accuracy(predictions, labels):
    accuracy = 0

    #################################################################################
    # TODO: calculate accuracy
    #################################################################################
    num_predictions = 0
    correct_predictions = 0
    for i, pred_seq in enumerate(predictions):
        for j, prediction in enumerate(pred_seq):
            num_predictions += 1
            if (prediction == labels[i][j]):
                correct_predictions += 1
    
    
    accuracy = correct_predictions / num_predictions
    #################################################################################

    return accuracy