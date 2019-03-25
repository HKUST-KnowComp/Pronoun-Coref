from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import threading
import tensorflow_hub as hub
import h5py

import util
from util import *
import time


class KnowledgePronounCorefModel(object):
    def __init__(self, config):
        self.config = config
        self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
        self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
        self.char_embedding_size = config["char_embedding_size"]
        self.char_dict = util.load_char_dict(config["char_vocab_path"])
        self.max_span_width = config["max_span_width"]
        self.genres = {g: i for i, g in enumerate(config["genres"])}
        self.softmax_threshold = config['softmax_threshold']
        if config["lm_path"]:
            self.lm_file = h5py.File(self.config["lm_path"], "r")
        else:
            self.lm_file = None
        self.lm_layers = self.config["lm_layers"]
        self.lm_size = self.config["lm_size"]
        self.eval_data = None  # Load eval data lazily.
        print('Start to load the eval data')
        st = time.time()
        self.load_eval_data()
        print("Finished in {:.2f}".format(time.time() - st))

        input_props = []
        input_props.append((tf.string, [None, None]))  # Tokens.
        input_props.append((tf.float32, [None, None, self.context_embeddings.size]))  # Context embeddings.
        input_props.append((tf.float32, [None, None, self.head_embeddings.size]))  # Head embeddings.
        input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers]))  # LM embeddings.
        input_props.append((tf.int32, [None, None, None]))  # Character indices.
        input_props.append((tf.int32, [None]))  # Text lengths.
        input_props.append((tf.int32, [None]))  # Speaker IDs.
        input_props.append((tf.int32, []))  # Genre.
        input_props.append((tf.bool, []))  # Is training.
        input_props.append((tf.int32, [None]))  # gold_starts.
        input_props.append((tf.int32, [None]))  # gold_ends.
        input_props.append((tf.int32, [None, None]))  # number_features.
        input_props.append((tf.int32, [None, None]))  # gender_features.
        input_props.append((tf.int32, [None, None]))  # nsubj_features.
        input_props.append((tf.int32, [None, None]))  # dobj_features.
        input_props.append((tf.int32, [None, None]))  # candidate_positions.
        input_props.append((tf.int32, [None, None]))  # pronoun_positions.
        input_props.append((tf.bool, [None, None]))  # labels
        input_props.append((tf.float32, [None, None]))  # candidate_masks

        self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
        dtypes, shapes = zip(*input_props)
        queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()

        self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.reset_global_step = tf.assign(self.global_step, 0)
        learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                                   self.config["decay_frequency"], self.config["decay_rate"],
                                                   staircase=True)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
        optimizers = {
            "adam": tf.train.AdamOptimizer,
            "sgd": tf.train.GradientDescentOptimizer
        }
        optimizer = optimizers[self.config["optimizer"]](learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

    def start_enqueue_thread(self, session):
        with open(self.config["train_path"]) as f:
            train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        def _enqueue_loop():
            while True:
                random.shuffle(train_examples)
                for example in train_examples:
                    tensorized_example = self.tensorize_pronoun_example(example, is_training=True)
                    feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                    session.run(self.enqueue_op, feed_dict=feed_dict)

        enqueue_thread = threading.Thread(target=_enqueue_loop)
        enqueue_thread.daemon = True
        enqueue_thread.start()

    def restore(self, session, log_path=None):
        # Don't try to restore unused variables from the TF-Hub ELMo module.
        vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
        saver = tf.train.Saver(vars_to_restore)
        if log_path:
            checkpoint_path = os.path.join(log_path, "model.max.ckpt")
        else:
            checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
        print("Restoring from {}".format(checkpoint_path))
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint_path)

    def load_lm_embeddings(self, doc_key):
        if self.lm_file is None:
            return np.zeros([0, 0, self.lm_size, self.lm_layers])
        file_key = doc_key.replace("/", ":")
        group = self.lm_file[file_key]
        num_sentences = len(list(group.keys()))
        sentences = [group[str(i)][...] for i in range(num_sentences)]
        lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
        for i, s in enumerate(sentences):
            lm_emb[i, :s.shape[0], :, :] = s
        return lm_emb

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def tensorize_span_labels(self, tuples, label_dict):
        if len(tuples) > 0:
            starts, ends, labels = zip(*tuples)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

    def tensorize_pronoun_example(self, example, is_training):
        gold_mentions = list()
        for pronoun_example in example['pronoun_info']:
            for tmp_np in pronoun_example['candidate_NPs']:
                if tmp_np not in gold_mentions and tmp_np[1] - tmp_np[0] < self.config["max_span_width"]:
                    gold_mentions.append(tmp_np)
            gold_mentions.append(pronoun_example['current_pronoun'])

        gold_mentions = sorted(gold_mentions)

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = util.flatten(example["speakers"])

        assert num_words == len(speakers)

        max_sentence_length = max(len(s) for s in sentences)
        max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
        text_len = np.array([len(s) for s in sentences])
        tokens = [[""] * max_sentence_length for _ in sentences]
        context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
        head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
        char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                tokens[i][j] = word
                context_word_emb[i, j] = self.context_embeddings[word]
                head_word_emb[i, j] = self.head_embeddings[word]
                char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
        tokens = np.array(tokens)

        speaker_dict = {s: i for i, s in enumerate(set(speakers))}
        speaker_ids = np.array([speaker_dict[s] for s in speakers])

        doc_key = example["doc_key"]
        genre = self.genres[doc_key[:2]]

        lm_emb = self.load_lm_embeddings(doc_key)

        num_candidate_NPs = list()
        for i, pronoun_example in enumerate(example['pronoun_info']):
            num_candidate_NPs.append(len(pronoun_example['candidate_NPs']))

        max_candidate_NP_length = max(num_candidate_NPs)

        candidate_NP_positions = np.zeros([len(example['pronoun_info']), max_candidate_NP_length])
        pronoun_positions = np.zeros([len(example['pronoun_info']), 1])
        labels = np.zeros([len(example['pronoun_info']), max_candidate_NP_length], dtype=bool)
        candidate_mask = np.zeros([len(example['pronoun_info']), max_candidate_NP_length])
        for i, pronoun_example in enumerate(example['pronoun_info']):
            for j, tmp_np in enumerate(pronoun_example['candidate_NPs']):
                candidate_mask[i, j] = 1
                for k, tmp_tuple in enumerate(gold_mentions):
                    if tmp_tuple == tmp_np:
                        candidate_NP_positions[i, j] = k
                        break

                if tmp_np in pronoun_example['correct_NPs']:
                    labels[i, j] = 1
            for k, tmp_tuple in enumerate(gold_mentions):
                if tmp_tuple == pronoun_example['current_pronoun']:
                    pronoun_positions[i, 0] = k
                    break

        number_features = np.zeros([len(example['pronoun_info']), max_candidate_NP_length])
        gender_features = np.zeros([len(example['pronoun_info']), max_candidate_NP_length])
        nsubj_features = np.zeros([len(example['pronoun_info']), max_candidate_NP_length])
        dobj_features = np.zeros([len(example['pronoun_info']), max_candidate_NP_length])
        for i, pronoun_example in enumerate(example['pronoun_info']):
            for j, tmp_np in enumerate(pronoun_example['candidate_NPs']):
                tmp_encoded_name = str(tmp_np[0]) + '$' + str(tmp_np[1])
                # print(list(pronoun_example.keys()))
                # print(pronoun_example)
                if tmp_encoded_name in pronoun_example['features']:
                    # print(pronoun_example['features'][tmp_encoded_name])
                    number_features[i, j] = pronoun_example['features'][tmp_encoded_name]['number']
                    gender_features[i, j] = pronoun_example['features'][tmp_encoded_name]['gender']
                    nsubj_features[i, j] = pronoun_example['features'][tmp_encoded_name]['nsubj']
                    dobj_features[i, j] = pronoun_example['features'][tmp_encoded_name]['dobj']
        # print(number_features)
        # print(gender_features)
        # print(nsubj_features)
        # print(dobj_features)

        example_tensors = (
            tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training,
            gold_starts, gold_ends,
            number_features, gender_features, nsubj_features, dobj_features,
            candidate_NP_positions, pronoun_positions, labels, candidate_mask)

        return example_tensors

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1),
                              tf.expand_dims(candidate_starts, 0))  # [num_labeled, num_candidates]
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1),
                            tf.expand_dims(candidate_ends, 0))  # [num_labeled, num_candidates]
        same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates]
        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        return candidate_labels

    def get_dropout(self, dropout_rate, is_training):
        return 1 - (tf.to_float(is_training) * dropout_rate)


    def get_predictions_and_loss(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len,
                                 speaker_ids, genre, is_training, gold_starts, gold_ends,
                                 number_features, gender_features, nsubj_features, dobj_features,
                                 candidate_positions, pronoun_positions,
                                 labels, candidate_mask):
        all_k = util.shape(number_features, 0)
        all_c = util.shape(number_features, 1)
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
        self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

        num_sentences = tf.shape(context_word_emb)[0]
        max_sentence_length = tf.shape(context_word_emb)[1]

        context_emb_list = [context_word_emb]
        head_emb_list = [head_word_emb]

        if self.config["char_embedding_size"] > 0:
            char_emb = tf.gather(
                tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]),
                char_index)  # [num_sentences, max_sentence_length, max_word_length, emb]
            flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2),
                                                       util.shape(char_emb,
                                                                  3)])  # [num_sentences * max_sentence_length, max_word_length, emb]
            flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config[
                "filter_size"])  # [num_sentences * max_sentence_length, emb]
            aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length,
                                                                             util.shape(flattened_aggregated_char_emb,
                                                                                        1)])  # [num_sentences, max_sentence_length, emb]
            context_emb_list.append(aggregated_char_emb)
            head_emb_list.append(aggregated_char_emb)

        if not self.lm_file:
            elmo_module = hub.Module("https://tfhub.dev/google/elmo/2")
            lm_embeddings = elmo_module(
                inputs={"tokens": tokens, "sequence_len": text_len},
                signature="tokens", as_dict=True)
            word_emb = lm_embeddings["word_emb"]  # [num_sentences, max_sentence_length, 512]
            lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                               lm_embeddings["lstm_outputs1"],
                               lm_embeddings["lstm_outputs2"]], -1)  # [num_sentences, max_sentence_length, 1024, 3]
        lm_emb_size = util.shape(lm_emb, 2)
        lm_num_layers = util.shape(lm_emb, 3)
        with tf.variable_scope("lm_aggregation"):
            self.lm_weights = tf.nn.softmax(
                tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
            self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
        flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
        flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights,
                                                                                 1))  # [num_sentences * max_sentence_length * emb, 1]
        aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
        aggregated_lm_emb *= self.lm_scaling
        if self.config['use_elmo']:
            context_emb_list.append(aggregated_lm_emb)

        context_emb = tf.concat(context_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
        head_emb = tf.concat(head_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
        context_emb = tf.nn.dropout(context_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]
        head_emb = tf.nn.dropout(head_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]

        text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)  # [num_sentence, max_sentence_length]

        context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask)  # [num_words, emb]
        num_words = util.shape(context_outputs, 0)

        genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]),
                              genre)  # [emb]

        flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask)  # [num_words]

        top_span_starts = gold_starts
        top_span_ends = gold_ends
        top_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, top_span_starts, top_span_ends)
        candidate_NP_embeddings = tf.gather(top_span_emb, candidate_positions)  # [k, max_candidate, embedding]
        candidate_starts = tf.gather(top_span_starts, candidate_positions)  # [k, max_candidate]
        pronoun_starts = tf.gather(top_span_starts, pronoun_positions)  # [k, 1]
        top_span_speaker_ids = tf.gather(speaker_ids, candidate_starts)  # [k]

        pronoun_embedding = tf.gather(top_span_emb, pronoun_positions)  # [k, embedding]
        pronoun_speaker_id = tf.gather(speaker_ids, pronoun_starts)  # [k, 1]

        mention_offsets = tf.range(util.shape(top_span_emb, 0)) + 1
        candidate_NP_offsets = tf.gather(mention_offsets, candidate_positions)
        pronoun_offsets = tf.gather(mention_offsets, pronoun_positions)
        k = util.shape(pronoun_positions, 0)
        dummy_scores = tf.zeros([k, 1])  # [k, 1]
        for i in range(self.config["coref_depth"]):
            with tf.variable_scope("coref_layer", reuse=(i > 0)):
                coreference_scores = self.get_coreference_score(candidate_NP_embeddings, pronoun_embedding,
                                                                top_span_speaker_ids,
                                                                pronoun_speaker_id, genre_emb, candidate_NP_offsets,
                                                                pronoun_offsets, number_features, gender_features,
                                                                nsubj_features,
                                                                dobj_features)  # [k, c]
        score_after_softmax = tf.nn.softmax(coreference_scores, 1)  # [k, c]
        if self.config['softmax_pruning']:
            threshold = tf.ones([all_k, all_c]) * self.config['softmax_threshold']  # [k, c]
        else:
            threshold = tf.zeros([all_k, all_c]) - tf.ones([all_k, all_c])
        ranking_mask = tf.to_float(tf.greater(score_after_softmax, threshold))  # [k, c]

        # number_features = tf.boolean_mask(number_features, ranking_mask)
        # gender_features = tf.boolean_mask(gender_features, ranking_mask)
        # nsubj_features = tf.boolean_mask(nsubj_features, ranking_mask)
        # dobj_features = tf.boolean_mask(dobj_features, ranking_mask)
        # coreference_scores = tf.boolean_mask(coreference_scores, ranking_mask)
        # labels = tf.boolean_mask(labels, ranking_mask)
        if self.config['apply_knowledge']:
            with tf.variable_scope("knowledge_layer"):
                knowledge_score, merged_score, attention_score, diagonal_mask, square_mask = self.get_knowledge_score(
                    candidate_NP_embeddings, number_features, gender_features,
                    nsubj_features,
                    dobj_features, candidate_mask * ranking_mask)  # [k, c]

            coreference_scores = coreference_scores + knowledge_score  # [k, c]
            if self.config['knowledge_pruning']:
                knowledge_score_after_softmax = tf.nn.softmax(knowledge_score, 1)  # [k, c]
                knowledge_threshold = tf.ones([all_k, all_c]) * self.config['softmax_threshold']  # [k, c]
                knowledge_ranking_mask = tf.to_float(
                    tf.greater(knowledge_score_after_softmax, knowledge_threshold))  # [k, c]
                ranking_mask = ranking_mask * knowledge_ranking_mask
        else:
            knowledge_score = tf.zeros([all_k, all_c])
            knowledge_score_after_softmax = tf.nn.softmax(knowledge_score, 1)  # [k, c]
            merged_score = tf.zeros([all_k, all_c])
            attention_score = tf.zeros([all_k, all_c])
            diagonal_mask = tf.zeros([all_k, all_c])
            square_mask = tf.zeros([all_k, all_c])

        top_antecedent_scores = tf.concat([dummy_scores, coreference_scores], 1)  # [k, c + 1]
        labels = tf.logical_and(labels, tf.greater(score_after_softmax, threshold))

        dummy_mask_1 = tf.ones([k, 1])
        dummy_mask_0 = tf.zeros([k, 1])
        mask_for_prediction = tf.concat([dummy_mask_0, candidate_mask], 1)
        ranking_mask_for_prediction = tf.concat([dummy_mask_0, ranking_mask], 1)
        if self.config['random_sample_training']:
            random_mask = tf.greater(tf.random_uniform([all_k, all_c]), tf.ones([all_k, all_c]) * 0.3)
            labels = tf.logical_and(labels, random_mask)
            ranking_mask = ranking_mask * tf.to_float(random_mask)
        dummy_labels = tf.logical_not(tf.reduce_any(labels, 1, keepdims=True))  # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, labels], 1)  # [k, c + 1]
        mask_for_training = tf.concat([dummy_mask_1, candidate_mask], 1)
        ranking_mask_for_training = tf.concat([dummy_mask_1, ranking_mask], 1)
        loss = self.softmax_loss(top_antecedent_scores * mask_for_training * ranking_mask_for_training,
                                 top_antecedent_labels)
        loss = tf.reduce_sum(loss)  # []

        return [top_antecedent_scores * mask_for_prediction * ranking_mask_for_prediction, score_after_softmax*candidate_mask], loss

    def get_knowledge_score(self, candidate_NP_embeddings, number_features, gender_features, nsubj_features,
                            dobj_features, candidate_mask):
        k = util.shape(number_features, 0)
        c = util.shape(number_features, 1)

        column_mask = tf.tile(tf.expand_dims(candidate_mask, 1), [1, c, 1])  # [k, c, c]
        row_mask = tf.tile(tf.expand_dims(candidate_mask, 2), [1, 1, c])  # [k, c, c]
        square_mask = column_mask * row_mask  # [k, c, c]

        diagonal_mask = tf.ones([k, c, c]) - tf.tile(tf.expand_dims(tf.diag(tf.ones([c])), 0), [k, 1, 1])
        # we need to find the embedding for these features
        number_emb = tf.gather(tf.get_variable("number_emb", [2, self.config["feature_size"]]),
                               number_features)  # [k, c, feature_size]
        gender_emb = tf.gather(tf.get_variable("gender_emb", [2, self.config["feature_size"]]),
                               gender_features)  # [k, c, feature_size]
        nsubj_emb = tf.gather(tf.get_variable("nsubj_emb", [10, self.config["feature_size"]]),
                              self.bucket_SP_score(nsubj_features))  # [k, c, feature_size]
        dobj_emb = tf.gather(tf.get_variable("dobj_emb", [10, self.config["feature_size"]]),
                             self.bucket_SP_score(dobj_features))  # [k, c, feature_size]

        if self.config['number']:
            number_score = self.get_feature_score(number_emb, 'number_score')  # [k, c, c, 1]
        else:
            number_score = tf.zeros([k, c, c, 1])
        if self.config['type']:
            gender_score = self.get_feature_score(gender_emb, 'gender_score')  # [k, c, c, 1]
        else:
            gender_score = tf.zeros([k, c, c, 1])
        if self.config['nsubj']:
            nsubj_score = self.get_feature_score(nsubj_emb, 'nsubj_score')  # [k, c, c, 1]
        else:
            nsubj_score = tf.zeros([k, c, c, 1])
        if self.config['dobj']:
            dobj_score = self.get_feature_score(dobj_emb, 'dobj_score')  # [k, c, c, 1]
        else:
            dobj_score = tf.zeros([k, c, c, 1])

        merged_score = tf.concat([number_score, gender_score, nsubj_score, dobj_score], 3)  # [k, c, c, 4]

        if self.config['attention']:
            if self.config['number']:
                number_attention_score = self.get_feature_attention_score(number_emb, candidate_NP_embeddings,
                                                                          'number_attention_score')
            else:
                number_attention_score = tf.ones([k, c, c, 1]) * -1000
            if self.config['type']:
                gender_attention_score = self.get_feature_attention_score(gender_emb, candidate_NP_embeddings,
                                                                          'gender_attention_score')
            else:
                gender_attention_score = tf.zeros([k, c, c, 1])
            if self.config['nsubj']:
                nsubj_attention_score = self.get_feature_attention_score(nsubj_emb, candidate_NP_embeddings,
                                                                         'nsubj_attention_score')
            else:
                nsubj_attention_score = tf.ones([k, c, c, 1]) * -1000
            if self.config['dobj']:
                dobj_attention_score = self.get_feature_attention_score(dobj_emb, candidate_NP_embeddings,
                                                                        'dobj_attention_score')
            else:
                dobj_attention_score = tf.zeros([k, c, c, 1])
            merged_attention_score = tf.concat(
                [number_attention_score, gender_attention_score, nsubj_attention_score, dobj_attention_score], 3)
            all_attention_scores = tf.nn.softmax(merged_attention_score, 3)  # [k, c, c, 4]
            all_scores = merged_score * all_attention_scores
        else:
            all_scores = merged_score
            all_attention_scores = tf.zeros([k, c, c, 4])
        all_scores = tf.reduce_sum(all_scores, 3)  # [k, c, c]
        all_scores = all_scores * diagonal_mask
        all_scores = all_scores * square_mask
        final_score = tf.reduce_mean(all_scores, 2)  # [k, c]

        return final_score, merged_score, all_attention_scores, diagonal_mask, square_mask

    def get_feature_attention_score(self, tmp_feature_emb, tmp_candidate_embedding, tmp_name):
        k = util.shape(tmp_feature_emb, 0)  # [k, c,
        c = util.shape(tmp_feature_emb, 1)
        tmp_feature_size = util.shape(tmp_feature_emb, 2)
        tmp_emb_size = util.shape(tmp_candidate_embedding, 2)
        overall_emb = tf.concat([tmp_candidate_embedding, tmp_feature_emb], 2)  # [k, c, feature_size+embedding_size]

        repeated_emb = tf.tile(tf.expand_dims(overall_emb, 1), [1, c, 1, 1])  # [k, c, c, feature_size+embedding_size]
        tiled_emb = tf.tile(tf.expand_dims(overall_emb, 2), [1, 1, c, 1])  # [k, c, c, feature_size+embedding_size]

        final_feature = tf.concat([repeated_emb, tiled_emb, repeated_emb * tiled_emb],
                                  3)  # [k, c, c, (feature_size+embedding_size)*3]
        final_feature = tf.reshape(final_feature, [k, c * c, (tmp_feature_size + tmp_emb_size) * 3])
        with tf.variable_scope(tmp_name):
            feature_attention_scores = util.ffnn(final_feature, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                                 self.dropout)  # [k, c*c, 1]
        feature_attention_scores = tf.reshape(feature_attention_scores, [k, c, c, 1])
        return feature_attention_scores

    def get_feature_score(self, tmp_feature_emb, tmp_feature_name):
        k = util.shape(tmp_feature_emb, 0)
        c = util.shape(tmp_feature_emb, 1)
        repeated_feature_emb = tf.tile(tf.expand_dims(tmp_feature_emb, 1), [1, c, 1, 1])  # [k, c, c, feature_size]
        tiled_feature_emb = tf.tile(tf.expand_dims(tmp_feature_emb, 2), [1, 1, c, 1])  # [k, c, c, feature_size]

        final_feature = tf.concat([repeated_feature_emb, tiled_feature_emb, repeated_feature_emb * tiled_feature_emb],
                                  3)  # [k, c, c, feature_size*3]
        final_feature = tf.reshape(final_feature,
                                   [k, c * c, self.config["feature_size"] * 3])  # [k, c*c, feature_size*3]

        with tf.variable_scope(tmp_feature_name):
            tmp_feature_scores = util.ffnn(final_feature, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                           self.dropout)  # [k, c*c, 1]
            tmp_feature_scores = tf.reshape(tmp_feature_scores, [k, c, c, 1])  # [k, c, c]
        return tmp_feature_scores

    def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts)  # [k, emb]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends)  # [k, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts  # [k]

        if self.config["use_features"]:
            span_width_index = span_width - 1  # [k]
            span_width_emb = tf.gather(
                tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]),
                span_width_index)  # [k, emb]
            span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts,
                                                                                                       1)  # [k, max_span_width]
            span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices)  # [k, max_span_width]
            span_text_emb = tf.gather(head_emb, span_indices)  # [k, max_span_width, emb]
            with tf.variable_scope("head_scores"):
                self.head_scores = util.projection(context_outputs, 1)  # [num_words, 1]
            span_head_scores = tf.gather(self.head_scores, span_indices)  # [k, max_span_width, 1]
            span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32),
                                       2)  # [k, max_span_width, 1]
            span_head_scores += tf.log(span_mask)  # [k, max_span_width, 1]
            span_attention = tf.nn.softmax(span_head_scores, 1)  # [k, max_span_width, 1]
            span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1)  # [k, emb]
            span_emb_list.append(span_head_emb)

        span_emb = tf.concat(span_emb_list, 1)  # [k, emb]
        return span_emb  # [k, emb]

    def get_mention_scores(self, span_emb):
        with tf.variable_scope("mention_scores"):
            return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout)  # [k, 1]

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [k, max_ant + 1]
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
        return log_norm - marginalized_gold_scores  # [k]

    def bucket_SP_score(self, sp_scores):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(sp_scores)) / math.log(2))) + 3
        use_identity = tf.to_int32(sp_scores <= 4)
        combined_idx = use_identity * sp_scores + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)

    def bucket_distance(self, distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances)) / math.log(2))) + 3
        use_identity = tf.to_int32(distances <= 4)
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)

    def get_coreference_score(self, candidate_NPs_emb, pronoun_emb, candidate_NPs_speaker_ids, pronoun_speaker_id,
                              genre_emb, candidate_NP_offsets, pronoun_offsets, number_features, gender_features,
                              nsubj_features,
                              dobj_features):
        k = util.shape(candidate_NPs_emb, 0)
        c = util.shape(candidate_NPs_emb, 1)

        feature_emb_list = []

        if self.config["use_metadata"]:
            same_speaker = tf.equal(candidate_NPs_speaker_ids, tf.tile(pronoun_speaker_id, [1, c]))  # [k, c]
            speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]),
                                         tf.to_int32(same_speaker))  # [k, c, emb]
            feature_emb_list.append(speaker_pair_emb)

            tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1])  # [k, c, emb]
            feature_emb_list.append(tiled_genre_emb)

        if self.config["use_features"]:
            antecedent_distance_buckets = self.bucket_distance(
                tf.nn.relu(tf.tile(pronoun_speaker_id, [1, c]) - candidate_NP_offsets))  # [k, c]
            antecedent_distance_emb = tf.gather(
                tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]),
                antecedent_distance_buckets)  # [c, emb]
            feature_emb_list.append(antecedent_distance_emb)
        if self.config['knowledge_as_feature']:
            number_emb = tf.gather(tf.get_variable("number_emb", [2, self.config["feature_size"]]),
                                   number_features)  # [k, c, feature_size]
            gender_emb = tf.gather(tf.get_variable("gender_emb", [2, self.config["feature_size"]]),
                                   gender_features)  # [k, c, feature_size]
            nsubj_emb = tf.gather(tf.get_variable("nsubj_emb", [10, self.config["feature_size"]]),
                                  self.bucket_SP_score(nsubj_features))  # [k, c, feature_size]
            # dobj_emb = tf.gather(tf.get_variable("dobj_emb", [10, self.config["feature_size"]]),
            #                      self.bucket_SP_score(dobj_features))  # [k, c, feature_size]
            feature_emb_list.append(number_emb)
            feature_emb_list.append(gender_emb)
            feature_emb_list.append(nsubj_emb)
            # feature_emb_list.append(dobj_emb)

        feature_emb = tf.concat(feature_emb_list, 2)  # [k, c, emb]
        feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [k, c, emb]

        target_emb = tf.tile(pronoun_emb, [1, c, 1])  # [k, c, emb]
        similarity_emb = candidate_NPs_emb * target_emb  # [k, c, emb]

        pair_emb = tf.concat([target_emb, candidate_NPs_emb, similarity_emb, feature_emb], 2)  # [k, c, emb]

        with tf.variable_scope("slow_antecedent_scores"):
            slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                               self.dropout)  # [k, c, 1]
        slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2)  # [k, c]
        return slow_antecedent_scores  # [c]

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

    def lstm_contextualize(self, text_emb, text_len, text_len_mask):
        num_sentences = tf.shape(text_emb)[0]

        current_inputs = text_emb  # [num_sentences, max_sentence_length, emb]

        for layer in range(self.config["contextualization_layers"]):
            with tf.variable_scope("layer_{}".format(layer)):
                with tf.variable_scope("fw_cell"):
                    cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences,
                                                  self.lstm_dropout)
                with tf.variable_scope("bw_cell"):
                    cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences,
                                                  self.lstm_dropout)
                state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
                state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

                (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=current_inputs,
                    sequence_length=text_len,
                    initial_state_fw=state_fw,
                    initial_state_bw=state_bw)

                text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [num_sentences, max_sentence_length, emb]
                text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
                if layer > 0:
                    highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs,
                                                                                        2)))  # [num_sentences, max_sentence_length, emb]
                    text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
                current_inputs = text_outputs

        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def load_eval_data(self):
        print('path name:', self.config["eval_path"])
        if self.eval_data is None:
            def load_line(line):
                example = json.loads(line)
                return self.tensorize_pronoun_example(example, is_training=False), example

            with open(self.config["eval_path"]) as f:
                self.eval_data = [load_line(l) for l in f.readlines()]
            num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
            print("Loaded {} eval examples.".format(len(self.eval_data)))

    def evaluate(self, session, evaluation_data=None, official_stdout=False):

        if evaluation_data:
            separate_data = list()
            for tmp_example in evaluation_data:
                separate_data.append((self.tensorize_pronoun_example(tmp_example, is_training=False), tmp_example))
        else:
            separate_data = self.eval_data

        all_coreference = 0
        predict_coreference = 0
        corrct_predict_coreference = 0

        result_by_pronoun_type = dict()
        for tmp_pronoun_type in interested_pronouns:
            result_by_pronoun_type[tmp_pronoun_type] = {'all_coreference': 0, 'predict_coreference': 0,
                                                        'correct_predict_coreference': 0}
        zero_counter = 0
        prediction_result = list()
        for example_num, (tensorized_example, example) in enumerate(separate_data):
            prediction_result_by_example = list()
            all_sentence = list()
            for s in example['sentences']:
                all_sentence += s

            _, _, _, _, _, _, _, _, _, gold_starts, gold_ends, number_features, gender_features, nsubj_features, dobj_features, candidate_NP_positions, pronoun_positions, labels, _ = tensorized_example

            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
            # pronoun_coref_scores, knowledge_score, merged_score, attention_score, diagonal_mask, square_mask = session.run(self.predictions, feed_dict=feed_dict)
            pronoun_coref_scores = session.run(
                self.predictions, feed_dict=feed_dict)

            pronoun_coref_scores = pronoun_coref_scores[0]
            gold_starts = list(gold_starts)

            for i, pronoun_coref_scores_by_example in enumerate(pronoun_coref_scores):
                current_pronoun_index = int(pronoun_positions[i][0])
                pronoun_position = gold_starts[current_pronoun_index]
                current_pronoun = all_sentence[pronoun_position:pronoun_position + 1][0]
                current_pronoun_type = get_pronoun_type(current_pronoun)
                pronoun_coref_scores_by_example = pronoun_coref_scores_by_example[1:]
                prediction_result_by_example.append(
                    (pronoun_coref_scores_by_example.tolist(), labels[i], current_pronoun_type))
                found_one = False
                for j, tmp_score in enumerate(pronoun_coref_scores_by_example.tolist()):
                    if tmp_score > 0:
                        found_one = True
                        predict_coreference += 1
                        result_by_pronoun_type[current_pronoun_type]['predict_coreference'] += 1
                        if labels[i][j]:
                            corrct_predict_coreference += 1
                            result_by_pronoun_type[current_pronoun_type]['correct_predict_coreference'] += 1
                for l in labels[i]:
                    if l:
                        all_coreference += 1
                        result_by_pronoun_type[current_pronoun_type]['all_coreference'] += 1
            prediction_result.append(prediction_result_by_example)
        for tmp_pronoun_type in interested_pronouns:
            print('Pronoun type:', tmp_pronoun_type)
            tmp_p = result_by_pronoun_type[tmp_pronoun_type]['correct_predict_coreference'] / \
                    result_by_pronoun_type[tmp_pronoun_type]['predict_coreference']
            tmp_r = result_by_pronoun_type[tmp_pronoun_type]['correct_predict_coreference'] / \
                    result_by_pronoun_type[tmp_pronoun_type]['all_coreference']
            tmp_f1 = 2 * tmp_p * tmp_r / (tmp_p + tmp_r)
            print('p:', tmp_p)
            print('r:', tmp_r)
            print('f1:', tmp_f1)
        summary_dict = {}
        if predict_coreference > 0:
            p = corrct_predict_coreference / predict_coreference
            r = corrct_predict_coreference / all_coreference
            f1 = 2 * p * r / (p + r)
            summary_dict["Average F1 (py)"] = f1
            print("Average F1 (py): {:.2f}%".format(f1 * 100))
            summary_dict["Average precision (py)"] = p
            print("Average precision (py): {:.2f}%".format(p * 100))
            summary_dict["Average recall (py)"] = r
            print("Average recall (py): {:.2f}%".format(r * 100))
        else:
            summary_dict["Average F1 (py)"] = 0
            summary_dict["Average precision (py)"] = 0
            summary_dict["Average recall (py)"] = 0
            print('there is no positive prediction')
            f1 = 0

        return util.make_summary(summary_dict), f1






