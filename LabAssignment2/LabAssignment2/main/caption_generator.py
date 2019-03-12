

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import heapq
import math

import numpy as np


class TopN(object):

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):

        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        self._data = []


class Caption(object):

    def __init__(self, sentence, state, logprob, score, metadata=None):

        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.metadata = metadata

    def __cmp__(self, other):
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score


class CaptionGenerator(object):


    def __init__(self,
                 model,
                 vocab,
                 beam_size=4,
                 max_caption_length=20,
                 length_normalization_factor=0.0):

        self.vocab = vocab
        self.model = model

        self.beam_size = beam_size
        self.max_caption_length = max_caption_length
        self.length_normalization_factor = length_normalization_factor

    def beam_search(self, encoded_image):
        # Feed in the image to get the initial state.
        partial_caption_beam = TopN(self.beam_size)
        complete_captions = TopN(self.beam_size)
        initial_state = self.model.feed_image(encoded_image)

        initial_beam = Caption(
            sentence=[self.vocab.start_id],
            state=initial_state[0],
            logprob=0.0,
            score=0.0,
            metadata=[""])

        partial_caption_beam.push(initial_beam)

        # Run beam search.
        for _ in range(self.max_caption_length - 1):
            partial_captions_list = partial_caption_beam.extract()
            partial_caption_beam.reset()
            input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
            state_feed = np.array([c.state for c in partial_captions_list])

            softmax, new_states, metadata = self.model.inference_step(input_feed,
                                                                      state_feed)

            for i, partial_caption in enumerate(partial_captions_list):
                word_probabilities = softmax[i]
                state = new_states[i]
                # For this partial caption, get the beam_size most probable next words.
                words_and_probs = list(enumerate(word_probabilities))
                words_and_probs.sort(key=lambda x: -x[1])
                words_and_probs = words_and_probs[0:self.beam_size]
                # Each next word gives a new partial caption.
                for w, p in words_and_probs:
                    if p < 1e-12:
                        continue  # Avoid log(0).
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + math.log(p)
                    score = logprob
                    if metadata:
                        metadata_list = partial_caption.metadata + [metadata[i]]
                    else:
                        metadata_list = None
                    if w == self.vocab.end_id:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence) ** self.length_normalization_factor
                        beam = Caption(sentence, state, logprob, score, metadata_list)
                        complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, state, logprob, score, metadata_list)
                        partial_caption_beam.push(beam)
            if partial_caption_beam.size() == 0:
                break


        if complete_captions.size() == 0:
            complete_captions = partial_caption_beam

        return complete_captions.extract(sort=True)
