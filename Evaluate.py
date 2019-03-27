#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import util
import model
import ujson as json


if __name__ == "__main__":
    test_data = list()
    print('Start to process data...')
    with open('test.jsonlines', 'r') as f:
        for line in f:
            tmp_example = json.loads(line)
            test_data.append(tmp_example)
    print('finish processing data')
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"


    config = util.initialize_from_env()
    model = model.KnowledgePronounCorefModel(config)

    with tf.Session() as session:
        model.restore(session)

        # print('we are working on NP-NP')
        model.evaluate(session, test_data, official_stdout=True)
        # model.evaluate(session)
        # model.evaluate_baseline_methods(session)




print('end')
