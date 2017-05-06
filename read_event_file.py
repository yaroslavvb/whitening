#!/usr/bin/env python
import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run', type=str, default='default',
                    help='name of experiment run')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

import tensorflow.tensorboard.backend.event_processing.event_file_inspector as insp

#gen = insp.generator_from_event_file(fn)
gens = insp.generators_from_logdir('runs/'+args.run)
for gen in gens:
  for event in gen:
    if (event.HasField('graph_def') or event.HasField('session_log') or
        event.HasField('meta_graph_def')):
      continue
    print(event)
