# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A single-threaded implementation of the FunSearch pipeline."""
from collections.abc import Sequence
from typing import Any

import code_manipulation
import config as config_lib
import evaluator
import programs_database
import sampler
from util import LOG

def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  run_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'run'))

  if len(run_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.run`.')

  evolve_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))

  if len(evolve_functions) != 2:
    raise ValueError('Expected 2 function decorated with `@funsearch.evolve`.')

  #if len(evolve_functions) != 1:
    #raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')

  #return evolve_functions[0], run_functions[0]
  return evolve_functions[0], evolve_functions[1], run_functions[0]


def main(specification: str, inputs: Sequence[Any], config: config_lib.Config, prompt_spec: str = None):
  """Launches a FunSearch experiment."""
  # Clear log file for writing output to.
  log_file = config_lib.TEST_LOG_FILE if config_lib.test_env else config_lib.LOG_FILE
  with open(log_file, 'w') as f:
    f.write('')
  LOG("Starting FunSearch experiment.")

  #function_to_evolve, function_to_run = _extract_function_names(specification)
  function_to_evolve, function_to_evolve2, function_to_run = _extract_function_names(specification)
  print("DEBUGGGGGGGGGGGGGGG")
  print("function_to_evolve: ", function_to_evolve)
  print("function_to_evolve2: ", function_to_evolve2)
  print("function_to_run: ", function_to_run)

  template = code_manipulation.text_to_program(specification)
  prompt_template = code_manipulation.text_to_program(prompt_spec)
  database = programs_database.ProgramsDatabase(
      config.programs_database, template, function_to_evolve, function_to_evolve2, prompt_template)

  evaluators = []
  for evaluation_file in config.evaluation_files:
    evaluators.append(evaluator.Evaluator(
        database,
        template,
        function_to_evolve,
        function_to_evolve2,
        function_to_run,
        inputs,
        evaluation_file,
    ))
  #print("evaluators: ", evaluators)
  # We send the initial implementation to be analysed by one of the evaluators.
  initial = prompt_template.get_function(function_to_evolve).body
  #print("initial", initial)
  evaluators[0].analyse(initial, island_id=None, version_generated=None, funcbool=False)
  
  samplers = [sampler.Sampler(database, evaluators, config.samples_per_prompt)
              for _ in range(config.num_samplers)]
  #print("Samplers: ", len(samplers))
  
  # This loop can be executed in parallel on remote sampler machines. As each
  # sampler enters an infinite loop, without parallelization only the first
  # sampler will do any work.
  for s in samplers:
    s.sample()
  
  #return best program, score



# Spec copied from cant_stop/cant_stop.py.

filepath = '../cant_stop/cant_stop.py'
with open(filepath, 'r') as file:
  _CANT_STOP_SPEC = file.read()

prompt_filepath = '../cant_stop/cant_stop_prompt_spec_small.py'
with open(prompt_filepath, 'r') as file:
  _CANT_STOP_PROMPT_SPEC = file.read()

if __name__ == '__main__':
  # The inputs field is the player to play against (0-3).
  # The prompt_spec field is what's passed to the sampler, the regular spec is what's used for evaluation.
  main(_CANT_STOP_SPEC, inputs=(0, 1, 2), config=config_lib.Config(), prompt_spec=_CANT_STOP_PROMPT_SPEC)
