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

"""Class for evaluating programs proposed by the Sampler."""
import ast
from collections.abc import Sequence
import copy
import subprocess
import requests
from typing import Any

import code_manipulation
import config as config_lib
import programs_database
from util import LOG


class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    return ''
  code = f'def fake_function_header():\n{generated_code}'
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      code = '\n'.join(code.splitlines()[:e.lineno - 1])
  if not code:
    # Nothing could be saved from `generated_code`
    return ''

  visitor = _FunctionLineVisitor('fake_function_header')
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
  return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
  """Returns the compiled generated function and the full runnable program."""
  body = _trim_function_body(generated_code)
  if version_generated is not None:
    body = code_manipulation.rename_function_calls(
        body,
        f'{function_to_evolve}_v{version_generated}',
        function_to_evolve)

  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)
  evolved_function.body = body
  return evolved_function, str(program)


class Sandbox:
  """Sandbox for executing generated code."""

  def __init__(self, evaluation_file: str,) -> None:
    self._evaluation_file = evaluation_file

  def run(
      self,
      program: str,
      function_to_run: str,
      test_input: str,
      timeout_seconds: int,
  ) -> tuple[Any, bool]:
    """Returns `function_to_run(test_input)` and whether execution succeeded."""
    # Add call to the function with test input.
    program += f'\n\nprint({function_to_run}({test_input}))'
    # Run a python program. The entire program is being run in a sandboxed environment, so just run it locally.
    # Create a temporary python file with the program, overwriting any previous file.
    with open(self._evaluation_file, 'w') as f:
      f.write(program)
    # Run the program.
    try:
      LOG("Evaluating...")
      output = subprocess.run(['python3', self._evaluation_file], timeout=timeout_seconds, capture_output=True, check=True).stdout
      LOG("Complete")
      # Convert output to int or float if possible.
      try:
        output = int(output)
      except ValueError:
        try:
          output = float(output)
        except ValueError:
          LOG("Unable to convert output to int or float")
          return "", False
    except subprocess.TimeoutExpired:
      LOG("Timeout expired")
      return "", False
    except subprocess.CalledProcessError as e:
      LOG(f"Error: {e.stderr}")
      return "", False
    return output, True


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
  """Returns whether the generated function is calling an earlier version."""
  for name in code_manipulation.get_functions_called(program):
    # In `program` passed into this function the most recently generated
    # function has already been renamed to `function_to_evolve` (wihout the
    # suffix). Therefore any function call starting with `function_to_evolve_v`
    # is a call to an ancestor function.
    if name.startswith(f'{function_to_evolve}_v'):
      return True
  return False


class Evaluator:
  """Class that analyses functions generated by LLMs."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      template: code_manipulation.Program,
      function_to_evolve: str,
      function_to_run: str,
      inputs: Sequence[Any],
      evaluation_file: str,
      timeout_seconds: int = 60,
  ):
    self._database = database
    self._template = template
    self._function_to_evolve = function_to_evolve
    self._function_to_run = function_to_run
    self._inputs = inputs
    self._timeout_seconds = timeout_seconds
    self._sandbox = Sandbox(evaluation_file)

  def analyse(
      self,
      sample: str,
      island_id: int | None,
      version_generated: int | None,
  ) -> None:
    """Compiles the sample into a program and executes it on test inputs."""
    new_function, program = _sample_to_program(
        sample, version_generated, self._template, self._function_to_evolve)
    # Skip evaluation if the generated function is already in the database.
    if self._database.check_program_in_island(new_function, island_id):
      LOG("Program already in island")
      return
  
    # Get the best program from the database.
    program_code = self._database.get_evaluation_program(island_id)
    # Replace the pass in the body of the get_action function in Custom_Player with the new function.
    program = program.replace("return random.choice(game.available_moves()) # To be replaced.", program_code.lstrip())

    scores_per_test = {}
    for current_input in self._inputs:
      test_output, runs_ok = self._sandbox.run(
          program, self._function_to_run, current_input, self._timeout_seconds)
      if (runs_ok and not _calls_ancestor(program, self._function_to_evolve)
          and test_output is not None):
        if not isinstance(test_output, (int, float)):
          raise ValueError('@function.run did not return an int/float score.')
        scores_per_test[current_input] = test_output
      LOG(f"Runs ok: {runs_ok}")
      LOG(f"Test output: {test_output}")
      LOG(f"Call ancestor: {_calls_ancestor(program, self._function_to_evolve)}")
      LOG(f"Scores per test: {scores_per_test}")
    if scores_per_test:
      LOG(f"Registering program: {new_function}\n")
      self._database.register_program(new_function, island_id, scores_per_test)
