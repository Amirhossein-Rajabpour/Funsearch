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

"""Class for sampling new programs."""
import asyncio
from collections.abc import Collection, Sequence

import requests
from datetime import datetime

import config as config_lib
import evaluator
import programs_database
from util import LOG

# For output from ollama.
def extract_llm_output(res: str) -> str:
  """Extracts the generated text from the LLM output."""
  LOG("Generated text start ======================")
  try:
    generated_text = res["response"]
    try:
      LOG(generated_text)
    except UnicodeEncodeError:
      # Throw it out.
      generated_text = ""
      LOG("UnicodeEncodeError")
    LOG("Generated text end ========================")
    return generated_text
  except KeyError:
    LOG(res)
    LOG("Generated text end ========================")
    # Got an error, typically model is too busy so times out.
    return ""

# For output from the HuggingFace inference API.
def extract_llm_output_hf(res: str) -> str:
  """Extracts the generated text from the LLM output."""
  LOG("Generated text start ======================")
  try:
    generated_text = res[0]["generated_text"]
    try:
      LOG(generated_text)
    except UnicodeEncodeError:
      # Throw it out.
      generated_text = ""
      LOG("UnicodeEncodeError")
    LOG("Generated text end ========================")
    return generated_text
  except KeyError:
    LOG(res)
    LOG("Generated text end ========================")
    # Got an error, typically model is too busy so times out.
    return ""

class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt
    # Remote inference.
    if (config_lib.test_env):
      self.api_url = "https://router.huggingface.co/hf-inference/models/bigcode/starcoder2-3b"
      self.headers = {"Authorization": "Bearer REDACTED", "Content-Type": "application/json", "x-use-cache": "false"}
    else:
      self.api_url = "http://REDACTED:REDACTED/api/generate"
      self.headers = {"Content-Type": "application/json"}

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""

    def query(payload):
      # Remote inference.
      try:
        response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=3000)
        return response.json()
      except requests.exceptions.Timeout:
        LOG("Timeout")
        return {}
      
    # Strip all whitespace from end of prompt.
    prompt = prompt.rstrip()
    LOG("Querying...")
    if (config_lib.test_env):
      res = query({"inputs": prompt, "parameters": {"do_sample": True, "max_new_tokens": 1000, "return_full_text": False, "stop": ["\ndef"]}, "stream": False})
    else:
      # Try to ignore code past the first function definition.
      # res = query({"model": "deepseek-coder-v2:16b", "prompt": prompt, "stream": False, "template": "{{ .Prompt }}", "options": {"num_ctx": 4096, "stop": ["\ndef", "\nclass", "\n#", "\nimport"]}})
      res = query({"model": "qwen2.5-coder:32b", "prompt": prompt, "stream": False, "template": "{{ .Prompt }}", "options": {"num_ctx": 32768, "stop": ["\ndef", "\nclass", "\n#", "\nimport"]}})
    LOG("Complete")
    return extract_llm_output_hf(res) if config_lib.test_env else extract_llm_output(res)

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt)

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    async def run_evaluator(evaluator, sample, island_id, version_generated):
      evaluator.analyse(sample, island_id, version_generated)
    
    while True:
      LOG(f"Starting iteration: {datetime.now()}")
      prompt = self._database.get_prompt()
      LOG(f"Prompt: {prompt.code[-700:]}")
      samples = self._llm.draw_samples(prompt.code)
      # This loop can be executed in parallel on remote evaluator machines.
      # Make sure number of evaluators and samples are the same.
      LOG("Running event loop")
      tasks = [run_evaluator(self._evaluators[i], sample, prompt.island_id, prompt.version_generated) for i, sample in enumerate(samples)]
      loop = asyncio.get_event_loop()
      loop.run_until_complete(asyncio.gather(*tasks))
      LOG("Event loop complete")
