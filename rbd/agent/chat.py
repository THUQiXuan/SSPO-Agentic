"""Token-exact multi-turn chat format for Qwen ChatML models (Qwen3, Qwen3.5), thinking off.

Every episode is one token sequence:

    prompt_ids  = template(system, task message) + generation prompt
    turn i      = action_ids (sampled, ends with <|im_end|>)  +  obs_ids
    obs_ids     = "\\n<|im_start|>user\\n" + observation + "<|im_end|>\\n" + generation prompt

Segments always meet at special tokens, so concatenating separately tokenized segments equals tokenizing the
whole conversation; ``check_against_template`` verifies this against the model's own chat template.
The generation prompt with thinking off is "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n".
"""
from __future__ import annotations

from typing import Dict, List, Optional


class ChatFormat:
    def __init__(self, model_path: str, enable_thinking: bool = False):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.enable_thinking = enable_thinking
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.gen_prompt_text = "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")
        self.gen_prompt_ids = self.ids(self.gen_prompt_text)

    def ids(self, text: str) -> List[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def text(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def prompt(self, system: str, user: str) -> List[int]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        txt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=self.enable_thinking)
        return self.ids(txt)

    def observation(self, obs: str, action_ids: Optional[List[int]] = None) -> List[int]:
        """Tokens that follow an assistant action: close the turn if the action was truncated, then the user
        message carrying the observation and the next generation prompt."""
        truncated = action_ids is not None and (not action_ids or action_ids[-1] != self.im_end)
        lead = "<|im_end|>\n" if truncated else "\n"  # synthesized actions (action_block) always end the turn
        return self.ids(f"{lead}<|im_start|>user\n{obs}<|im_end|>\n") + self.gen_prompt_ids

    def action_block(self, text: str) -> List[int]:
        """A synthesized assistant message (the generation prompt precedes it in the previous obs_ids)."""
        return self.ids(text) + [self.im_end]

    def hint(self, text: str) -> List[int]:
        """A complete user message with a privileged hint (no generation prompt); the teacher sequence inserts it
        right before the generation prompt of the pivot turn (rbd.pivot.synthesize)."""
        return self.ids(f"<|im_start|>user\n{text}<|im_end|>\n")

    @property
    def gen_prompt(self) -> List[int]:
        return self.gen_prompt_ids

    def check_against_template(self, system: str, user: str, action: str, obs: str) -> bool:
        """Incremental rendering == the chat template's rendering of the same 2-turn conversation."""
        inc = self.prompt(system, user) + self.action_block(action) + self.observation(obs, [self.im_end])
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user},
                {"role": "assistant", "content": action}, {"role": "user", "content": obs}]
        full = self.ids(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=self.enable_thinking))
        # the template may drop the empty think block of earlier assistant turns; compare without it
        strip = self.ids("<think>\n\n</think>\n\n")
        def drop_inner(seq: List[int]) -> List[int]:
            out, i, n = [], 0, len(strip)
            last = len(seq) - len(self.gen_prompt_ids)
            while i < len(seq):
                if i < last and seq[i:i + n] == strip:
                    i += n
                    continue
                out.append(seq[i])
                i += 1
            return out
        return drop_inner(inc) == drop_inner(full)
