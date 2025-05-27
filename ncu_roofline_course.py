"""
 * Description: This script is used to show how to use NCU to analyze the performance bottleneck in the inference process of transformer model.
 * Usage: ncu --devices 0 --nvtx --nvtx-include "prf_prefill*" --set roofline --launch-count 4 --launch-skip 4 --kill yes --export profile_%i.ncu-rep python3 ncu_roofline_course.py
 * Date: 2025-05-26
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import nvtx
# generate random prompt with given length
def gen_prompt_ids(tokenizer, length, skip_special_tokens: bool = False):
        prompt_ids = [random.randint(0, 32000) for _ in range(length)]
        return tokenizer.decode(prompt_ids, skip_special_tokens=skip_special_tokens)

prefill_length = 800
decoding_length = 10
model_id = "/localssd/models/llama-2-7b-hf"


def myLlamaDecoderLayerForward(
self,
hidden_states: torch.Tensor,
attention_mask: Optional[torch.Tensor] = None,
position_ids: Optional[torch.LongTensor] = None,
past_key_value: Optional[Cache] = None,
output_attentions: Optional[bool] = False,
use_cache: Optional[bool] = False,
cache_position: Optional[torch.LongTensor] = None,
position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
**kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        if hidden_states.shape[-2] != 1:
                hidden_states = self.input_layernorm(hidden_states)

                # Self Attention
                nvtx.push_range("prf_prefill_attn", color="green")
                hidden_states, self_attn_weights, present_key_value = self.self_attn(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                )
                nvtx.pop_range()
                hidden_states = residual + hidden_states

                # Fully Connected
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                nvtx.push_range("prf_prefill_ffn", color="blue")
                hidden_states = self.mlp(hidden_states)
                nvtx.pop_range()
                hidden_states = residual + hidden_states

                outputs = (hidden_states,)

                if output_attentions:
                        outputs += (self_attn_weights,)

                if use_cache:
                        outputs += (present_key_value,)
        else:
                hidden_states = self.input_layernorm(hidden_states)

                # Self Attention
                nvtx.push_range("prf_decoding_attn", color="red")
                hidden_states, self_attn_weights, present_key_value = self.self_attn(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                )
                nvtx.pop_range()
                hidden_states = residual + hidden_states

                # Fully Connected
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                nvtx.push_range("prf_decoding_ffn", color="orange")
                hidden_states = self.mlp(hidden_states)
                nvtx.pop_range()
                hidden_states = residual + hidden_states

                outputs = (hidden_states,)

                if output_attentions:
                        outputs += (self_attn_weights,)

                if use_cache:
                        outputs += (present_key_value,)

        return outputs

def replace_llama_decoder_layer_forward(model):
        for name, module in model.named_modules():
                if isinstance(module, LlamaDecoderLayer):
                        module.forward = myLlamaDecoderLayerForward.__get__(module, module.__class__)
                        print(f"Replaced forward method of {name}")
        # return model

def infer(model_id, prefill_length, decoding_length):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda:0")
        replace_llama_decoder_layer_forward(model)

        prompt = gen_prompt_ids(tokenizer, prefill_length)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


        out = model.generate(**inputs, do_sample=False, max_new_tokens=1000)


if  __name__ == "__main__":
        infer(model_id, prefill_length, decoding_length)
