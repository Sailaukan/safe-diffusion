# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertModel, BertOnlyMLMHead, BertPreTrainedModel


class SafeBertForMaskedLM(BertPreTrainedModel):
    """BERT MLM backbone for SAFE-MDLM."""

    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> MaskedLMOutput:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()

        input_shape = input_ids.size()
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            input_shape,
            device=input_ids.device,
        )
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = encoder_outputs.last_hidden_state
        prediction_scores = self.cls(sequence_output)

        return MaskedLMOutput(
            logits=prediction_scores,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
