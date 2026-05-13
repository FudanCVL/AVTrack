"""Vision-Language Model skill: local Qwen3-VL and DashScope API."""

import logging
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)


class Qwen3VLChat:
    """Local Qwen3-VL model for vision-language inference."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
    ) -> None:
        """Initialize Qwen3-VL model and processor (call once, reuse).

        Args:
            model_path: Path or HuggingFace model ID for Qwen3-VL.
            dtype: Model dtype.
            attn_implementation: Attention implementation backend.
            device_map: Device mapping strategy.
        """
        logger.info("Loading Qwen3-VL from %s", model_path)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        logger.info("Qwen3-VL loaded successfully")

    @torch.no_grad()
    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 1024,
        **generate_kwargs: Any,
    ) -> str:
        """Generate response from messages.

        Args:
            messages: Qwen3-VL chat template messages.
            max_new_tokens: Maximum number of tokens to generate.
            **generate_kwargs: Extra arguments passed to model.generate().

        Returns:
            Generated text string.
        """
        # 1. Apply chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        # 2. Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        # 3. Trim prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 4. Decode
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Default batch size = 1, return single string
        return output_text[0]


class Qwen3VLPlusChat:
    """DashScope API-based Qwen3-VL-Plus for vision-language inference."""

    def __init__(
        self,
        model_name: str = "qwen3-vl-plus",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/api/v1",
    ) -> None:
        """Initialize DashScope API client.

        Args:
            model_name: DashScope model name.
            api_key: DashScope API key. If None, reads from
                DASHSCOPE_API_KEY environment variable.
            base_url: DashScope API base URL.

        Raises:
            ValueError: If DASHSCOPE_API_KEY is not set.
        """
        import dashscope

        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set in environment")

        dashscope.base_http_api_url = base_url

        self.model_name = model_name
        self._api_key = api_key

        logger.info(
            "Qwen3VLPlusChat initialized (model=%s, base_url=%s)",
            model_name,
            base_url,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Send messages to DashScope API and return response text.

        Args:
            messages: Multi-modal conversation messages.

        Returns:
            Generated text string.
        """
        import dashscope

        response = dashscope.MultiModalConversation.call(
            api_key=self._api_key,
            model=self.model_name,
            messages=messages,
        )
        return response.output.choices[0].message.content[0]["text"]
