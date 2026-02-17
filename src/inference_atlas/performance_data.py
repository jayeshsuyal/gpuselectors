"""Bundled model performance defaults for runtime-safe imports."""

from __future__ import annotations

MODEL_REQUIREMENTS = {
    "llama_8b": {
        "display_name": "Llama 3.1 8B",
        "recommended_memory_gb": 16,
        "parameter_count": 8_000_000_000,
    },
    "llama_70b": {
        "display_name": "Llama 3.1 70B",
        "recommended_memory_gb": 80,
        "parameter_count": 70_000_000_000,
    },
    "llama_405b": {
        "display_name": "Llama 3.1 405B",
        "recommended_memory_gb": 400,
        "parameter_count": 405_000_000_000,
    },
    "mixtral_8x7b": {
        "display_name": "Mixtral 8x7B",
        "recommended_memory_gb": 90,
        "parameter_count": 47_000_000_000,
    },
    "mistral_7b": {
        "display_name": "Mistral 7B",
        "recommended_memory_gb": 16,
        "parameter_count": 7_000_000_000,
    },
    "qwen2_5_7b": {
        "display_name": "Qwen 2.5 7B",
        "recommended_memory_gb": 16,
        "parameter_count": 7_000_000_000,
    },
    "qwen2_5_72b": {
        "display_name": "Qwen 2.5 72B",
        "recommended_memory_gb": 80,
        "parameter_count": 72_000_000_000,
    },
    "gpt_4o_mini": {
        "display_name": "GPT-4o Mini",
        "recommended_memory_gb": 16,
        "parameter_count": 8_000_000_000,
    },
    "gpt_4o": {
        "display_name": "GPT-4o",
        "recommended_memory_gb": 80,
        "parameter_count": 70_000_000_000,
    },
    "claude_haiku_4_5": {
        "display_name": "Claude Haiku 4.5",
        "recommended_memory_gb": 16,
        "parameter_count": 8_000_000_000,
    },
    "claude_sonnet_4_5": {
        "display_name": "Claude Sonnet 4.5",
        "recommended_memory_gb": 80,
        "parameter_count": 70_000_000_000,
    },
    "claude_opus_4_5": {
        "display_name": "Claude Opus 4.5",
        "recommended_memory_gb": 400,
        "parameter_count": 405_000_000_000,
    },
    "deepseek_v3": {
        "display_name": "DeepSeek V3",
        "recommended_memory_gb": 80,
        "parameter_count": 70_000_000_000,
    },
    "deepseek_r1": {
        "display_name": "DeepSeek R1",
        "recommended_memory_gb": 80,
        "parameter_count": 70_000_000_000,
    },
    "gemma_2_27b": {
        "display_name": "Gemma 2 27B",
        "recommended_memory_gb": 48,
        "parameter_count": 27_000_000_000,
    },
    "llama_3_1_70b_instruct": {
        "display_name": "Llama 3.1 70B Instruct",
        "recommended_memory_gb": 80,
        "parameter_count": 70_000_000_000,
    },
    "llama_3_1_8b_instruct": {
        "display_name": "Llama 3.1 8B Instruct",
        "recommended_memory_gb": 16,
        "parameter_count": 8_000_000_000,
    },
}
