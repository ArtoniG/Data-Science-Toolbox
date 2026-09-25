"""
src/credit_toolbox/metrics/genai.py

Deterministic evaluation metrics for Generative AI in Credit Risk.
Uses the "LLM-as-a-Judge" pattern with strict Pydantic guardrails via Instructor.
Designed to act as automated Quality Gates in multi-agent orchestration pipelines,
preventing unfaithful or toxic narratives from reaching the client or credit analyst.
"""

from typing import Any, Type, TypeVar

from pydantic import BaseModel, Field

from credit_toolbox.core.exceptions import CreditToolboxError, GenAIGuardrailError
from credit_toolbox.logging.decorators import log_execution_time

# Graceful degradation: If a client installs the core SDK without the [genai] extra,
# we shouldn't crash their entire application upon importing this module.
try:
    import instructor
except ImportError:
    instructor = None

# TypeVar for generic Pydantic model return types
T = TypeVar("T", bound=BaseModel)


# ------------------------------------------------------------------------------
# LLM-AS-A-JUDGE SCHEMAS (Deterministic Guardrails)
# ------------------------------------------------------------------------------

class FaithfulnessEvaluation(BaseModel):
    """Evaluates if an LLM's claim is strictly grounded in the provided context."""
    is_faithful: bool = Field(
        ..., 
        description="True ONLY if every part of the claim is directly supported by the context. No external knowledge allowed."
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, le=1.0, 
        description="Probability score (0.0 to 1.0) representing the judge's confidence."
    )
    reasoning: str = Field(
        ..., 
        description="Step-by-step logical deduction explaining why the claim is or is not faithful."
    )


class ToxicityEvaluation(BaseModel):
    """Evaluates text for bias, discrimination, or non-compliant language (Fair Lending/ECOA)."""
    is_toxic: bool = Field(
        ..., 
        description="True if the text contains discriminatory, biased, or aggressive language."
    )
    compliance_violation: bool = Field(
        ...,
        description="True if the text violates standard financial fair lending practices (e.g., mentioning race, gender)."
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Explanation of the specific toxic elements or compliance violations found.")


class HallucinationEvaluation(BaseModel):
    """Evaluates if the model hallucinated entities, numbers, or credit terms not present in the prompt."""
    has_hallucination: bool = Field(
        ..., 
        description="True if the text introduces specific numbers, names, or policies not grounded in the prompt."
    )
    hallucinated_entities: list[str] = Field(
        default_factory=list,
        description="List of the specific words, numbers, or entities that were hallucinated."
    )
    reasoning: str = Field(...)


# ------------------------------------------------------------------------------
# EVALUATION RUNNERS
# ------------------------------------------------------------------------------

def _ensure_genai_dependencies() -> None:
    """Fails fast if the user attempts to run AI metrics without the [genai] extra."""
    if instructor is None:
        raise CreditToolboxError(
            "The 'instructor' package is missing. Install the genai extras via: "
            "`pip install custom-credit-toolbox[genai]`"
        )


@log_execution_time
def run_quality_gate(
    client: Any, 
    model_name: str, 
    system_prompt: str, 
    user_content: str, 
    response_model: Type[T]
) -> T:
    """
    Generic execution function for LLM-as-a-Judge evaluations.
    Takes a patched Instructor client and forces a deterministic Pydantic output.
    
    Args:
        client: An LLM client already patched with `instructor.from_client()`.
        model_name: The LLM model string (e.g., 'gpt-4o', 'gemini-1.5-pro').
        system_prompt: The instructions for the judge.
        user_content: The text/claims to evaluate.
        response_model: The Pydantic class to enforce.
        
    Returns:
        An instantiated object of the requested Pydantic model.
    """
    _ensure_genai_dependencies()
    
    try:
        # Utilizing Instructor's unified API to enforce strict JSON extraction
        evaluation = client.chat.completions.create(
            model=model_name,
            response_model=response_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0, # Zero temperature is mandatory for deterministic quality gates
            max_retries=2
        )
        return evaluation
        
    except Exception as e:
        raise GenAIGuardrailError(f"Quality gate evaluation failed: {str(e)}")


@log_execution_time
def evaluate_faithfulness(client: Any, model_name: str, context: str, claim: str) -> FaithfulnessEvaluation:
    """
    Evaluates if an LLM's output (claim) hallucinated beyond the provided credit policy (context).
    """
    system_prompt = (
        "You are a strict compliance auditor for a credit risk department. "
        "Your job is to determine if a given claim is entirely faithful to the provided context. "
        "If the claim introduces outside knowledge, it is NOT faithful."
    )
    
    user_content = f"--- CONTEXT ---\n{context}\n\n--- CLAIM ---\n{claim}"
    
    return run_quality_gate(
        client=client, 
        model_name=model_name, 
        system_prompt=system_prompt, 
        user_content=user_content, 
        response_model=FaithfulnessEvaluation
    )