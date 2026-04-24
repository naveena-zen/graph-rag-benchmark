"""
llm/caller.py - Groq LLM Caller
===================================
All communication with the Groq API happens here.
Handles sending prompts, receiving responses, and measuring
tokens used + time taken.
"""

import time
import sys
import groq
import config


def _get_groq_client():
    """
    Initialize the Groq client with the API key from config.
    """
    if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
        print("❌ ERROR: GROQ_API_KEY is not set in your .env file.")
        sys.exit(1)

    return groq.Groq(api_key=config.GROQ_API_KEY)


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Groq free tier cost = 0.0
    """
    return 0.0


def call_llm_baseline(question: str) -> dict:
    """
    Send a direct prompt to the Groq API (no context).
    """
    client = _get_groq_client()

    messages = [
        {"role": "system", "content": "You are a knowledgeable AI assistant specializing in technology and AI."},
        {"role": "user", "content": f"Answer the following question clearly and concisely:\n\nQuestion: {question}\n\nProvide a comprehensive but focused answer."}
    ]

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )
        elapsed = time.time() - start_time

        ans = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        return {
            "answer":        ans,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "total_tokens":  total_tokens,
            "response_time": round(elapsed, 3),
            "cost_usd":      estimate_cost(input_tokens, output_tokens),
            "model":         config.LLM_MODEL,
            "error":         None,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        print(f"❌ Groq API error: {error_msg}")
        return {
            "answer":        f"Error calling Groq API: {error_msg}",
            "input_tokens":  0,
            "output_tokens": 0,
            "total_tokens":  0,
            "response_time": round(elapsed, 3),
            "cost_usd":      0.0,
            "model":         config.LLM_MODEL,
            "error":         error_msg,
        }


def call_llm_with_graph_context(question: str, graph_context: str) -> dict:
    """
    Send a prompt to the Groq API WITH graph context.
    """
    client = _get_groq_client()

    sys_msg = (
        "You are a knowledgeable AI assistant specializing in technology and AI.\n"
        "You have been provided with relevant context retrieved from a knowledge graph.\n"
        "Use this context to give a more accurate, detailed, and well-grounded answer."
    )

    prompt = f"""=== KNOWLEDGE GRAPH CONTEXT ===
{graph_context}
=== END OF CONTEXT ===

Question: {question}

Instructions:
- Use the context above to enhance your answer
- Mention specific concepts and their relationships from the context
- If the context is highly relevant, cite it explicitly
- Provide a comprehensive and accurate answer"""

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt}
    ]

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )
        elapsed = time.time() - start_time

        ans = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        return {
            "answer":        ans,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "total_tokens":  total_tokens,
            "response_time": round(elapsed, 3),
            "cost_usd":      estimate_cost(input_tokens, output_tokens),
            "model":         config.LLM_MODEL,
            "error":         None,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        print(f"❌ Groq API error: {error_msg}")
        return {
            "answer":        f"Error calling Groq API: {error_msg}",
            "input_tokens":  0,
            "output_tokens": 0,
            "total_tokens":  0,
            "response_time": round(elapsed, 3),
            "cost_usd":      0.0,
            "model":         config.LLM_MODEL,
            "error":         error_msg,
        }
