"""Per-answer timing and LLM token counts for the CLI, the web demo, and evaluation."""

import logging
import time

from llama_index.core.callbacks import TokenCountingHandler

LOGGER = logging.getLogger(__name__)


class AnswerTimer:
    """Time retrieval, the first visible chunk, and the end of the streamed answer."""

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self.started = clock()
        self.retrieved = None
        self.first_chunk = None
        self.finished = None
        self.characters = 0

    def mark_retrieved(self):
        """Call when the query engine returns: rewrite, search, and reranking are done."""
        self.retrieved = self._clock()

    def stream(self, chunks):
        """Yield chunks unchanged while recording when the answer starts and ends."""
        if self.retrieved is None:
            self.mark_retrieved()
        try:
            for chunk in chunks:
                if self.first_chunk is None and chunk.strip():
                    self.first_chunk = self._clock()
                self.characters += len(chunk)
                yield chunk
        finally:
            self.finished = self._clock()

    def summary(self):
        """Seconds for each stage; None where a stage did not happen."""

        def between(start, end):
            return None if start is None or end is None else round(end - start, 3)

        generation = between(self.first_chunk, self.finished)
        return {
            "retrieval_s": between(self.started, self.retrieved),
            "first_token_s": between(self.retrieved, self.first_chunk),
            "generation_s": generation,
            "total_s": between(self.started, self.finished),
            "characters": self.characters,
            "chars_per_s": round(self.characters / generation) if generation else None,
        }


class TokenMeter:
    """Count LLM calls and tokens for one question.

    Token counts are estimated with LlamaIndex's default tokenizer, not Qwen's own, so treat
    them as relative cost. One meter is shared per process; concurrent questions in the web
    demo would be counted together.
    """

    def __init__(self):
        self.handler = TokenCountingHandler()

    def attach(self, llm):
        manager = getattr(llm, "callback_manager", None)
        if manager is not None and self.handler not in manager.handlers:
            manager.add_handler(self.handler)
        return llm

    def reset(self):
        self.handler.reset_counts()

    def snapshot(self):
        return {
            "llm_calls": len(self.handler.llm_token_counts),
            "prompt_tokens": self.handler.prompt_llm_token_count,
            "completion_tokens": self.handler.completion_llm_token_count,
        }


TOKEN_METER = TokenMeter()


def log_answer(mode, timing, tokens):
    LOGGER.info("answer mode=%s timing=%s tokens=%s", mode, timing, tokens)


def format_seconds(value):
    return "–" if value is None else f"{value:.2f}"
