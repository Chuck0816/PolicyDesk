"""Answer timing stages and token counting."""

from llama_index.core.llms import MockLLM

from metrics import AnswerTimer, TokenMeter, format_seconds


class Clock:
    def __init__(self, *ticks):
        self.ticks = list(ticks)

    def __call__(self):
        return self.ticks.pop(0)


def test_timer_splits_retrieval_first_token_and_generation():
    timer = AnswerTimer(clock=Clock(0.0, 1.5, 2.0, 4.0))
    timer.mark_retrieved()
    # Whitespace-only chunks do not count as the first visible token.
    assert list(timer.stream([" ", "Hello", " world"])) == [" ", "Hello", " world"]
    assert timer.summary() == {
        "retrieval_s": 1.5,
        "first_token_s": 0.5,
        "generation_s": 2.0,
        "total_s": 4.0,
        "characters": 12,
        "chars_per_s": 6,
    }


def test_timer_without_any_answer_reports_missing_stages():
    timer = AnswerTimer(clock=Clock(0.0, 1.0, 1.0))
    assert list(timer.stream([])) == []
    summary = timer.summary()
    assert summary["first_token_s"] is None and summary["generation_s"] is None
    assert summary["chars_per_s"] is None and summary["total_s"] == 1.0


def test_timer_records_the_end_when_the_stream_fails():
    timer = AnswerTimer(clock=Clock(0.0, 1.0, 2.0, 3.0))

    def broken():
        yield "partial"
        raise RuntimeError("provider closed the stream")

    try:
        list(timer.stream(broken()))
    except RuntimeError:
        pass
    assert timer.summary()["total_s"] == 3.0


def test_token_meter_counts_llm_calls_and_resets():
    meter, llm = TokenMeter(), MockLLM(max_tokens=5)
    meter.attach(llm)
    meter.attach(llm)  # Attaching twice does not double count.
    llm.complete("How many tokens is this prompt?")
    snapshot = meter.snapshot()
    assert snapshot["llm_calls"] == 1
    assert snapshot["prompt_tokens"] > 0 and snapshot["completion_tokens"] > 0
    meter.reset()
    assert meter.snapshot() == {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    assert meter.attach(object()) is not None


def test_format_seconds():
    assert format_seconds(None) == "–" and format_seconds(1.234) == "1.23"
