"""Policy-answer prompts and reusable chat template variants."""

from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts import ChatPromptTemplate, RichPromptTemplate

QA_PROMPT = PromptTemplate(
    "Answer using only the policy context, in the same language as the question "
    "(Chinese or English), even when the context is in another language. "
    "If the context does not answer the question, say the policy does not specify it.\n"
    "Context:\n{context_str}\nQuestion: {query_str}\nAnswer:"
)

CHAT_PROMPT = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role="system", content="Explain company policies in English."),
        ChatMessage(role="user", content="Context: {context_str}\nQuestion: {query_str}"),
    ]
)

RICH_PROMPT = RichPromptTemplate(
    "{% if context_str %}Context: {{ context_str }}{% else %}No context.{% endif %}\n"
    "Question: {{ query_str }}"
)

QUERY_REWRITE_PROMPT = PromptTemplate(
    "Rewrite the question as one search query for company policies, "
    "in the same language as the question.\n"
    "Preserve intent, negation, comparisons, dates, amounts, units, grades, and policy IDs.\n"
    "Copy existing digit sequences exactly; do not add new digit sequences.\n"
    "Do not add policy facts, assume eligibility, or answer the question.\n"
    "Return only the search query on one line.\n"
    "Question: {query}\nSearch query:"
)

ONCALL_QA_PROMPT = PromptTemplate(
    "You are an on-call assistant for an e-commerce platform. Use only the context below. "
    "Lines starting with 'Graph path' or 'Graph reasoning' come from the service dependency "
    "graph; everything else is document text. Answer in the same language as the question.\n"
    "Structure the answer as: likely root cause or direct answer, impact, steps to take, and "
    "owner team with escalation path, skipping parts the question does not need. Keep "
    "'Document evidence' (cite file and section) separate from 'Graph reasoning'. If the "
    "context is insufficient, say you are not sure and recommend escalating to the owning team. "
    "Never invent services, commands, or teams.\n"
    "Finish with one fenced json block: "
    '{"root_cause_services": [], "affected_services": [], "owner_team": null, "uncertain": false}\n'
    "Context:\n{context_str}\nQuestion: {query_str}\nAnswer:"
)


if __name__ == "__main__":
    for prompt in (QA_PROMPT, CHAT_PROMPT, RICH_PROMPT):
        print(type(prompt).__name__)
        print(prompt.format(context_str="Managers approve short leave.", query_str="Who approves?"))
        print()
