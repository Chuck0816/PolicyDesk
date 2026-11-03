"""Policy-answer prompts and reusable chat template variants."""

from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts import ChatPromptTemplate, RichPromptTemplate

QA_PROMPT = PromptTemplate(
    "Answer in English using only the policy context. "
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


if __name__ == "__main__":
    for prompt in (QA_PROMPT, CHAT_PROMPT, RICH_PROMPT):
        print(type(prompt).__name__)
        print(prompt.format(context_str="Managers approve short leave.", query_str="Who approves?"))
        print()
