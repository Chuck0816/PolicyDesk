"""Load registered policies and attach validated business metadata before splitting."""

from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader

DATA_DIR = Path(__file__).resolve().parent / "data"

POLICY_CATEGORIES = {
    "leave_policy.txt": "leave",
    "travel_policy.txt": "travel",
    "overtime_policy.txt": "overtime",
    "salary_policy.txt": "salary",
    "promotion_policy.txt": "promotion",
    "employee_benefits_policy.txt": "benefits",
    "information_classification_policy.txt": "information_security",
    "company_hierarchy_policy.txt": "organization",
}


def parse_policy_header(text):
    """Read business fields already declared in the policy header."""
    lines = text.splitlines()
    if not lines or not lines[0].strip():
        raise ValueError("The policy title must not be empty.")

    fields = {}
    for line in lines[1:]:
        if not line.strip():
            break

        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()

    owner = fields.get("Policy owner") or fields.get("Policy owners")
    if not fields.get("Policy ID") or not fields.get("Version") or not owner:
        raise ValueError("Each policy must declare Policy ID, Version, and Policy owner.")

    return {
        "policy_title": lines[0].strip(),
        "policy_id": fields["Policy ID"],
        "version": fields["Version"],
        "owner": owner,
    }


def load_documents():
    """Load policies plus general guidance available only in all-policies searches."""
    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".txt"],
        raise_on_error=True,
    ).load_data()

    for document in documents:
        file_name = document.metadata["file_name"]

        if file_name not in POLICY_CATEGORIES:
            raise ValueError(f"Register {file_name!r} in POLICY_CATEGORIES before indexing.")

        document.metadata.update(
            {
                "doc_type": "company_policy",
                "language": "en",
                "category": POLICY_CATEGORIES[file_name],
                **parse_policy_header(document.text),
            }
        )

    documents.append(
        Document(
            text="Contact HR when the available company policies do not cover your situation.",
            metadata={
                "file_name": "manual_faq",
                "doc_type": "general_guidance",
                "language": "en",
                "category": "general",
                "policy_id": "GENERAL-FAQ",
                "policy_title": "General HR Guidance",
                "version": "1.0",
                "owner": "Human Resources",
            },
        )
    )

    return documents
