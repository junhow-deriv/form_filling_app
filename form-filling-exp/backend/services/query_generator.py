"""
Query generation service for intelligent field-based retrieval.

Analyzes form fields and user instructions to generate targeted search queries
for waterfall retrieval (ephemeral uploads → global knowledge base).
"""

import os
import json
from typing import List, Dict
from anthropic import Anthropic
from pdf_processor import DetectedField


def get_anthropic_client() -> Anthropic:
    """Get Anthropic client for query generation."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not configured. "
            "Required for query generation."
        )
    return Anthropic(api_key=api_key)


async def generate_field_queries(
    form_fields: List[DetectedField],
    user_instructions: str,
    anthropic_api_key: str | None = None,
    has_uploaded_docs: bool = False
) -> Dict[str, List[Dict]]:
    """
    Generate targeted search queries for each form field based on user instructions.
    
    This analyzes the user's instructions to determine:
    1. Which fields should be filled
    2. Whether to use uploaded documents or knowledge base for each field
    3. What query to use for semantic search
    
    Args:
        form_fields: List of detected form fields from PDF
        user_instructions: User's natural language instructions
        anthropic_api_key: Optional API key override
        has_uploaded_docs: Whether user has uploaded context documents
        
    Returns:
        {
            "uploaded_docs_queries": [
                {"field_id": "...", "query": "...", "field_label": "..."},
                ...
            ],
            "knowledge_base_queries": [
                {"field_id": "...", "query": "...", "field_label": "..."},
                ...
            ]
        }
    
    Example:
        User says: "Fill name and email from the resume, and address from knowledge base"
        Returns:
            {
                "uploaded_docs_queries": [
                    {"field_id": "page0_name", "query": "What is the full name?", "field_label": "Full Name"},
                    {"field_id": "page0_email", "query": "What is the email address?", "field_label": "Email"}
                ],
                "knowledge_base_queries": [
                    {"field_id": "page0_address", "query": "What is the current address?", "field_label": "Address"}
                ]
            }
    """
    # Create client
    if anthropic_api_key:
        client = Anthropic(api_key=anthropic_api_key)
    else:
        client = get_anthropic_client()
    
    # Prepare form fields summary
    fields_summary = []
    for field in form_fields:
        fields_summary.append({
            "field_id": field.field_id,
            "field_label": field.friendly_label or field.label_context[:100],
            "field_type": field.field_type.value,
            "context": field.label_context[:200]  # Truncate for token efficiency
        })
    
    # Adjust prompt based on available data sources
    if has_uploaded_docs:
        source_guidance = """2. For EACH field to be filled, determine the data source:
   - "uploaded" if user mentions: uploaded/provided/resume/document/file they uploaded
   - "knowledge_base" if user mentions: knowledge base/stored data/existing info/database
   - "skip" if user doesn't mention this field at all"""
        rules_addition = "- If user says \"fill everything\" or similar, include all fields from uploaded docs"
    else:
        source_guidance = """2. For EACH field to be filled, use the knowledge base as the data source:
   - "knowledge_base" for ANY field the user wants to fill (only knowledge base is available)
   - "skip" if user doesn't mention this field at all
   
   Note: User has NO uploaded documents, so all queries should target knowledge_base."""
        rules_addition = "- Since no uploaded documents are available, ALL queries go to knowledge_base"
    
    # Create prompt for query generation
    prompt = f"""You are analyzing a form-filling request to generate semantic search queries.

Form Fields (total {len(fields_summary)}):
{json.dumps(fields_summary, indent=2)}

User Instructions:
"{user_instructions}"

Task:
1. Determine which fields the user wants to fill based on their instructions
{source_guidance}
3. Generate a concise semantic search query for each field to fill

Rules:
- Only generate queries for fields the user explicitly or implicitly asks to fill
- Be precise about data source classification
- Keep queries focused on extracting the specific information needed
{rules_addition}

Output as JSON:
{{
  "uploaded_docs_queries": [
    {{"field_id": "...", "field_label": "...", "query": "...", "reasoning": "..."}}
  ],
  "knowledge_base_queries": [
    {{"field_id": "...", "field_label": "...", "query": "...", "reasoning": "..."}}
  ]
}}

Return ONLY the JSON, no other text."""
    
    # Call Claude
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text.strip()
        
        # Extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        
        # Validate structure
        if "uploaded_docs_queries" not in result:
            result["uploaded_docs_queries"] = []
        if "knowledge_base_queries" not in result:
            result["knowledge_base_queries"] = []
        
        print(f"[QueryGen] Generated {len(result['uploaded_docs_queries'])} uploaded doc queries, "
              f"{len(result['knowledge_base_queries'])} KB queries")
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"[QueryGen] Failed to parse JSON: {e}")
        print(f"[QueryGen] Response: {response_text[:500]}")
        # Return empty queries on failure
        return {"uploaded_docs_queries": [], "knowledge_base_queries": []}
    except Exception as e:
        print(f"[QueryGen] Error generating queries: {e}")
        raise


def format_queries_for_display(queries_result: Dict[str, List[Dict]]) -> str:
    """
    Format generated queries for human-readable display.
    
    Args:
        queries_result: Output from generate_field_queries()
        
    Returns:
        Formatted string for logging/debugging
    """
    lines = []
    
    if queries_result["uploaded_docs_queries"]:
        lines.append("Uploaded Documents Queries:")
        for q in queries_result["uploaded_docs_queries"]:
            lines.append(f"  • {q['field_label']}: '{q['query']}'")
    
    if queries_result["knowledge_base_queries"]:
        lines.append("\nKnowledge Base Queries:")
        for q in queries_result["knowledge_base_queries"]:
            lines.append(f"  • {q['field_label']}: '{q['query']}'")
    
    if not queries_result["uploaded_docs_queries"] and not queries_result["knowledge_base_queries"]:
        lines.append("No queries generated (user may not want to fill any fields)")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    import asyncio
    from pdf_processor import DetectedField, FieldType
    
    async def test():
        # Mock form fields
        fields = [
            DetectedField(
                field_id="page0_name",
                field_type=FieldType.TEXT,
                bbox=(0, 0, 100, 20),
                page=0,
                label_context="Full Name:",
                friendly_label="Full Name"
            ),
            DetectedField(
                field_id="page0_email",
                field_type=FieldType.TEXT,
                bbox=(0, 30, 100, 50),
                page=0,
                label_context="Email Address:",
                friendly_label="Email"
            ),
            DetectedField(
                field_id="page0_address",
                field_type=FieldType.TEXT,
                bbox=(0, 60, 100, 80),
                page=0,
                label_context="Current Address:",
                friendly_label="Address"
            )
        ]
        
        instructions = "Fill the name and email from the resume, and get the address from knowledge base"
        
        result = await generate_field_queries(fields, instructions)
        print(format_queries_for_display(result))
    
    asyncio.run(test())

