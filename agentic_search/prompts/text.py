from langchain_core.prompts import ChatPromptTemplate


def get_claims_consistency_comparison_prompt():
    """
    Get a prompt for comparing two claims.
    """
    claims_consistency_comparison_prompt_template = """Compare these claims for consistency. Output JSON only: {{"consistent": "yes" | "no"}}

CLAIM 1:
{claim_1}

CLAIM 2:
{claim_2}"""
    return ChatPromptTemplate.from_template(
        claims_consistency_comparison_prompt_template
    )


def get_content_answers_to_query_prompt():
    """
    Get a prompt for answering a query with a context.
    """
    context_answers_to_query_prompt_template = """You are a helpful assistant tasked with determining if a query can be fully answered using ONLY the provided context, without relying on any other external knowledge or your internal knowledge.

QUERY:
{query}

CONTENT, delimited by dashes:
---
{content}
---

Analyze if the query can be completely and accurately answered using ONLY the information present in the CONTENT section above.
Respond with a JSON object containing a single key "fully_answered" with value either "yes" or "no".
Example response format: {{"fully_answered": "yes"}} or {{"fully_answered": "no"}}"""
    return ChatPromptTemplate.from_template(context_answers_to_query_prompt_template)


def get_content_summary_prompt(output_json: bool = True):
    """
    Get a prompt for summarizing content.
    """
    content_summary_prompt_template = """Summarize the following content in markdown format:
---
{content}
---

Rules:
- Extract key information only
- No source attribution
- No summary prefaces
- Direct markdown output"""

    if output_json:
        content_summary_prompt_template += """\n\nOutput as JSON: {{"content": "your_summary"}}"""
    return ChatPromptTemplate.from_template(content_summary_prompt_template)


def get_formatted_report_prompt():
    """
    Get a formatted report prompt with a unstructured text as an input.

    The prompt directs the LLM to generate a comprehensive report about the provided unstructured text.
    """
    formatted_report_prompt_template = """You are an expert research analyst and technical writer.
Your task is to create a detailed, well-structured report based on the provided unstructured text.

## ANALYSIS REQUIREMENTS

1. DEPTH OF ANALYSIS
- Perform a thorough analysis of ALL major topics and subtopics
- Include relevant statistics, data points, and specific examples from the text
- Identify and explain key relationships between different concepts
- Highlight important findings and their implications

2. REPORT STRUCTURE
- Begin with an executive summary (2-3 paragraphs)
- Include a comprehensive table of contents
- Organize content into logical sections with clear hierarchical structure
- Use appropriate headings (H1 for main sections, H2 for subsections, H3 for detailed points)
- End with a conclusion section summarizing key takeaways

## FORMATTING REQUIREMENTS

1. MARKDOWN FORMAT
- Use proper Markdown syntax throughout
- Include table of contents with working links to sections
- Format code blocks, quotes, and lists appropriately
- Use bold and italic text for emphasis where relevant

2. SECTION ORGANIZATION
- Each major section should begin with a brief overview
- Use bullet points and numbered lists for better readability
- Include relevant quotes from the source text when appropriate
- Add tables or structured lists to organize complex information

3. SOURCES SECTION
- Include a "Sources" section (H2 heading) at the end
- List all URLs mentioned in the text as bullet points
- Add any specific citations or references from the input text

Here is the unstructured text, delimited by colons:
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
{unstructured_text}
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

IMPORTANT GUIDELINES:
- Use ONLY information from the provided text
- Ensure comprehensive coverage of ALL topics mentioned
- Maintain professional, clear, and concise language
- Focus on extracting and organizing maximum value from the source material
- If the text contains technical content, maintain appropriate technical depth
- Ensure all sections are properly developed with substantial content"""
    return ChatPromptTemplate.from_template(formatted_report_prompt_template)


def get_qa_summary_prompt(json_output: bool = False):
    summary_prompt_template = """You are a desk clerk. Output ONLY:
1. A markdown summary answering the query using EXCLUSIVELY the provided content, OR
2. An empty string "" if content is empty/irrelevant or query can't be answered

Query:
{query}

Content:
---
{content}
---

Rules:
1. Use ONLY information from the content - no external knowledge
2. Return "" if:
   - Content is empty/meaningless
   - Query can't be answered explicitly from content
   - Answer would require unavailable information
3. Never explain or add commentary
4. Never acknowledge inability to answer
5. Format response in markdown

Response must be either a clear markdown summary or ""."""
    if json_output:
        summary_prompt_template += """\n\nOutput your response in the following JSON format: {{"content": "your summary"}}"""
    return ChatPromptTemplate.from_template(summary_prompt_template)
