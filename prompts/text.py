from langchain_core.prompts import ChatPromptTemplate


def get_formatted_report_prompt():
    """
    Generates a formatted report prompt with a unstructured text as an input.

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


def get_summary_from_query_prompt():
    """
    Generates a loose summary prompt with query and content as inputs.
    """
    loose_summary_with_content_prompt_template = """You are a helpful assistant.
Your job is to provide a summary with an input query and content.

IMPORTANT: If the content section below is empty or contains no meaningful text, you MUST return an empty string ("") with no other text.

Here is the user's query, delimited by dashes:
-----------------------------------------------------------------------------------------------------------
{query}
-----------------------------------------------------------------------------------------------------------

Here is the content, delimited by dashes:
-----------------------------------------------------------------------------------------------------------
{content}
-----------------------------------------------------------------------------------------------------------

If and only if there is actual content provided above:
1. Focus ONLY on the relationship between the initial query and the provided content
2. If there is no relationship between the query and content or if the content is empty or not meaningful, return an empty string ("")
3. Base your response ONLY on the provided content, not any other sources"""
    return ChatPromptTemplate.from_template(loose_summary_with_content_prompt_template)


def get_summary_prompt():
    """
    Generates a summary prompt with content as input.
    """
    summary_prompt_template = """You are a desk clerk.
Your job is to provide a summary of the input content in Markdown format.

IMPORTANT: If the content section below is empty or contains no meaningful text, you MUST return an empty string ("") with no other text or explanations.

Here is the content, delimited by dashes:
-----------------------------------------------------------------------------------------------------------
{content}
-----------------------------------------------------------------------------------------------------------

If and only if there is actual content provided above:
1. Write your summary in Markdown format
2. Base your summary ONLY on the provided content
3. Do not include any information from your own knowledge base
4. If the content is empty or not meaningful, just output an empty string ("")"""
    return ChatPromptTemplate.from_template(summary_prompt_template)
