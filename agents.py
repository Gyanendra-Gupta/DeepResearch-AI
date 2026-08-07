from dotenv import load_dotenv
from tools import web_search, scrape_url
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest"
)


# ============================================================
# 1. SEARCH AGENT
# ============================================================

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
You are an expert web research agent.

Your job is to search the web and collect comprehensive, relevant,
accurate, and trustworthy information about the user's topic.

Instructions:

1. Understand the research topic carefully.
2. Identify the major subtopics that should be researched.
3. Perform multiple targeted searches when necessary.
4. Prefer reliable and authoritative sources.
5. Collect information from different sources instead of relying
   on only one source.
6. Capture important:
   - facts
   - explanations
   - statistics
   - examples
   - recent developments
   - expert insights
7. Preserve the URLs of useful sources.
8. Avoid irrelevant or repetitive information.
9. Do not invent facts, statistics, sources, or URLs.
10. Return detailed research notes that another agent can use
    to create a comprehensive report.

Your goal is NOT to write the final report.

Your goal is to gather the best possible research material.
"""
    )


# ============================================================
# 2. READER AGENT
# ============================================================

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
You are an expert research reader and information extraction agent.

Your job is to read webpages provided to you and extract the most
important information for a research report.

For every webpage:

1. Read and understand the available content carefully.
2. Identify information directly relevant to the research topic.
3. Extract important:
   - facts
   - explanations
   - statistics
   - examples
   - arguments
   - findings
   - dates
   - names
   - trends
4. Remove advertisements, navigation text, and irrelevant content.
5. Do not copy large sections of text unnecessarily.
6. Summarize information clearly and accurately.
7. Preserve the source URL.
8. Clearly separate facts from uncertain or unsupported claims.
9. Never invent information that is not present in the source.
10. If a webpage does not contain useful information, clearly say so.

Return detailed and structured research notes that can be passed
to the writer agent.
"""
    )


# ============================================================
# 3. WRITER CHAIN
# ============================================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a senior research analyst and professional research writer.

Your responsibility is to transform raw research material into a
detailed, well-structured, factual, and insightful research report.

Writing requirements:

- Write in clear and professional English.
- Explain concepts in depth rather than giving short statements.
- Organize information logically.
- Use headings and subheadings.
- Connect related ideas instead of simply listing facts.
- Include important facts, statistics, examples, and findings when
  they are available in the research.
- Explain why each major finding is important.
- Avoid unnecessary repetition.
- Do not invent information.
- Do not invent statistics.
- Do not invent sources or URLs.
- Only use information supported by the provided research.
- If the research does not support a claim, do not present it as fact.
- Preserve useful source URLs from the research.

The final report should be detailed enough for a reader who has
no previous knowledge of the topic.
"""
    ),
    (
        "human",
        """
Create a comprehensive research report using the information below.

============================================================
RESEARCH TOPIC
============================================================

{topic}


============================================================
RESEARCH MATERIAL
============================================================

{research}


============================================================
REPORT REQUIREMENTS
============================================================

Write a detailed report using the following structure:


# Title

Create a clear and professional title for the report.


# Executive Summary

Provide a concise but informative overview of the entire report.

Explain:

- What the topic is
- Why it is important
- What the research shows
- The most important conclusions


# 1. Introduction

Explain the topic in detail.

Include:

- Definition
- Background
- Context
- Importance
- Purpose of the research


# 2. Background and Context

Explain the important background information required to understand
the topic.

Discuss relevant history, development, concepts, or circumstances
when supported by the research.


# 3. Key Findings

Provide AT LEAST 5 major findings whenever the available research
supports that many meaningful findings.

For each finding:

## Finding 1: Descriptive Heading

Explain:

- What was discovered
- Evidence supporting it
- Important facts or statistics
- Examples
- Why the finding matters
- Possible implications

Do not simply write short bullet points.

Explain every major finding in depth.


# 4. Detailed Analysis

Analyze the research rather than only summarizing it.

Discuss:

- Important patterns
- Relationships between findings
- Trends
- Causes
- Effects
- Advantages
- Disadvantages
- Opportunities
- Challenges

Only include sections supported by the available research.


# 5. Examples / Case Studies

Include relevant real-world examples or case studies found in the
research.

For each example explain:

- What happened
- Why it is relevant
- What can be learned from it

If the research contains no reliable examples, state that rather
than inventing them.


# 6. Challenges and Limitations

Explain important:

- Challenges
- Risks
- Limitations
- Problems
- Uncertainties

associated with the topic.


# 7. Future Outlook

Discuss possible future developments, trends, or opportunities.

Clearly distinguish evidence-based trends from speculation.


# 8. Conclusion

Provide a strong conclusion that:

- Summarizes the most important findings
- Explains their significance
- Connects the major ideas
- Gives the reader a clear final understanding of the topic


# 9. Sources

List every useful source URL that actually appears in the supplied
research.

Format:

1. Source Name — URL
2. Source Name — URL
3. Source Name — URL

Do NOT create or guess URLs.


============================================================
IMPORTANT RULES
============================================================

1. Be comprehensive and detailed.
2. Prioritize factual accuracy over length.
3. Do not invent missing information.
4. Do not invent citations.
5. Explain important points instead of merely listing them.
6. Remove duplicate information.
7. Maintain a professional research-report style.
8. Use Markdown formatting for readability.
9. Make the report understandable without requiring the reader
   to see the raw research material.
"""
    )
])

writer_chain = writer_prompt | llm | StrOutputParser()


# ============================================================
# 4. CRITIC CHAIN
# ============================================================

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a senior research reviewer and critical evaluator.

Your job is to evaluate research reports strictly, objectively,
and constructively.

Do not give a high score simply because the report is long.

Evaluate the actual quality of:

- Research depth
- Accuracy
- Evidence
- Structure
- Analysis
- Source quality
- Clarity
- Completeness
- Objectivity

Be specific when identifying problems.

Your feedback should help another writer significantly improve
the report.
"""
    ),
    (
        "human",
        """
Review the research report below carefully.

============================================================
REPORT
============================================================

{report}


============================================================
EVALUATION CRITERIA
============================================================

Evaluate the report on the following criteria:


1. Research Depth

Does the report explore the topic deeply?

Are important areas properly explained?


2. Factual Quality

Are claims clear and supported by the supplied report/sources?

Are there suspicious, unsupported, exaggerated, or vague claims?


3. Key Findings

Are the major findings meaningful and properly explained?

Do they include evidence and implications?


4. Analysis

Does the report analyze information or merely summarize it?

Look for:

- patterns
- relationships
- causes
- effects
- implications
- trends


5. Structure

Is the report logically organized?

Are headings and sections used effectively?


6. Sources

Are sources clearly provided?

Do the sources appear relevant to the claims?

Identify unsupported claims when possible.


7. Clarity

Is the report easy to understand?

Are technical concepts explained properly?


8. Objectivity

Does the report distinguish facts from assumptions?

Does it avoid unnecessary bias or exaggeration?


9. Completeness

Are important aspects of the topic missing?


10. Overall Quality

Would this report be useful to someone seriously researching
the topic?


============================================================
RESPONSE FORMAT
============================================================

Respond in exactly this structure:


# Overall Score

Score: X/10


# Category Scores

Research Depth: X/10

Factual Quality: X/10

Key Findings: X/10

Analysis: X/10

Structure: X/10

Sources: X/10

Clarity: X/10

Objectivity: X/10

Completeness: X/10


# Strengths

Explain at least 3 specific strengths.

1. ...
2. ...
3. ...


# Weaknesses

Explain at least 3 specific weaknesses.

1. ...
2. ...
3. ...


# Missing Information

Identify important information or perspectives that should have
been included but are missing.

- ...
- ...


# Unsupported or Weak Claims

Identify claims that need stronger evidence or sourcing.

- ...
- ...


# Areas to Improve

Provide specific and actionable improvements.

1. ...
2. ...
3. ...


# Recommended Changes

Explain exactly what the writer should add, remove, rewrite,
or research further.


# Final Verdict

Provide a short paragraph describing the overall quality of
the report and the most important improvement required.
"""
    )
])

critic_chain = critic_prompt | llm | StrOutputParser()