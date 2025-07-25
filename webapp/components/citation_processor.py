#!/usr/bin/env python3
"""
Citation Processor Component
"""

import re
import logging
from openai import OpenAI
from webapp.utils.api_keys import get_api_key

logger = logging.getLogger(__name__)

def get_conversational_response(query: str, rag_results: list) -> str:
    """Generate a conversational response using RAG results with inline citations."""
    try:
        # Prepare context from RAG results
        context_parts = []
        source_mapping = {}  # Map source numbers to actual source info
        
        for i, result in enumerate(rag_results[:3]):  # Use top 3 results
            chunk = result['chunk']
            source_title = chunk.get('source_title', 'Unknown')
            year = chunk.get('year', 'Unknown')
            source_key = f"Source_{i+1}"
            
            context_parts.append(f"{source_key} ({source_title}, {year}): {chunk.get('text', '')}")
            source_mapping[source_key] = {
                'title': source_title,
                'year': year,
                'text': chunk.get('text', ''),
                'source_file': chunk.get('source_file', ''),
                'rank': i + 1
            }
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for conversational response with citation instructions
        prompt = f"""You are Michael Levin, a renowned developmental biologist and researcher. 
You are being interviewed about your research on developmental biology, collective intelligence, and bioelectricity.

Based on the following research context from your papers, provide a conversational answer to the question.
Write as if you're speaking directly to the person asking the question.

IMPORTANT: When referencing specific research findings, use inline citations in this format:
- For direct references: "In our work on [topic], we found [finding] [Source_1]"
- For general references: "Our research has shown [finding] [Source_2]"
- Use [Source_1], [Source_2], [Source_3] etc. to reference the sources provided

Research Context:
{context}

Question: {query}

Please provide a conversational response that:
1. Directly answers the question
2. Draws from the research context provided
3. Uses inline citations [Source_1], [Source_2], etc. when referencing specific findings
4. Sounds like you're speaking naturally
5. Shows your expertise and enthusiasm for the topic
6. Is informative but accessible

Response:"""

        # Generate response using OpenAI with the same API key
        api_key = get_api_key()
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are Michael Levin, a developmental biologist. Respond conversationally and draw from the provided research context. Use inline citations [Source_1], [Source_2], etc. when referencing specific findings."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content
        
        # Process the response to add hyperlinks
        processed_response = process_citations(response_text, source_mapping)
        
        return processed_response
        
    except Exception as e:
        logger.error(f"Failed to generate conversational response: {e}")
        return f"Sorry, I couldn't generate a response. Error: {e}"

def process_citations(response_text: str, source_mapping: dict) -> str:
    """Process response text to add hyperlinks for citations."""
    # Find all citation patterns like [Source_1], [Source_2], etc.
    citation_pattern = r'\[Source_(\d+)\]'
    
    def replace_citation(match):
        source_num = int(match.group(1))
        source_key = f"Source_{source_num}"
        
        if source_key in source_mapping:
            source_info = source_mapping[source_key]
            title = source_info['title']
            year = source_info['year']
            
            # Create a filename based on the title (for now)
            # In a full implementation, you'd have the actual PDF filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            pdf_filename = f"{safe_title}_{year}.pdf"
            
            # Create hyperlinks
            pdf_link = f"<a href='#' onclick='openPDF(\"{pdf_filename}\")' target='_blank'>[PDF]</a>"
            journal_link = f"<a href='https://drmichaellevin.org/publications/#{year}' target='_blank'>[Journal]</a>"
            
            return f"<sup>{pdf_link} {journal_link}</sup>"
        else:
            return match.group(0)  # Return original if source not found
    
    # Replace citations with hyperlinks
    processed_text = re.sub(citation_pattern, replace_citation, response_text)
    
    return processed_text 