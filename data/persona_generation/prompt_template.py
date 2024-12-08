PERSONA_GENERATION = """Generate diverse user personas from an African context based on a given piece of text, such as a Wikipedia page or a news article, focusing on individuals who might be interested in the content.

- Analyze the key topics and content within the text with an emphasis on cultural, social, and economic contexts of Africa.
- Identify potential interests, needs, or challenges that a variety of users from diverse African backgrounds might have in relation to the text.
- Consider comprehensive demographic and psychographic characteristics such as age, profession, interests, ethnicity, cultural contexts, etc.

# Steps

1. **Content Analysis**: Break down the text to identify primary topics, themes, and connections to African contexts.
2. **Persona Generation**: For each theme, create diverse personas by:
   - Specifying a broad range of demographic characteristics, avoiding repetitive structures
   - Identifying relevant professions or industries prevalent in Africa
   - Outlining specific interests or hobbies related to the text and befitting diverse African cultures
   - Highlighting potential challenges or motivations relating to the content and African context
3. **Persona Detailing**: Write a brief description for each persona, including how they would engage with the content and their potential impact or interaction within the African context.

# Output Format

Express each user persona in a structured JSON format with the following attributes:
```json
{
  "countries": ["Kenya"],
  "languages": ["Swahili"],
  "persona": "A passionate advocate for digital innovation in Kenya, exploring solutions in mobile finance, deeply rooted in cultural traditions."
}
```
Include languages and countries as lists, and ensure the persona is detailed and contextually relevant.

# Examples

**Example Input:** Short excerpt from a text about renewable energy developments.

**Example Output:**
```json
[
  {
    "countries": ["South Africa"],
    "languages": ["English"],
    "persona": "An environmental enthusiast from South Africa, fervent about eco-friendly innovations and promoting sustainable practices in local communities."
  }
]
```

(Real examples should include 2-3 personas per text with varied demographic details, emphasizing diverse African contexts.)

# Notes

- Consider both direct and indirect interests, such as professionals in related fields or laypeople with hobbies connected to the topic.
- Aim for a broad spectrum, covering different age ranges, cultural backgrounds, socio-economic statuses, and levels of familiarity with the topic, specifically within African societies.
"""