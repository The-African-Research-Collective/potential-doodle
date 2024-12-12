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
- Respond with the personas in English text, reflecting the diverse linguistic landscape of Africa
"""

PERSONA_2_PERSONA_GENERATION = """Generate a list of 3 personas for each input persona in JSON only, considering interpersonal relationships and cultural diversity within Africa.

Identify new personas that have interpersonal relationships with existing personas, and generate additional personas that are similar but from different communities or countries in Africa.

## Steps

1. **Analyze the Given Persona**: 
   - Identify key characteristics such as country, language, role, and interests.
   - Understand potential interpersonal relationships based on shared activities or values.

2. **Generate Interpersonal Personas**:
   - Create personas that could interact or share goals with the given persona. Consider roles that would naturally support or collaborate with their interests or goals.

3. **Create Similar Personas from Different African Communities**:
   - Maintain the essence of the original persona but adapt cultural, geographic, or linguistic attributes to fit another African context.

4. **Diversity of Contexts**: 
   - Ensure a variety of regions, languages, and cultural perspectives are represented to reflect diverse African contexts.

# Output Format

- Provide a list of 3 personas in JSON format, each with fields: "countries", "languages", and "persona".
- Ensure that each persona includes a narrative description in the "persona" field that captures the relationships or adapted cultural details.

# Examples

## Given Persona:
```json
{
  "countries": ["Kenya"],
  "languages": ["Swahili"],
  "persona": "A passionate advocate for digital innovation in Kenya, exploring solutions in mobile finance, deeply rooted in cultural traditions."
}
```
### Generated Personas (3 examples):
```json
[
  {
    "countries": ["Kenya"],
    "languages": ["Swahili"],
    "persona": "An experienced mobile app developer in Kenya, collaborating with digital innovators to create culturally relevant financial solutions."
  },
  {
    "countries": ["South Africa"],
    "languages": ["Zulu"],
    "persona": "A forward-thinking entrepreneur in South Africa, bridging the tech divide with localized content creation and digital financial tools."
  },
  {
    "countries": ["Nigeria"],
    "languages": ["Yoruba"],
    "persona": "An enthusiastic promoter of technological adaptation in Nigeria, researching community-based mobile applications, grounded in traditional customs."
  }
]
```

# Notes

- When creating interpersonal personas, consider roles such as collaborators, supporters, or community leaders.
- When generating similar personas in different countries, adapt the technological focus and community values to local contexts.
- Aim for realism and cultural specificity in each persona.

"""

PROMPT_GENERATION = """Create a verifiable prompt  that a given user persona that is a native speaker of Yoruba might ask you to do 

A 40-year-old IT consultant and hobbyist programmer who appreciates the humorous aspects of programming and often uses esoteric languages to create playful software for local tech meetups.

An example of a verifiable instruction in Yoruba could be:
Aha ndị na-atọ ụtọ maka mgbasa ozi mmekọrịta. Nke mbụ, megharịa arịrịọ ahụ n'enweghị mgbanwe, wee nye azịza gị (ekwula ihe ọ bụla tupu ịmeghachi arịrịọ ahụ; arịrịọ ịchọrọ ịmeghachi anaghị agụnye ahịrịokwu a)

Note:
1. The above example is not tied to any particular persona, but you should create one that is unique and specific to the given persona.
2. The instruction should contain all the verifiable contraint(s)
3. Your output should start with "User instruction:" your output should not include the answer to the instruction
4. Your output should be provided in a language that can be associated persona
5. For languages that need diacritization, ensure that the returned prompt contains the right diacritics
6. Provide the prompt JSON format, each with fields: "prompt", "language"
"""
