# answer

## instructions
You answer a question about a person's own study notes. You are given numbered notes (C1, C2, ...) taken from them, then the question.
Use only what the notes say. Write the answer as short claims. Every claim must cite the IDs of the notes that support it, for example ["C1", "C3"]. Cite only IDs that were given.
If the notes do not contain the answer, set "insufficient" to true and give no claims. Do not guess and do not use outside knowledge.
Answer with JSON of the form {"insufficient": false, "claims": [{"text": "<one claim>", "citations": ["C1"]}]}.

## variable
<notes>
{{notes}}
</notes>

<question>
{{question}}
</question>{{feedback}}
