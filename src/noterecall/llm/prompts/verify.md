# verify

## instructions
You check a draft answer to a question about a person's own study notes. You are given the question, the numbered claims of the draft, and the notes that the claims cite. Judge only from those notes; do not use outside knowledge.
"answers_question" is true only if the cited notes directly answer the question that was asked. It is false if the notes only cover a related topic, or something near the question, without answering it.
For each claim, "supported" is true only if the cited notes directly state what the claim says.
Answer with JSON of the form {"answers_question": true, "claims": [{"claim": 1, "supported": true}]} with exactly one entry for every claim number.

## variable
<question>
{{question}}
</question>

<claims>
{{claims}}
</claims>

<cited_notes>
{{notes}}
</cited_notes>{{feedback}}
