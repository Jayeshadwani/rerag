# clean_markdown

## instructions
You correct one markdown cell from a person's study notes. Fix spelling mistakes and broken heading markup (for example a missing space after the # signs). Change nothing else.
Do not rewrite or reorder sentences. Do not add or remove content. Keep every inline code span, identifier, number, link and fenced code block exactly as written, character for character.
Answer with JSON of the form {"text": "<the corrected cell>"}.

## variable
<cell>
{{cell}}
</cell>
