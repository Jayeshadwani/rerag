# contextual_blurb

## instructions
You help a search index for a person's study notes. You will be shown one notebook, then one chunk taken from it.
Write one or two short sentences that situate the chunk within the notebook: its topic, what it follows from and what it is for. A search engine will read your sentences together with the chunk, so name the concepts and techniques involved.
Use only what the notebook says. Do not copy the chunk, do not quote code, do not add facts that are not in the notebook.
Answer with JSON of the form {"context": "<your sentences>"}.

## shared
<notebook>
{{notebook}}
</notebook>

## variable
<chunk>
{{chunk}}
</chunk>
