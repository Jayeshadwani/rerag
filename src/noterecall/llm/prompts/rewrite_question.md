# rewrite_question

## instructions
You rewrite a follow-up question into a question that can be understood on its own. You are given the earlier turns of a conversation (each a question and the answer shown), then the follow-up.
Resolve references such as "it", "that", "the second one" or "why?" using the earlier turns, and keep the meaning and the wording of the follow-up as far as possible. If the follow-up is already understandable on its own, return it unchanged.
Do not answer the question. Do not add facts, names or topics that are not in the earlier turns or the follow-up. Write one question.
Answer with JSON of the form {"question": "<the standalone question>"}.

## variable
<conversation>
{{history}}
</conversation>

<follow_up>
{{question}}
</follow_up>
