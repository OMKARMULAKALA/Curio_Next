# Problem Formulation

Audio Question Answering takes two inputs:

1. An audio sample.
2. A natural-language question about that audio.

The output is an answer grounded in acoustic evidence. In this PoC, the answer is supervised by the DCASE metadata `answer` field and may correspond to one of the provided multiple-choice options.

Supported question categories are taken from the discovered `question_type` field, including sound detection, sound counting, audio tagging, species, vocalization, acoustic understanding, frequency, duration, memory-style, and combined reasoning questions.

