GLOBAL_WINDOW_PROMPT: str = """
Task:
Analyze the persons appearing in the given frames and determine whether they belong to the same individual across frames.

Instructions:
- Group frames that contain the same person. Frame id starts from 0.
- Assign a unique identifier (person_0, person_1, etc.) to each individual.
- One frame may belong to multiple individuals.

Note, your rationale should be concise and to the point, do not repeat. Use bullet points only. Only analysis given frames, do not hallucinate.
Do not include frame IDs in the rationale; they should appear only in the 'persons' field.
Pay close attention to the actual number of frames. Do not hallucinate an infinite duration. 
If the analysis enters a dead loop, terminate the reasoning and output the JSON result strictly following the required format.

Output format (JSON only):
{
  "rationale": ...,
  "persons": {
    "person_0": [0, 2],  # frame 2 中有person_0, 和person_1. 
    "person_1": [1, 2, ...],
    ...
  }
}
"""
