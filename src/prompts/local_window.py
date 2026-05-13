LOCAL_WINDOW_PROMPT: str = """
    Given a video clip and its corresponding speech transcription, thinking step by step,
    based on the person face, hair, clothing, etc., determine whether the spoken utterance comes from a person visible in the current frames. 
    Visible persons in frames are highlighted with red bounding boxes.
    If yes, output the bounding box (bbox) of the speaking person in each frame; otherwise, output [0, 0, 0, 0].
    Note, your rationale should be concise and to the point, do not repeat. Use bullet points only. Only analysis given frames, do not hallucinate.
    Pay close attention to the actual number of frames. Do not hallucinate an infinite duration. Max frame count is 30.
    If the analysis enters a dead loop, terminate the reasoning and output the JSON result strictly following the required format.


    Output format:
    {
        "rationale": ...,
        "boxes": [
            {
                "frame_id": 0,
                "bbox": [x, y, w, h]
            },
            {
                "frame_id": 1,
                "bbox": [0, 0, 0, 0]  # no speaking person, or the speaking person is not visible in the frame.
            },
            ...,
            {
                "frame_id": N,
                "bbox": [x, y, w, h]
            },
        ]
    }
"""
