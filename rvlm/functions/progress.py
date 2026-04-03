import asyncio
import json
import textwrap

import numpy as np

from rvlm.requests.api import call_api


async def get_progress_from_description_distributional(
    description: dict,
    task: str,
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "MEDIUM",
    k_requests: int = 5,
) -> list:
    prompt = textwrap.dedent(
        f"""\

        Scene descriptions:
        {{description}}

        1. Ground the following task in the scene descriptions and reason about what it means for the robot to have completed the task: "{{task}}"
        2. Define precisely what the environment would look like if the task were FULLY and COMPLETELY finished (not started, not halfway done — entirely done).
        3. Summarize the scene descriptions and alias over potentially noisy descriptions.
        4. For each timestep, reason about the % progress towards the fully completed state.
        5. If the task cannot be completed, all progress values are 0%.
        
        Progress is defined as % progress towards the fully completed state. Progress does not have to be monotonic, i.e., the robot can reverse progress.
        
        Respond with a list of progress values (ordered by frame) in the following JSON format:
        [
            {{
                "completion state": str,
                "progress": [int, int, ...]
            }}
        ]

        Before you respond, reflect whether the task is actually fully completed, otherwise adjust the progress values!
    """
    )
    prompt = prompt.replace("{task}", task)
    prompt = prompt.replace("{description}", json.dumps(description))

    res = await asyncio.gather(
        *[
            call_api(
                prompt,
                thinking_level=thinking_level,
                model_id=model_id,
                json_output=True,
                include_thoughts=False,
            )
            for _ in range(k_requests)
        ]
    )
    return res


async def get_progress_from_description(
    description: dict,
    task: str,
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "MEDIUM",
):
    prompt = textwrap.dedent(
        f"""\

        Task:
        "{{task}}"

        Scene descriptions:
        {{description}}

        1. The task text fixes which objects count (category, color, shape, and wording). Scene descriptions are the only evidence of what is in the environment. Do not "ground" the task by renaming scene objects to match the task.
        2. For each object the task requires, it may be tied only to scene mentions that refer to that same entity without contradiction. Do not substitute a different object because it could serve the same role. If attributes in the scene conflict with the task's description of that object, or no clear referent exists, treat that requirement as absent.
        3. You may align short paraphrases across timesteps only when they plainly denote the same physical object already present in the scene—not to map task nouns onto other objects.
        4. If any required object is absent by these rules, the task cannot be completed in this scene: use all 0% progress and explain in "completion state". Otherwise define precisely what the environment would look like if the task were FULLY and COMPLETELY finished (not started, not halfway—entirely done).
        5. For each timestep, reason about the % progress towards that fully completed state.

        
        Progress is defined as % progress towards the fully completed state. Progress does not have to be monotonic, i.e., the robot can reverse progress.
        
        Respond with a list of progress values (ordered by frame) in the following JSON format:
        [
            {{
                "completion state": str,
                "progress": [int, int, ...]
            }}
        ]

        Before you respond, reflect whether the task is actually fully completed, otherwise adjust the progress values!
    """
    )

    prompt = prompt.replace("{task}", task)
    prompt = prompt.replace("{description}", json.dumps(description))

    res = await call_api(
        prompt,
        thinking_level=thinking_level,
        model_id=model_id,
        json_output=True,
        include_thoughts=False,
    )
    return res


async def get_progress_from_description_rubric(
    description: dict,
    task: str,
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "MEDIUM",
):
    prompt = textwrap.dedent(
        f"""\

        Task:
        "{{task}}"

        Scene descriptions:
        {{description}}

        Score progress using an explicit rubric:
        1. List 3–5 concrete observable criteria for full task completion, grounded only in the scene descriptions (not inventing objects).
        2. For each timestep, score each criterion 0–100; then combine into one overall % progress per frame (be conservative if any critical criterion is near 0).
        3. Progress does not have to be monotonic.

        Respond with JSON (single-element list) in this format:
        [
            {{
                "rubric": [{{"criterion": str, "note": str}}, ...],
                "completion state": str,
                "progress": [int, int, ...]
            }}
        ]

        Before you respond, reflect whether the task is actually fully completed, otherwise adjust the progress values!
    """
    )
    prompt = prompt.replace("{task}", task)
    prompt = prompt.replace("{description}", json.dumps(description))
    return await call_api(
        prompt,
        thinking_level=thinking_level,
        model_id=model_id,
        json_output=True,
        include_thoughts=False,
    )


async def get_progress_from_video(
    video: np.ndarray,
    task: str,
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "LOW",
):
    prompt = textwrap.dedent(
        f"""\

        Task:
        "{{task}}"

        1. The task text fixes which objects count (category, color, shape, and wording). The video frames are the only evidence of what is in the environment. Do not "ground" the task by renaming visible objects to match the task.
        2. For each object the task requires, it may be tied only to objects visible in the video that refer to that same entity without contradiction. Do not substitute a different object because it could serve the same role. If the visual appearance of an object conflicts with the task's description of that object, or no clear referent is visible, treat that requirement as absent.
        3. You may track the same physical object across frames only when it is plainly the same entity already visible in the video—not to map task nouns onto other objects.
        4. If any required object is absent by these rules, the task cannot be completed in this scene: use all 0% progress and explain in "completion state". Otherwise define precisely what the environment would look like if the task were FULLY and COMPLETELY finished (not started, not halfway—entirely done).
        5. For each timestep, reason about the % progress towards that fully completed state.

        
        Progress is defined as % progress towards the fully completed state. Progress does not have to be monotonic, i.e., the robot can reverse progress.
        
        Respond with a list of progress values (ordered by frame) in the following JSON format:
        [
            {{
                "completion state": str,
                "progress": [int, int, ...]
            }}
        ]

        Before you respond, reflect whether the task is actually fully completed, otherwise adjust the progress values!
    """
    )

    prompt = prompt.replace("{task}", task)

    res = await call_api(
        prompt,
        video_input=video,
        thinking_level=thinking_level,
        model_id=model_id,
        json_output=True,
        include_thoughts=False,
    )
    return res


async def get_progress_from_all_frames(
    frames: np.ndarray,
    task: str,
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "LOW",
):
    """End-to-end progress from an ordered list of RGB frames (same prompt as video path)."""
    prompt = textwrap.dedent(
        f"""\

        Task:
        "{{task}}"

        You are given an ordered list of RGB images, one per timestep (earliest to latest).

        1. The task text fixes which objects count (category, color, shape, and wording). The images are the only evidence of what is in the environment. Do not "ground" the task by renaming visible objects to match the task.
        2. For each object the task requires, it may be tied only to objects visible in the images that refer to that same entity without contradiction. Do not substitute a different object because it could serve the same role. If the visual appearance of an object conflicts with the task's description of that object, or no clear referent is visible, treat that requirement as absent.
        3. You may track the same physical object across timesteps only when it is plainly the same entity already visible—not to map task nouns onto other objects.
        4. If any required object is absent by these rules, the task cannot be completed in this scene: use all 0% progress and explain in "completion state". Otherwise define precisely what the environment would look like if the task were FULLY and COMPLETELY finished (not started, not halfway—entirely done).
        5. For each timestep, reason about the % progress towards that fully completed state.

        Progress is defined as % progress towards the fully completed state. Progress does not have to be monotonic.

        Respond with a list of progress values (ordered by frame) in the following JSON format:
        [
            {{
                "completion state": str,
                "progress": [int, int, ...]
            }}
        ]

        Before you respond, reflect whether the task is actually fully completed, otherwise adjust the progress values!
    """
    )
    prompt = prompt.replace("{task}", task)
    img_list = [frames[i] for i in range(len(frames))]
    return await call_api(
        prompt,
        img_input=img_list,
        thinking_level=thinking_level,
        model_id=model_id,
        json_output=True,
        include_thoughts=False,
    )
