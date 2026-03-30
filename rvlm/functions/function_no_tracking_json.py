import io
import os
import time
import json
import textwrap

from google.genai import types

from rvlm.functions.function_helpers import _key_pool

import imageio

import asyncio
import numpy as np

global _ctr
_ctr = 0

async def call_gemini(prompt, video_input=None, img_input=None, thinking_level="MEDIUM", model_id: str = "gemini-3-flash-preview", json_output: bool = False, response_schema=None):
    """
    Calls Gemini with a video, an image, and a text prompt.
    
    Args:
        video_input: Path to .mp4 file OR base64/bytes data
        img_input: Path to .jpg/.png file OR base64/bytes data
        prompt: String text
        model_id: Gemini model ID
    """

    parts = []

    # (optional) process video
    if video_input is not None:
        if isinstance(video_input, np.ndarray):
            buf = io.BytesIO()
            imageio.mimwrite(buf, video_input, format="mp4", fps=1)
            video_data = buf.getvalue()
        elif isinstance(video_input, str) and os.path.exists(video_input):
            with open(video_input, "rb") as f:
                video_data = f.read()
        else:
            video_data = video_input
        parts.append(types.Part.from_bytes(mime_type="video/mp4", data=video_data))

    # (optional) process image
    if img_input is not None:
        if isinstance(img_input, np.ndarray):
            buf = io.BytesIO()
            imageio.imwrite(buf, img_input, format="JPEG")
            img_data = buf.getvalue()
        elif isinstance(img_input, str) and os.path.exists(img_input):
            with open(img_input, "rb") as f:
                img_data = f.read()
        else:
            img_data = img_input
        parts.append(types.Part.from_bytes(mime_type="image/jpeg", data=img_data))

    # process prompt
    parts.append(types.Part.from_text(text=prompt))

    # configuration
    if model_id == "gemini-3-flash-preview":
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level, include_thoughts=True),
            media_resolution="MEDIA_RESOLUTION_HIGH",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            **({"response_mime_type": "application/json", 
                **({"response_schema": response_schema} if response_schema else {})} 
               if json_output else {}),
        )
    else:
        raise ValueError(f"Unsupported model ID: {model_id}")
    response = await _key_pool.generate_content(
        model=model_id,
        contents=[types.Content(role="user", parts=parts)],
        config=generate_content_config,
    )
    return response

# 1. image -> labels + points
async def get_labels_and_points(image: np.ndarray, model_id: str = "gemini-3-flash-preview") -> str:
    prompt = textwrap.dedent(f"""\
        1. Describe the scene in great detail. Keep it concise (250 words).
        
        2. List the objects AND (multiple) descriptors (e.g., top, bottom, handle, center, left, right, blue, green, transparent, ...).
        
        3. Ground your description in the image by pointing to the object descriptors. The points are in [y, x] format normalized to 0-1000.
        
        Respond in the following JSON format:
        [
            {{
                "description": <description>,
                "objects": {{
                    <object_descriptor>: [y,x],
                    <object_descriptor>: [y,x],
                    ...
                }}
            }}
        ]

    """)

    res = await call_gemini(prompt, img_input=image, thinking_level="LOW", model_id=model_id, json_output=True)
    return res.text

# 2. labels + image -> scene description + points]
async def get_descriptions_and_points(video: np.ndarray, labels: list[str], model_id: str = "gemini-3-flash-preview") -> list[str]:
    prompt = textwrap.dedent(f"""\
        1. Describe the scene in great detail. Keep it concise (250 words).
        
        2. Given this list of object descriptors: {labels}, ground your description in the image by pointing to the object descriptors. The points are in [y, x] format normalized to 0-1000. If the object is not visible, set the corresponding point to [-1, -1]. Only use the predefined object descriptors in the list!
        
        Respond in the following JSON format:
        [
            {{
                "description": <description>,
                "objects": {{
                    <object_descriptor>: [y,x],
                    <object_descriptor>: [y,x],
                    ...
                }}
            }}
        ]
    """)
    prompt = prompt.replace("{labels}", str(labels))

    res = await asyncio.gather(*[call_gemini(prompt, img_input=img, thinking_level="LOW", model_id=model_id, json_output=True) for img in video])
    
    return [r.text for r in res]

# 2. labels + task -> subtasks + target points + constraints
async def get_subtasks(image: np.ndarray, task: str, labels: list[str], model_id: str = "gemini-3-flash-preview") -> str:
    prompt = textwrap.dedent("""\

        Task: "{task}"
        List of object descriptors: {labels}

        1. Provide granular (e.g., pick-and-place are two separate sub-tasks) list of relevant sub-tasks the robot must complete to solve the following task.
        2. Given the list of object descriptors, define a minimize distance constraint between two object descriptors that is required to complete the subtask.
        3. Ground the subtasks in the image by providing the target points of the two objects and descriptors.
        4. Don't include free-space motions like "moving" or "reaching".

        Your answer should conclude with the following JSON format:
        ```json
        [{"sub_task_1": str, "constraint": [<object_descriptor_0>, <object_descriptor_1>], "points": [[y_0, x_0], [x_1, y_1]]}, ...]
        ```

        Respond in the following JSON format:
        [
            {{
                "sub_task_1": str,
                "constraint": [<object_descriptor_0>, <object_descriptor_1>],
                "points": [[y_0, x_0], [x_1, y_1]]
            }},
            {{
                "sub_task_2": str,
                "constraint": [<object_descriptor_0>, <object_descriptor_1>],
                "points": [[y_0, x_0], [x_1, y_1]]
            }}
             ...
        ]

        If the task cannot completed within the scene due to missing objects, return:
        []

        Only use the predefined object descriptors in the list!
    """)
    prompt = prompt.replace("{task}", task)
    prompt = prompt.replace("{labels}", str(labels))

    res = await call_gemini(prompt, img_input=image, thinking_level="MEDIUM", model_id=model_id, json_output=True)
    return res.text

# 3. subtasks + scene description(s) -> verification
async def get_verification(subtasks: list[str], scene_descriptions: list[str], model_id: str = "gemini-3-flash-preview") -> str:
    prompt = textwrap.dedent("""\

        Given the list of sub-tasks:
        "{subtasks}"

        and temporal scene description of what happened:
        "{scene_descriptions}"

        provide success/failure and exact timestamps when the sub-task was successful (or -1 for failures) for each of the sub-tasks.

        Some descriptions might be wrong, average or smooth out incorrect descriptions over multiple timesteps!

        Respond in the following JSON format:
        [
            {{
                "sub_task_1": str,
                "success": bool,
                "time": int
            }},
            {{
                "sub_task_2": str,
                "success": bool,
                "time": int
            }}
             ...
        ]
    """)
    prompt = prompt.replace("{subtasks}", str(subtasks))
    prompt = prompt.replace("{scene_descriptions}", str(scene_descriptions))

    res = await call_gemini(prompt, thinking_level="MEDIUM", model_id=model_id, json_output=True)
    return res.text

async def compute_rewards(language_instruction: str, video: np.ndarray, model_id: str = "gemini-3-flash-preview") -> tuple[list[float], list[float]]:
    
    text_history = [language_instruction]
    start_time = time.time()

    # 1. image -> labels + points
    labels_and_points = await get_labels_and_points(video[0], model_id=model_id)
    print(labels_and_points)
    # text -> list[str]
    labels_list = [k for k in json.loads(labels_and_points)[0]["objects"].keys()]
    # print("1. LABELS\n", labels_list)

    subtasks = await get_subtasks(video[0], language_instruction, labels_list, model_id=model_id)
    text_history.append(subtasks)
    subtasks = json.loads(subtasks)
    # text -> list[dict]
    subtasks_list = {k: v for r in subtasks for k, v in r.items() if "sub_task" in k}
    # print("2. SUBTASKS\n", subtasks)

    # early stopping
    progress = np.zeros(len(video))
    subtask_progress = np.zeros(len(video))
    if not subtasks_list:
        return progress, subtask_progress

    # NOTE: for matrix computation, we save calls by prematurely exiting once we know there are no subtasks
    descriptions_and_points = await get_descriptions_and_points(video, labels_list, model_id=model_id)
    text_history.append(descriptions_and_points)
    # list[str] -> dict{int: str}
    scene_descriptions = {t: json.loads(v)[0] for t, v in enumerate(descriptions_and_points)}
    
    # print("2. DESCRIPTIONS\n", scene_descriptions)
    
    if subtasks:
        # 3. subtasks + scene description(s) -> verification
        verification = await get_verification(subtasks_list, scene_descriptions, model_id=model_id)
        text_history.append(verification)
        # text -> list[dict]
        verification_list = json.loads(verification)
    else:
        verification_list = []
    # print("3. VERIFICATION\n", verification_list)

    # early stopping
    progress = np.zeros(len(video))
    subtask_progress = np.zeros(len(video))
    if not subtasks_list:
        return progress, subtask_progress

    # misc
    tracked_points = {}
    for k in labels_list:
        tracked_points[k] = []
        for sd in scene_descriptions.values():
            # set point to [-1, -1] if object is not in scene description
            if k in sd["objects"]:
                point = sd["objects"][k]
            else:
                point = [-1, -1]
            tracked_points[k].append(point)
            
        tracked_points[k] = np.array(tracked_points[k])

    # 4. compute tracking progress

    # TODO: sort verfication by order of sub-tasks completion
    # verification_list = sorted(verification_list, key=lambda x: x["time"] if x["time"] != -1 else float("inf"))
    
    prev_terminal = 0
    subtask_distances = {}
    for i, (sub, ver) in enumerate(zip(subtasks, verification_list)):

        terminal = ver["time"]

        # if terminal == -1:
        #     continue

        # TODO: instead of skipping, set terminal to end of video to apply tracking for first failed sub-task
        if terminal == -1:
            terminal = len(video)

        sub_constraints = sub["constraint"]
        sub_target_points = sub["points"]

        source_key = sub_constraints[0]
        target_point = np.array(sub_target_points[1])

        # track distance of constraint to target_point - H \in [0,1] and W \in [0,1] - distance should be [0,1]
        distances = np.linalg.norm(tracked_points[source_key] / 1000 - target_point / 1000, axis=1)

        subtask_distances[list(sub.keys())[0]] = distances

        # negative distance = progress/reward
        reward = - distances

        # normalize by starting distance because we want to evaluate progress
        progress[prev_terminal:terminal] = reward[prev_terminal:terminal] - reward[prev_terminal]

        prev_terminal = terminal

    progress = np.cumsum(progress) / len(video)
    text_history.append(progress)

    # 4. compute subtask progress
    for v in verification_list:
        if v["time"] > 1:
            subtask_progress[v["time"]] = 1
    subtask_progress = np.cumsum(subtask_progress) / len(subtasks)
    text_history.append(subtask_progress)

    print(f"Full computation took: {time.time() - start_time} seconds")

    # logging
    if True:
        global _ctr
        root_dir = "/gpfs/home/memmelma/projects/rvlm/tmp"
        os.makedirs(root_dir, exist_ok=True)
        imageio.mimwrite(f"{root_dir}/video_{_ctr}.mp4", video, fps=1)
        with open(f"{root_dir}/text_history_{_ctr}.txt", "w") as f:
            f.write(str(text_history))
        _ctr += 1

    return progress, subtask_progress