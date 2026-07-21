import hashlib
import json
import textwrap

import numpy as np

from dna.requests.api import call_api

_grounding_cache: dict[tuple, list] = {}


def _video_fingerprint(video: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(np.asarray(video.shape).tobytes())
    h.update(str(video.dtype).encode())
    h.update(video[0].tobytes())
    h.update(video[len(video) // 2].tobytes())
    h.update(video[-1].tobytes())
    return h.hexdigest()


def _stack_frames_into_image(frames: np.ndarray, max_cols: int = 4) -> np.ndarray:
    """Tile (N,H,W,C) frames into one RGB image, row-major grid."""
    n, h, w, c = frames.shape
    if n == 0:
        raise ValueError("empty frames")
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    pad_n = rows * cols - n
    if pad_n:
        pad = np.zeros((pad_n, h, w, c), dtype=frames.dtype)
        frames = np.concatenate([frames, pad], axis=0)
    strips = []
    for r in range(rows):
        row_imgs = [frames[r * cols + c] for c in range(cols)]
        strips.append(np.concatenate(row_imgs, axis=1))
    return np.concatenate(strips, axis=0)


def get_cached_objects(video: np.ndarray, task: str) -> list | None:
    """Return the grounded object list cached for ``(task, video)``, or None.

    The grounding step in ``get_description_from_video_grounded`` caches the
    detected objects keyed by task + video fingerprint. This accessor lets
    callers (e.g. terminal logging) read them back without re-running grounding.
    """
    return _grounding_cache.get((task, _video_fingerprint(video)))


def _parse_descriptions_payload(parsed: object) -> list[dict]:
    if not isinstance(parsed, list):
        parsed = [parsed]
    descriptions = []
    for p in parsed:
        if not isinstance(p, dict):
            continue
        if isinstance(p.get("descriptions"), list):
            descriptions.extend(p["descriptions"])
        elif "description" in p:
            descriptions.append(p["description"])
    return [{"description": d} for d in descriptions]


async def get_description_from_video_grounded(
    video: np.ndarray,
    task: str,
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "LOW",
) -> list[dict]:
    cache_key = (task, _video_fingerprint(video))

    if cache_key in _grounding_cache:
        objects = _grounding_cache[cache_key]
    else:
        prompt = textwrap.dedent(
            """\
            <role>
            You are an expert in object recognition.
            </role>
            
            <task>
            1. all_objects: list objects that are in the video.
            2. are some objects relevant to the following task? Task: "{task}" If yes, list them in task_objects. If no, return an empty list for task_objects.
            </task>

            <output_format>
            Respond with a list of objects in the following JSON format:
            [
                {
                    "all_objects": [str, str, ...],
                    "task_objects": [str, str, ...],
                }
            ]
            </output_format>
        """
        )
        prompt = prompt.replace("{task}", task)
        res = await call_api(
            prompt,
            video_input=video,
            thinking_level=thinking_level,
            model_id=model_id,
            json_output=True,
        )
        res_json = json.loads(res.text)[0]
        objects = res_json["all_objects"]
        _grounding_cache[cache_key] = objects

    prompt = textwrap.dedent(
        """\
        <role>
        You are an expert in scene understanding. Provide highly detailed descriptions including the robot and state of the following objects {objects} at each timestep. Describe robot motion and distance between robot and objects.
        Without looking at the rest of the video, describe the scene in the first and last frame to ground the descriptions.
        </role>

        <constraints>
        - The video may be captured from either a third-person or wrist-mounted (robot POV) camera.
        - Do not make any judgement about what the robot is trying to accomplish. The robot is imperfect and might do things that don't make any sense to you.
        - Describe the scene as objectively as possible.
        </constraints>

        <output_format>
        Respond with exactly {N} descriptions, where {N} is the number of video frames. There is only one string per frame. in the following JSON format:
        [
            {
                "description_first_frame": str,
                "description_last_frame": str,
                "descriptions": [str, str, ...],
            }
        ]
        </output_format>
    """
    )
    prompt = prompt.replace("{task}", task)
    prompt = prompt.replace("{objects}", json.dumps(objects))
    prompt = prompt.replace("{N}", str(len(video)))

    res = await call_api(
        prompt,
        video_input=video,
        thinking_level=thinking_level,
        model_id=model_id,
        json_output=True,
    )

    return _parse_descriptions_payload(json.loads(res.text))


async def get_description_from_video(
    video: np.ndarray,
    task: str,
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "LOW",
) -> list[dict]:
    prompt = textwrap.dedent(
        """\
        <role>
        You are an expert in scene understanding. Provide highly detailed descriptions including the robot and state of all objects in the video at each timestep.
        Without looking at the rest of the video, describe the scene in the first and last frame to ground the descriptions.
        </role>

        <constraints>
        - The video may be captured from either a third-person or wrist-mounted (robot POV) camera.
        - Do not make any judgement about what the robot is trying to accomplish. The robot is imperfect and might do things that don't make any sense to you.
        - Describe the scene as objectively as possible.
        </constraints>

        <output_format>
        Respond with exactly {N} descriptions, where {N} is the number of video frames. There is only one string per frame. in the following JSON format:
        [
            {
                "description_first_frame": str,
                "description_last_frame": str,
                "descriptions": [str, str, ...],
            }
        ]
        </output_format>
    """
    )
    prompt = prompt.replace("{task}", task)
    prompt = prompt.replace("{N}", str(len(video)))

    res = await call_api(
        prompt,
        video_input=video,
        thinking_level=thinking_level,
        model_id=model_id,
        json_output=True,
    )

    return _parse_descriptions_payload(json.loads(res.text))


async def get_description_from_all_frames(
    frames: np.ndarray,
    task: str,
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "LOW",
) -> list[dict]:
    """One model call with an ordered list of per-timestep RGB images."""
    n = len(frames)
    prompt = textwrap.dedent(
        """\
        <role>
        You are an expert in scene understanding. You see an ordered list of RGB images, one per timestep (robot camera).
        Provide highly detailed descriptions including the robot and state of all objects at each timestep.
        Describe the first and last image only to ground temporal extent, then describe each timestep.
        </role>

        <constraints>
        - Third-person or wrist-mounted camera; describe objectively.
        - Task context (for focus only): {task}
        - Do not assume the robot succeeds; describe only what is visible.
        </constraints>

        <output_format>
        Respond with exactly {N} descriptions (one string per frame, in order) in JSON:
        [
            {
                "description_first_frame": str,
                "description_last_frame": str,
                "descriptions": [str, str, ...],
            }
        ]
        </output_format>
    """
    )
    prompt = prompt.replace("{task}", task)
    prompt = prompt.replace("{N}", str(n))
    img_list = [frames[i] for i in range(n)]
    res = await call_api(
        prompt,
        img_input=img_list,
        thinking_level=thinking_level,
        model_id=model_id,
        json_output=True,
    )
    return _parse_descriptions_payload(json.loads(res.text))


async def get_description_from_all_frames_grounded(
    frames: np.ndarray,
    task: str,
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "LOW",
) -> list[dict]:
    fp = _video_fingerprint(frames)
    cache_key = ("all_frames", task, fp)
    if cache_key in _grounding_cache:
        objects = _grounding_cache[cache_key]
    else:
        prompt_g = textwrap.dedent(
            """\
            <role>You are an expert in object recognition.</role>
            <task>
            Given the ordered RGB images (one per timestep), list:
            1. all_objects visible across the sequence
            2. task_objects relevant to: "{task}"
            Each object should have its own unique str descriptor!
            </task>
            <output_format>
            [{"all_objects": [str,...], "task_objects": [str,...]}]
            </output_format>
        """
        )
        prompt_g = prompt_g.replace("{task}", task)
        img_list = [frames[i] for i in range(len(frames))]
        res = await call_api(
            prompt_g,
            img_input=img_list,
            thinking_level=thinking_level,
            model_id=model_id,
            json_output=True,
        )
        res_json = json.loads(res.text)[0]
        objects = res_json["all_objects"]
        _grounding_cache[cache_key] = objects

    n = len(frames)
    prompt = textwrap.dedent(
        """\
        <role>
        Expert scene understanding. Describe the robot and the state of these objects at each timestep: {objects}
        Use only the ordered RGB images as evidence. Ground using the first and last image.
        </role>
        <output_format>
        Exactly {N} strings in "descriptions" (one per frame, in order). JSON:
        [{{"description_first_frame": str, "description_last_frame": str, "descriptions": [str,...]}}]
        </output_format>
    """
    )
    prompt = prompt.replace("{objects}", json.dumps(objects))
    prompt = prompt.replace("{N}", str(n))
    img_list = [frames[i] for i in range(n)]
    res = await call_api(
        prompt,
        img_input=img_list,
        thinking_level=thinking_level,
        model_id=model_id,
        json_output=True,
    )
    return _parse_descriptions_payload(json.loads(res.text))


async def get_description_from_all_frames_ungrounded(
    frames: np.ndarray,
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "LOW",
) -> list[dict]:
    """Per-frame descriptions without task text (pure visual narration)."""
    n = len(frames)
    prompt = textwrap.dedent(
        f"""\
        <role>
        Describe each RGB image in an ordered robot-camera sequence. One concise but detailed string per timestep.
        </role>
        <output_format>
        Exactly {n} descriptions in JSON:
        [
            {{
                "description_first_frame": str,
                "description_last_frame": str,
                "descriptions": [str, str, ...],
            }}
        ]
        </output_format>
    """
    )
    img_list = [frames[i] for i in range(n)]
    res = await call_api(
        prompt,
        img_input=img_list,
        thinking_level=thinking_level,
        model_id=model_id,
        json_output=True,
    )
    return _parse_descriptions_payload(json.loads(res.text))


async def get_description_from_stacked_frames(
    frames: np.ndarray,
    task: str,
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "LOW",
    max_cols: int = 4,
) -> list[dict]:
    """Single tiled image; model returns one description per original frame."""
    n = len(frames)
    grid = _stack_frames_into_image(frames, max_cols=max_cols)
    prompt = textwrap.dedent(
        f"""\
        The image is a row-major grid of {n} consecutive robot-camera frames ({max_cols} per row, black padding if needed).
        Task context (for focus): {task}
        Describe each cell's scene in temporal order (frame 0 .. frame {n - 1}). JSON:
        [
            {{
                "description_first_frame": str,
                "description_last_frame": str,
                "descriptions": [str × {n}],
            }}
        ]
    """
    )
    res = await call_api(
        prompt,
        img_input=grid,
        thinking_level=thinking_level,
        model_id=model_id,
        json_output=True,
    )
    return _parse_descriptions_payload(json.loads(res.text))


async def get_description_from_stacked_frames_grounded(
    frames: np.ndarray,
    task: str,
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "LOW",
    max_cols: int = 4,
) -> list[dict]:
    n = len(frames)
    grid = _stack_frames_into_image(frames, max_cols=max_cols)
    fp = hashlib.md5(grid.tobytes()).hexdigest()
    cache_key = ("stacked", task, fp)
    if cache_key in _grounding_cache:
        objects = _grounding_cache[cache_key]
    else:
        prompt_g = textwrap.dedent(
            f"""\
            In this tiled image, each cell is a consecutive robot-camera frame (row-major, {n} frames).
            List all_objects and task_objects for task: "{task}". JSON:
            [{{"all_objects": [str,...], "task_objects": [str,...]}}]
        """
        )
        res = await call_api(
            prompt_g,
            img_input=grid,
            thinking_level=thinking_level,
            model_id=model_id,
            json_output=True,
        )
        res_json = json.loads(res.text)[0]
        objects = res_json["all_objects"]
        _grounding_cache[cache_key] = objects

    prompt = textwrap.dedent(
        f"""\
        Same tiled image ({n} frames in row-major order). Objects of interest: {json.dumps(objects)}
        For each timestep (cell 0..{n - 1}), describe the robot and those objects. JSON:
        [{{"description_first_frame": str, "description_last_frame": str, "descriptions": [str × {n}]}}]
    """
    )
    res = await call_api(
        prompt,
        img_input=grid,
        thinking_level=thinking_level,
        model_id=model_id,
        json_output=True,
    )
    return _parse_descriptions_payload(json.loads(res.text))


async def get_description_from_single_frames(
    frames: np.ndarray,
    task: str,
    model_id: str = "gemini-3-flash-preview",
    thinking_level: str = "LOW",
) -> list[dict]:
    """One API call per frame (isolated visual context)."""
    n = len(frames)
    out: list[dict] = []

    async def one_frame(i: int) -> str:
        prompt = textwrap.dedent(
            f"""\
            Single robot-camera frame (timestep {i} of {n}). Task context: {task}
            Describe the scene objectively in one detailed paragraph. JSON:
            [{{"description": str}}]
        """
        )
        res = await call_api(
            prompt,
            img_input=frames[i],
            thinking_level=thinking_level,
            model_id=model_id,
            json_output=True,
        )
        data = json.loads(res.text)
        if isinstance(data, list):
            data = data[0]
        return data["description"]

    for i in range(n):
        desc = await one_frame(i)
        out.append({"description": desc})
    return out
