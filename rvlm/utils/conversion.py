import numpy as np

def image_to_gemini_frame(paths: list[dict], H: int = 256, W: int = 256, dtype: np.dtype = float):
    path_dict = []
    for k,v in paths.items():
        path = np.array(v)

        # convert to gemini coordinates [0, H], [0, W] -> [0, 1000]
        path = path / np.array([H, W]) * 1000
        path = path.astype(dtype)
        if v.ndim == 2:
            # x, y -> y, x
            path_dict.append({"label": k, "points": path[:, ::-1]})
        else:
            path_dict.append({"label": k, "point": path[::-1]})

    return path_dict
    

def gemini_to_image_frame(paths: list[dict], H: int = 256, W: int = 256, dtype: np.dtype = float):
    path_dict = {}
    for p in paths:

        point = p.get("point", None)
        points = p.get("points", None)

        if points is not None:
            path = np.array(points) 
            # y, x -> x, y
            path = path[:, ::-1]
        elif point is not None:
            path = np.array(point)
            path = path[::-1]
        else:
            raise ValueError(f"No point or points found for {p['label']}")

        # convert to pixel coordinates [0, 1000] -> [0, H], [0, W]
        path = path / 1000.0 * np.array([H, W])
        path = path.astype(dtype)                
        path_dict[p["label"]] = path

    return path_dict
