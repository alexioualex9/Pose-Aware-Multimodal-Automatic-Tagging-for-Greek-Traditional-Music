from pathlib import Path
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


def fetch_danced_video_ids(url: str) -> set[str]:
    """Download metadata and return the IDs of videos annotated as danced."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text), sep="\t")
    danced_ids = df.loc[df["is-danced"] == 1, "id"].astype(str)

    return set(danced_ids.tolist())


def list_video_files(directory: Path, extension: str = ".mp4") -> list[Path]:
    """Return sorted video files from a directory."""
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == extension
    )

def write_results(scene_labels, scene_list, output_dir: Path, video_name: str) -> Path:
    """Write frame ranges of detected dance scenes to a text file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"results_of_video_{video_name}.txt"

    with output_path.open("w", encoding="utf-8") as file:
        for label, (start_timecode, end_timecode) in zip(scene_labels, scene_list):
            if label == 1:
                start_frame = start_timecode.get_frames()
                end_frame = end_timecode.get_frames() - 1
                file.write(f"{start_frame} - {end_frame}\n")

    return output_path