import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import drawline
import ffmpeg
from fastapi import UploadFile


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


def split_video(path: Path, tmpdirname) -> Path:
    in_file = ffmpeg.input(path)
    # convert video into jpg frames with a 1 second interval
    out_file = in_file.output(tmpdirname + "/%03d.jpg", vf="fps=1")
    out_file.run()
    # in_file.output(tmpdirname + '/image-%05d.jpg').run()
    # get list of frames
    return Path(tmpdirname)


def chunks(generator, chunk_size):
    """Yield successive chunks from a generator"""
    chunk = []

    for item in generator:
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = [item]
        else:
            chunk.append(item)

    if chunk:
        yield chunk


def draw_predictions(chunk, predictions):
    for filename, image, coords in zip(chunk, predictions.ims, predictions.pred):
        # Filter coords
        coords = coords[coords[:, -1] == 2]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR, image)
        # Draw bounding boxes
        # image = drawline.draw_rect(image, coords[:, :4].numpy().astype(np.uint8))
        image = drawline.draw_process.draw_text(
            image, f"{len(coords)} cars detected", (0, 0), (0, 200)
        )
        cv2.imwrite(filename, image)


def join_frames_into_video(image_folder):
    # convert jpg images from image_folder to mp4 video
    # 1 second per frame
    ffmpeg.input(image_folder + "/%03d.jpg", r=1).output(
        "output.mp4", vcodec="libx264", pix_fmt="yuv420p"
    ).run(overwrite_output=True)
    # upload video to s3
    return Path(image_folder + "/output.mp4")
