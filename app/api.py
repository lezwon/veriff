import pathlib
import time
from tempfile import TemporaryDirectory

from fastapi import FastAPI, UploadFile, Depends

from app import utils
from app.logger import Logger
from app.model import YoloModel

server = FastAPI()
logger = Logger.getLogger(__name__, True)


async def get_classifier():
    # TODO: do this at startup
    return YoloModel()


@server.post("/infer/")
async def create_upload_file(
    file: UploadFile, model: YoloModel = Depends(get_classifier)
):
    logger.info("Request received")
    url = ""
    success = False
    errors = []
    start_time = time.time()

    try:
        # save file to /tmp
        path = utils.save_upload_file_tmp(file)
        # split video into frames
        with TemporaryDirectory() as tmpdirname:
            print("created temporary directory", tmpdirname)
            # convert entire video into jpg frames
            utils.split_video(path, tmpdirname)
            # get list of frames
            frames = pathlib.Path(tmpdirname).glob("*.jpg")
            chunks = utils.chunks(frames, 10)
            # draw predictions on each frame
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk..{i}")
                predictions = model.infer(chunk)
                # join frames back together with count
                utils.draw_predictions(chunk, predictions)
            # join frames into video
            output_path = utils.join_frames_into_video(tmpdirname)
            # return output
            url = output_path.as_uri()
            success = True

        success = True
    except Exception as e:
        print(e)
        errors.append(str(e))

    time_taken = time.time() - start_time

    return {
        "output_url": url,
        "success": success,
        "time_taken": time_taken,
        "errors": [],
    }
