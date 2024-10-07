FROM osaiai/dokai:23.05-vpf

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY video.mp4 /workdir/data/soccernet/spotting-ball-2024/video.mp4
#ENV OPENCV_FFMPEG_CAPTURE_OPTIONS "video_codec;hevc"
