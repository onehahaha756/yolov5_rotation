# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

# Install linux packages
RUN apt update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-dev

COPY . /work/
COPY ./runs/train/tzplane21/weights/best.pt /work/runs/train/tzplane21/weights/best.pt
COPY ./submit_infer_rsaicp_plane.sh /work/submit_infer_rsaicp_plane.sh
# Create working directory
RUN cd /work/ && \
    pip install -r requirements.txt && \
    python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html


WORKDIR /work/

CMD ["sh", "submit_infer_rsaicp_plane.sh"]



