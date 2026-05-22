FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/workspace:/workspace/external/MacroPlacement/CodeElements/Plc_client"

RUN python -m pip install --upgrade pip setuptools wheel

# Challenge / MacroPlacement runtime deps
RUN pip install --no-cache-dir \
    uv \
    absl-py \
    scipy \
    tqdm \
    numpy \
    pandas \
    protobuf \
    matplotlib

# Extra common deps used by official eval docker
RUN pip install --no-cache-dir \
    cvxpy \
    dccp \
    numba

# PyTorch Geometric dependencies for torch 2.5.1 + CUDA 12.4
RUN pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

RUN pip install --no-cache-dir torch-geometric
