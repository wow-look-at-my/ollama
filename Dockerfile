# vim: filetype=dockerfile

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS build

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build ccache ca-certificates curl gcc g++
ENV CMAKE_GENERATOR=Ninja

WORKDIR /build
COPY CMakeLists.txt CMakePresets.json ./
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml

RUN cmake --preset CPU \
    && cmake --build --preset CPU -j$(nproc) \
    && cmake --install build --component CPU --strip

RUN cmake --preset 'CUDA 13' \
    && cmake --build --preset 'CUDA 13' -j$(nproc) \
    && cmake --install build --component CUDA --strip

WORKDIR /build/ollama
COPY go.mod go.sum ./
RUN GO_VERSION=$(awk '/^go / { print $2 }' go.mod) \
    && curl -fsSL "https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz" \
       | tar xz -C /usr/local
ENV PATH=/usr/local/go/bin:$PATH

RUN go mod download
COPY . .
ENV CGO_ENABLED=1
RUN go build -trimpath -buildmode=pie -ldflags='-w -s' -o /bin/ollama .

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl

COPY --from=build /bin/ollama /usr/bin/ollama
COPY --from=build /build/dist/lib/ollama /usr/lib/ollama

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all
ENV OLLAMA_HOST=0.0.0.0:11434
EXPOSE 11434
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
