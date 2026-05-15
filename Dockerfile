# vim: filetype=dockerfile
#
# Default Dockerfile for ollama.
#
# Produces a CPU-only Go build of the ollama binary. Small enough to build
# on a standard CI runner in a few minutes and works as a `docker build .`
# target when pointing a builder at the repo.
#
# The full GPU-enabled production build (CUDA, ROCm, Vulkan, JetPack, MLX)
# lives in Dockerfile.release and is driven by scripts/build_docker.sh and
# the release workflow.

FROM golang:1.26-bookworm AS build
WORKDIR /go/src/github.com/ollama/ollama

COPY go.mod go.sum ./
RUN go mod download

COPY . .
ENV CGO_ENABLED=1
RUN --mount=type=cache,target=/root/.cache/go-build \
    go build -trimpath -buildmode=pie -ldflags='-w -s' -o /bin/ollama .

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /bin/ollama /usr/bin/ollama
ENV OLLAMA_HOST=0.0.0.0:11434
EXPOSE 11434
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
