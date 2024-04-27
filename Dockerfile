ARG UBUNTU_VERSION=22.04
FROM ubuntu:$UBUNTU_VERSION as build

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    libssl-dev \
    zlib1g-dev

WORKDIR /sd.cpp
COPY . .

RUN mkdir build && cd build && cmake .. && cmake --build . --config Release

FROM ubuntu:$UBUNTU_VERSION as runtime
COPY --from=build /sd.cpp/build/bin/server /sd

ENTRYPOINT ["/sd"]