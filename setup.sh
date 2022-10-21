docker run -it \
--rm \
--name devtest \
--platform=linux/amd64 \
-p 8000:8000 \
--mount type=bind,source="$(pwd)"/src,target=/workspace \
webapp