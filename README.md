# crocodile-detectron2-captum
Trained a Faster-RCNN for detecting crocodiles using the detectron2 library, then applying captum library to generate explainability measures.

After cloning:

`sudo docker build -t captum -f Dockerfile .`

 `bash run_docker_gpu.bash`
 
 `python3 main.py`