from roboflow import Roboflow


rf = Roboflow(api_key="X5MUXd8lEdTqFTtOgUvj")
project = rf.workspace("autismo").project("autism-jlyfw")
version = project.version(4)
dataset = version.download("yolov8")