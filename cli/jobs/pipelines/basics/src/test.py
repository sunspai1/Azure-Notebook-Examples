import os
import mlflow

os.system('echo "hello world" > helloworld.txt')
print(os.environ)

mlflow.log_param("hello", "world")
mlflow.log_metric("hello_factor", 0.9)
mlflow.log_artifact("helloworld.txt")
