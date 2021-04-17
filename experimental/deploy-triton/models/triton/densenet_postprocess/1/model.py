import numpy as np
import os
import sys
import json

sys.path.append('../../')
import triton_python_backend_utils as pb_utils
from PIL import Image
import io


class TritonPythonModel:

    def initialize(self, args):
        model_dir = os.path.abspath(os.path.join('/var', 'azureml-app', os.environ["AZUREML_MODEL_DIR"], 'triton'))
        label_path = os.path.join(
            model_dir, "densenet_onnx", "densenet_labels.txt"
        )
        print(f"DEBUG: model dir is {model_dir}")
        label_file = open(label_path, "r")
        labels = label_file.read().split("\n")
        self.label_dict = dict(enumerate(labels))

    def execute(self, requests):
        """Output the label of a given image
        """
        print('execute called')
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "fc6_1")
            print(f'in_0 is {in_0} with type {type(in_0)}')
            in_0 = in_0.as_numpy()
            output_array = in_0
            max_label = np.argmax(output_array)
            final_label = self.label_dict[max_label]

            ordered = np.array([final_label], dtype=bytes)
            ordered = pb_utils.Tensor("label", ordered)
            # Channels are in RGB order. Currently model configuration data
            # doesn't provide any information as to other channel orderings
            # (like BGR) so we just assume RGB.
            responses.append(
                pb_utils.InferenceResponse([ordered]))
        return responses