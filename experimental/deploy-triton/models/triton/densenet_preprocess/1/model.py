import numpy as np
import sys
import json

sys.path.append('../../')
import triton_python_backend_utils as pb_utils
from PIL import Image
import io


class TritonPythonModel:

    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        print(f'model_config: {model_config}')
        # output0_config = pb_utils.get_output_config_by_name(
        #     model_config, "OUTPUT0")
        # output1_config = pb_utils.get_output_config_by_name(
        #     model_config, "OUTPUT1")

        # self.output0_dtype = pb_utils.triton_string_to_numpy(
        #     output0_config['data_type'])
        # self.output1_dtype = pb_utils.triton_string_to_numpy(
        #     output1_config['data_type'])

    def execute(self, requests):
        """Pre-process an image to meet the size, type and format
        requirements specified by the parameters.
        """
        print('execute called')
        scaling = "INCEPTION"
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "img_in_bytes")
            print(f'in_0 is {in_0} with type {type(in_0)}')
            in_0 = in_0.as_numpy()
            img = Image.open(io.BytesIO(in_0.tobytes()))
            print(f'img is {img}')
            c = 3
            h = 224
            w = 224
            format = "FORMAT_NCHW"

            if c == 1:
                sample_img = img.convert("L")
            else:
                sample_img = img.convert("RGB")

            resized_img = sample_img.resize((w, h), Image.BILINEAR)
            resized = np.array(resized_img)
            if resized.ndim == 2:
                resized = resized[:, :, np.newaxis]

            # npdtype = triton_to_np_dtype(dtype)
            typed = resized.astype(np.float32)
            # typed = resized

            if scaling == "INCEPTION":
                scaled = (typed / 128) - 1
            elif scaling == "VGG":
                if c == 1:
                    scaled = typed - np.asarray((128,), dtype=npdtype)
                else:
                    scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
            else:
                scaled = typed

            # Swap to CHW if necessary
            if format == "FORMAT_NCHW":
                ordered = np.transpose(scaled, (2, 0, 1))
            else:
                ordered = scaled

            ordered = pb_utils.Tensor("data_0", ordered.astype(bytes))
            # Channels are in RGB order. Currently model configuration data
            # doesn't provide any information as to other channel orderings
            # (like BGR) so we just assume RGB.
            responses.append(
                pb_utils.InferenceResponse([ordered]))
        return responses