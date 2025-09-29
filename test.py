import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")
print(client)



# raw_image = cv2.imread("./img2.jpg")
# preprocessed_image = detection_preprocessing(raw_image)
#
# detection_input = httpclient.InferInput("input_images:0", preprocessed_image.shape, datatype="FP32")
# detection_input.set_data_from_numpy(preprocessed_image, binary_data=True)