import os
from dotenv import load_dotenv
import google.generativeai as genai
import PIL.Image
import requests
import io
import re
import json

load_dotenv()

generation_config = {
  "temperature": 0.9,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 1024,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)


instruction = "You are a helpful assistant designed to output JSON object. You are helping a user generate a title and description for an image. You should always keep the title 120 characters or less and description 500 characters or less. And return the title and description as a JSON object: {'title': '...', 'description': '...'}."
prompt = "Generate Conclusion for this Observation. Return only the conclusion as a JSON object in string format: \"{'conclusion': '...'}\""


def generate_conclusion(text):

    response = model.generate_content(['Observation: \n'+text+'\n'+prompt])

    response_text = response.text
    print("response_text: ", response_text)

    json_string = re.search(r"\{.*\}", response_text, re.DOTALL).group()

    json_string = json_string.replace("'", '"')

    response_dict = json.loads(json_string)
    print("response_dict: ", response_dict)

    return response_dict

# Usage 
# text = """

# Scanning technique: T1 FSE-sagital, T2 FLAIR, T2 FSE-axial, T2 FSE-coronar.
# On a series of tomograms, the longitudinal fissure of the cerebrum is located centrally. Convexital grooves are not expanded, their number is not changed. The thickness of the cerebral cortex is preserved, the distribution of gray matter is not disturbed.
# The lateral ventricles are symmetrical, the width of the ventricles at the level of the foramen of Monro is 7 mm on the right, 9 mm on the left. The third ventricle is 4 mm wide. Sylvian aqueduct has not been changed. The fourth ventricle is tent-shaped and not dilated.
# The white matter of the brain has homogeneous isointense signal characteristics; the signal intensity in the periventricular areas is not changed. The basal ganglia are usually located, symmetrical, with clear, even contours, the dimensions are not changed, the MR signal is not changed. The corpus callosum is of normal shape and size. The brain stem is without features. The cerebellum is of normal shape, the signal characteristics of the white matter are not changed. The width of the cerebellar cortex is preserved. The craniovertebral junction is unchanged.
# The pituitary gland is of normal shape, height in the sagittal projection is 4 mm. The pituitary stalk is located centrally. The chiasma of the optic nerves is located usually, the contours are clear and even. Parasellar cisterns without areas of pathological intensity. The siphons of the internal carotid arteries are not changed. The cavernous sinuses of both carotid arteries are symmetrical, with clear, even contours.
# The shape of the orbital cones on both sides is unchanged. The eyeballs are spherical in shape and of normal size; the MR signal of the vitreous body is not changed. The diameter of the optic nerves was preserved. The perineural subarachnoid space of the orbits is moderately diffusely dilated. The extraocular muscles are of normal size, the structure is without pathological signals. Retrobulbar fatty tissue without pathological signals.
# Region of the cerebellopontine angle: the prevestocochlear nerve is clearly differentiated on both sides. Pneumatization of the mastoid processes of both temporal bones is not impaired.
# The paranasal sinuses are usually pneumatized.
# On angiograms of intracranial vessels performed in 3D TOF 3 sl mode, the internal carotid arteries are symmetrical, can be traced along their entire length, their course and caliber are not changed. The siphons of the carotid arteries on both sides are not displaced, without signs of external compression. The intraluminal signal is intense and homogeneous. The MCA on both sides arise normally, with normal segmental division and intraluminal signal, without lumen narrowing or displacement. The stroke and caliber of the PMA has been preserved. The basal artery has a normal course and caliber. The PMA usually recedes, the stroke and caliber are not changed.

# """

# t = generate_conclusion(text)
# print(t)