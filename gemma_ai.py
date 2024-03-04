import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, GemmaTokenizer
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

model_id = "abdibrokhim/gemma-fine-tuned-brainmri-2402"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_config,
                                             device_map={"":0},
                                             token=os.environ['HF_TOKEN'])


def generate_conclusion(text):
    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=20)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    start_index = generated_text.find("Conclusion") + len("Conclusion")
    
    end_index = generated_text.find(".", start_index)
    conclusion_text = generated_text[start_index:end_index].strip()

    return conclusion_text



# Usage
# text = """
# Question: \n
# Scanning technique: T1 FSE-sagital, T2 FLAIR, T2 FSE-axial, T2 FSE-coronar.
# On a series of tomograms, in the structure of the subcortical and periventricular white matter of the cerebral hemispheres on both sides, single round-shaped foci of pathological intensity are not symmetrically determined, with unclear contours, homogeneous hyperintense signal characteristics on T2 VI and T2 FLAIR, with a diameter of 1-3 mm.
# The longitudinal fissure of the cerebrum is located centrally. Convexital grooves are not expanded, their number is not changed. The thickness of the cerebral cortex is preserved, the distribution of gray matter is not disturbed.
# The lateral ventricles are symmetrical, the width of the ventricles at the level of the foramen of Monro is 5 mm on the right, 6 mm on the left. The third ventricle is 3 mm wide. Sylvian aqueduct has not been changed. The fourth ventricle is tent-shaped and not dilated. The cisterna magna is dilated with retrocerbellar extension.
# The basal ganglia are usually located, symmetrical, with clear, even contours, the dimensions are not changed, the MR signal is not changed. The corpus callosum is of normal shape and size. The brain stem is without features. The cerebellum is of normal shape, the signal characteristics of the white matter are not changed. The width of the cerebellar cortex is preserved. The craniovertebral junction is unchanged.
# The pituitary gland is of normal shape, height in the sagittal projection is 6 mm. The pituitary stalk is located centrally. The chiasma of the optic nerves is located usually, the contours are clear and even. Parasellar cisterns without areas of pathological intensity. The siphons of the internal carotid arteries are not changed. The cavernous sinuses of both carotid arteries are symmetrical, with clear, even contours.
# The shape of the orbital cones on both sides is unchanged. The eyeballs are spherical in shape and of normal size; the MR signal of the vitreous body is not changed. The diameter of the optic nerves was preserved. The perineural subarachnoid space of the orbits is moderately diffusely dilated. The extraocular muscles are of normal size, the structure is without pathological signals. Retrobulbar fatty tissue without pathological signals.
# Region of the cerebellopontine angle: the prevestocochlear nerve is clearly differentiated on both sides. Pneumatization of the cells of the mastoid processes of the temporal bones on the left is not impaired, on the right it is reduced due to sclerotic changes.
# The paranasal sinuses are developed correctly. The nasal turbinates are markedly hypertrophied. There is thickening of the mucous membranes of the cells of the ethmoidal labyrinth on both sides. In the right maxillary sinus, uneven hyperplasia of the mucous membrane with the presence of inclusions is noted, with transitional signal characteristics on T1 and T2 VI; on the left, there is thickening of the mucous membrane up to 4 mm. In the main sinus on the right, a cyst measuring 11x10 mm is detected. The nasal septum is slightly curved to the left.

# Write Conclusion:
# \n
# """

# t = generate_conclusion(text)
# print(t)
    