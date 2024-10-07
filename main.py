import os
import json
import glob
import re
import cv2
import torch
import numpy as np

from pypdf import PdfReader
from transformers import DonutProcessor, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# with open("config.json", "r") as f:
#     config = json.loads(f.read())

with open("result.json", "r") as f:
    result = json.loads(f.read())

# states = list(map(lambda x: x.lower(), config["states"]))
dob_keys = ["-1979", "1980-1989", "1990-1999", "2000-"]

processor = DonutProcessor.from_pretrained("thinkersloop/donut-demo")
pretrained_model = VisionEncoderDecoderModel.from_pretrained(
    "thinkersloop/donut-demo"
).to(device)

files = glob.glob("*.pdf")


def parse_dob(dob):
    if dob < 1979:
        return 0
    elif dob > 1979 and dob < 1990:
        return 1
    elif dob > 1990 and dob < 2000:
        return 2
    else:
        return 3


for file in files:
    filename = os.path.basename(file)

    try:
        reader = PdfReader(file)
    except:
        # result["other"].append(filename)
        continue

    for image in reader.pages.images:

        try:
            with open(image.name, "wb") as fp:
                img = np.array(image.image)
        except:
            # result["other"].append(filename)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pixel_values = processor(img, return_tensors="pt").pixel_values

        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]

        outputs = pretrained_model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=pretrained_model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
        )
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
            processor.tokenizer.pad_token, ""
        )
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

        out = processor.token2json(sequence)

        print("Processing {} ...".format(filename))

        if "name" not in out.keys() or "date" not in out.keys():
            # result["other"].append(filename)
            continue
        else:
            if (
                "state" not in out["name"]
                or "person" not in out["name"]
                or "dob" not in out["date"]
            ):
                # result["other"].append(filename)
                continue

            else:
                # if out["name"]["state"].lower() not in states:
                #     result["other"].append(filename)

                # else:
                state = out["name"]["state"].lower()
                try:
                    dob = int(out["date"]["dob"][-4:])
                except Exception as e:
                    # result["other"].append(filename)
                    continue

                dob_idx = parse_dob(dob)

                if state == "michiga":
                    if dob_keys[dob_idx] not in result["michigan"].keys():
                        result["michigan"][dob_keys[dob_idx]] = [filename]
                    else:
                        result["michigan"][dob_keys[dob_idx]].append(filename)

                    target_idx = result["other"].index(filename)
                    result["other"].pop(target_idx)

                    old_path = os.path.join(
                        "data", "downloaded_files_breezemed", filename
                    )
                    new_path = os.path.join("data", "classified", filename)
                    os.rename(new_path, old_path)

                # if state not in result.keys():
                #     result[state] = {dob_keys[dob_idx]: [filename]}
                # else:
                #     if dob_keys[dob_idx] not in result[state].keys():
                #         result[state][dob_keys[dob_idx]] = [filename]
                #     else:
                #         result[state][dob_keys[dob_idx]].append(filename)

with open("result.json", "w") as f:
    f.write(json.dumps(result))
