import torchaudio
import folder_paths
import os
import io
import torch
from PIL import Image
import numpy as np
from io import BytesIO
import json
import struct
from comfy.cli_args import args
import base64
from server import PromptServer

class PMSSendImage:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"default": None, "forceInput": True}),
                "Actions": ("STRING", {"default": None})}
            }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)

    FUNCTION = "sendimgtowc"

    OUTPUT_NODE = True

    CATEGORY = "PMSnodes"
    
    def sendimgtowc(self, images, Actions = ""):
        outs = []

        for single_image in images:
            img = np.asarray(single_image * 255., dtype=np.uint8)
            img = Image.fromarray(img)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            outs.append(img_str)

        PromptServer.instance.send_sync("knodes", {"images": outs, "Actions": Actions})

        return (images,)

class LoadImageBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("IMAGE", "MASK")

    CATEGORY = "PMSnodes"

    FUNCTION = "LoadbaseImg"

    def LoadbaseImg(self, image):
        imgdata = base64.b64decode(image)
        img = Image.open(BytesIO(imgdata))

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]

        return (img, mask)

class LoadMaskBase64:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("STRING", {"multiline": False})}}

    RETURN_TYPES = ("MASK",)
    CATEGORY = "PMSnodes"
    FUNCTION = "pmsload_mask"

    def pmsload_mask(self, mask):
        imgdata = base64.b64decode(mask)
        img = Image.open(BytesIO(imgdata))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)
        if img.dim() == 3:  # RGB(A) input, use red channel
            img = img[:, :, 0]
        return (img.unsqueeze(0),)

class LoadBase64Audio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"base64_audio": ("STRING", {"multiline": True})}}

    CATEGORY = "PMSnodes"

    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "load"

    def load(self, base64_audio):
        audio_bytes = base64.b64decode(base64_audio)
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_buffer)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio, )

def create_vorbis_comment_block(comment_dict, last_block):
    vendor_string = b'ComfyUI'
    vendor_length = len(vendor_string)

    comments = []
    for key, value in comment_dict.items():
        comment = f"{key}={value}".encode('utf-8')
        comments.append(struct.pack('<I', len(comment)) + comment)

    user_comment_list_length = len(comments)
    user_comments = b''.join(comments)

    comment_data = struct.pack('<I', vendor_length) + vendor_string + struct.pack('<I', user_comment_list_length) + user_comments
    if last_block:
        id = b'\x84'
    else:
        id = b'\x04'
    comment_block = id + struct.pack('>I', len(comment_data))[1:] + comment_data

    return comment_block

def insert_or_replace_vorbis_comment(flac_io, comment_dict):
    if len(comment_dict) == 0:
        return flac_io

    flac_io.seek(4)

    blocks = []
    last_block = False

    while not last_block:
        header = flac_io.read(4)
        last_block = (header[0] & 0x80) != 0
        block_type = header[0] & 0x7F
        block_length = struct.unpack('>I', b'\x00' + header[1:])[0]
        block_data = flac_io.read(block_length)

        if block_type == 4 or block_type == 1:
            pass
        else:
            header = bytes([(header[0] & (~0x80))]) + header[1:]
            blocks.append(header + block_data)

    blocks.append(create_vorbis_comment_block(comment_dict, last_block=True))

    new_flac_io = io.BytesIO()
    new_flac_io.write(b'fLaC')
    for block in blocks:
        new_flac_io.write(block)

    new_flac_io.write(flac_io.read())
    return new_flac_io


class PMSSendAudio:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                              "filename_prefix": ("STRING", {"default": "audio/ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"

    OUTPUT_NODE = True

    CATEGORY = "PMSnodes"

    def save_audio(self, audio, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        results = list()

        metadata = {}
        if not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.flac"

            buff = io.BytesIO()
            torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")

            buff = insert_or_replace_vorbis_comment(buff, metadata)

            with open(os.path.join(full_output_folder, file), 'wb') as f:
                f.write(buff.getbuffer())

            # Correctly combine folder path and filename
            audiofile_path = os.path.join(full_output_folder, file)
             # Read the saved FLAC file and convert it to Base64
            with open(audiofile_path, 'rb') as f:
                base64_encodedaudio = base64.b64encode(f.read()).decode('utf-8')

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
        
        PromptServer.instance.send_sync("pmsnodes", {"audio": base64_encodedaudio})
        return { "ui": { "audio": results } }

class PMSLoadText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"default": "Enter your text here", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_text"
    CATEGORY = "PMSnodes"
    
    def load_text(self, text_input):
        return (text_input,)
        
NODE_CLASS_MAPPINGS = {
    "PMSSendAudio": PMSSendAudio,
    "LoadBase64Audio" : LoadBase64Audio,
    "PMSLoadText": PMSLoadText,
    "PMSSendImage" : PMSSendImage,
    "LoadImageBase64" : LoadImageBase64,
    "LoadMaskBase64" : LoadMaskBase64,
    
}
