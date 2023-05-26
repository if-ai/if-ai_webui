import modules.scripts as scripts
import gradio as gr
import os
import requests
import json
from modules import images
from modules.processing import Processed, process_images
from modules.shared import state

params = {
    'selected_character': None,
    'pre_prompt': 'has nothing yet',
    'pre_prompt_text': 'this text should update when the character is changed',
    'prompt_prefix': 'Style-SylvaMagic, ',
    'input_prompt': '(Dark elf empress:1.2), enchanted Forrest',
    'negative_prompt': '(worst quality, low quality:1.3)',
    'prompt_subfix': '(rim lighting,:1.1) two tone lighting, <lora:epiNoiseoffset_v2:0.8>',
}

script_dir = scripts.basedir()
print(f"Script dir: {script_dir}")

character_dir = os.path.join(script_dir, "characters")
print(f"Character dir: {character_dir}")

def get_characters(character_dir):
    characters = []

    for json_file in os.listdir(character_dir):
        with open(os.path.join(character_dir, json_file), 'r') as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError as e:
                print(f"Error parsing {json_file}: {e}")
                continue
            try:
                char_name = str(data["char_name"])
                example_dialogue = str(data["example_dialogue"])
                characters.append((char_name, example_dialogue))
            except KeyError as e:
                print(f"Error parsing {json_file}: {e}")
                continue

    return characters

characters = get_characters(character_dir)

class ifpromptmkrScript(scripts.Script): 
    
    
    def title(self):
        return "if prompt mkr"
    

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):

        with gr.Accordion('if prompt mkr', open=False):
            with gr.Column():
                selected_character = gr.Dropdown(
                    label="characters", 
                    choices=[char[0] for char in characters],
                    type="index", 
                    elem_id="if_prompt_mkr_dropdown",
                    default=0
                )
                prompt_prefix = gr.Textbox(lines=1, placeholder=params['prompt_prefix'], value=params['prompt_prefix'], label="Prompt Prefix")
                input_prompt = gr.Textbox(lines=1, placeholder=params['input_prompt'], value=params['input_prompt'], label="Input Prompt")
                prompt_subfix = gr.Textbox(lines=1, placeholder=params['prompt_subfix'], value=params['prompt_subfix'], label="Subfix for adding Loras (optional)")
                negative_prompt = gr.Textbox(lines=2, placeholder=params['negative_prompt'], value=params['negative_prompt'], label="Negative Prompt")
                
            selected_character.change(lambda x: (
                params.update({'pre_prompt_text':characters[x][1]}), 
                print(f"Updated pre_prompt_text: {params['pre_prompt_text']}")), 
            selected_character, None)
            prompt_prefix.change(lambda x: params.update({"prompt_prefix": x}), prompt_prefix, None)
            input_prompt.change(lambda x: params.update({"input_prompt": x}), input_prompt, None)
            prompt_subfix.change(lambda x: params.update({"prompt_subfix": x}), prompt_subfix, None)
            negative_prompt.change(lambda x: params.update({"negative_prompt": x}), negative_prompt, None)
            
        return [prompt_prefix, input_prompt, negative_prompt, prompt_subfix, selected_character]



    def run(self, p, prompt_prefix, input_prompt, negative_prompt, prompt_subfix, selected_character, *args, **kwargs):
        if selected_character is not None:
            generated_text = self.generate_text(input_prompt)
            prompt_generated_text = prompt_prefix + ' ' + generated_text + ' ' + prompt_subfix

            
            p.prompt = prompt_generated_text
            p.negative_prompt = negative_prompt
            return self.process_images(p)

    
    def generate_text(self, prompt):
        print("Generating text...")
        pre_prompt = params['pre_prompt_text']
        
        data = {
            'prompt': pre_prompt + ' ' + prompt,
            'max_new_tokens': 140,
            'do_sample': True,
            'temperature': 0.3,
            'top_p': 0.9,
            'typical_p': 1,
            'repetition_penalty': 1,
            'encoder_repetition_penalty': 1.0,
            'top_k': 30,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'seed': -1,
            'add_bos_token': True,
            'custom_stopping_strings': ["You:",],
            'truncation_length': 2048,
            'ban_eos_token': False,
        }  
        headers = {     
        "Content-Type": "application/json" 
        } 

        response = requests.post("http://127.0.0.1:5000/api/v1/generate",
                                 data=json.dumps(data), headers=headers)

        if response.status_code == 200:
            results = json.loads(response.content)["results"]
            generated_text = ""
            for result in results:
                generated_text += result["text"]
            return generated_text
        else:
            return f"Request failed with status code {response.status_code}."
    
    def process_images(self, p):
        state.job_count = 0
        state.job_count += p.n_iter

        proc = process_images(p)

        return Processed(p, [proc.images[0]], p.seed, "", all_prompts=proc.all_prompts, infotexts=proc.infotexts)