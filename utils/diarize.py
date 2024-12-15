# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import json
import os

def createManifest(audio_filepath):
    with open(f"_transcripts/text/{audio_filepath}.txt", "r") as f:
        text = f.read()

    meta = {
        'audio_filepath': f'./_audio/{audio_filepath}.wav',
        'offset': 0,
        'duration': None, 
        'label': 'infer', 
        'text': text,
        'num_speakers': None,
        'rttm_filepath': None
    }

    if not os.path.exists('./_msdd'):
        os.mkdir(path='./_msdd')

    with open(f'_msdd/{audio_filepath}-diar_manifest.json','w') as f:
        f.write(json.dumps(meta))
    
    return True

import wget
from omegaconf import OmegaConf
import torch
import gc

def diarize(audio_filepath):
    MODEL_CONFIG = os.path.join('./_conf','diar_infer_telephonic.yaml')

    if not os.path.exists(MODEL_CONFIG):
        if not os.path.exists('./_conf'):
            os.mkdir(path='./_conf')
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
        MODEL_CONFIG = wget.download(config_url, '_conf/')

    config = OmegaConf.load(MODEL_CONFIG)
    config.num_workers = 1
    config.batch_size = 800
    config.verbose = False

    config.diarizer.out_dir = './_msdd_output'
    config.diarizer.manifest_filepath = f'_msdd/{audio_filepath}-diar_manifest.json'

    config.diarizer.clustering.parameters.oracle_num_speakers = False
    config.diarizer.clustering.parameters.max_num_speakers = 40
    config.diarizer.clustering.parameters.sparse_search_volume = 85
    config.diarizer.clustering.parameters.maj_vote_spk_count = True
    config.diarizer.clustering.parameters.diar_window_length = 30
    config.diarizer.clustering.parameters.chunk_cluster_count = 100
    config.diarizer.clustering.parameters.embeddings_per_chunk = 5000 # adjust this to fit your GPU memory

    config.diarizer.msdd_model.parameters.use_speaker_model_from_ckpt = False
    config.diarizer.msdd_model.parameters.infer_batch_size = 4 # adjust this to fit your GPU memory
    
    gc.collect()
    torch.cuda.empty_cache()

    system_vad_msdd_model = NeuralDiarizer(cfg=config)
    system_vad_msdd_model.diarize()

    gc.collect()
    torch.cuda.empty_cache()

    return True

if __name__ == '__main__':
    import time
    start_time = time.time()
    createManifest('judiciary121024_1')
    diarize('judiciary121024_1')
    print(f"Time taken: {time.time() - start_time}")