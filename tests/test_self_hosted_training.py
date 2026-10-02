from __future__ import annotations

import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from edugraph_classify.constrained_generation import json_constraint, tokenizer_constraints
from edugraph_classify.self_hosted_training import TorchTrainer, model_batch, trainable_audit
from edugraph_classify.qwen_rendering import VisionExample
from test_learning import png
from test_runpod import recipe


class CharTokenizer:
    alphabet = list(dict.fromkeys('0 {}[],:"AdditionSubtractionareascopesbiltyZxyz\n`ä'))
    all_special_ids = [0]
    eos_token_id = 0

    def __len__(self): return len(self.alphabet) + 1

    def encode(self, text, **kwargs): return [self.alphabet.index(c) + 1 for c in text]

    def decode(self, ids, **kwargs): return ''.join(self.alphabet[i-1] for i in ids if i)


class Pixels:
    merge_size = 2

    def __call__(self, **kwargs):
        return {'image_grid_thw':torch.tensor([[1,4,8]]), 'pixel_values':torch.zeros((32,4))}


def example(completion=(4,5)):
    return VisionExample((1,2),(3,),completion,png(),8,99)


def test_masks_image_span_and_multimodal_rope_types():
    ex = example()
    batch = model_batch(ex,Pixels(),training=True)
    assert batch['input_ids'][0].tolist()==[1,2,*([99]*8),3,4,5]
    assert batch['labels'][0].tolist()==[-100]*11+[4,5]
    assert batch['mm_token_type_ids'][0].tolist()==[0,0]+[1]*8+[0,0,0]
    assert len(model_batch(ex,Pixels(),training=False)['input_ids'][0])==ex.prompt_tokens
    with pytest.raises(ValueError,match='different number'):
        model_batch(example().__class__((1,),(2,),(3,),png(),7,99),Pixels(),training=True)


def test_constraints_enforce_json_and_vocabulary_without_fixing_label_order():
    tokenizer = CharTokenizer()
    data = tokenizer_constraints(tokenizer)
    schema = {'type':'object','properties':{'areas':{'type':'array','items':{'type':'string','enum':['Addition','Subtraction']}}},
              'required':['areas'],'additionalProperties':False}
    for text in ['{"areas":["Addition","Subtraction"]}', '{"areas":["Subtraction","Addition"]}']:
        fn=json_constraint(data,schema)
        prefix=[200]
        assert tokenizer.encode('`')[0] not in fn(0,torch.tensor(prefix))
        for token in tokenizer.encode(text):
            assert token in fn(0,torch.tensor(prefix))
            assert 0 not in fn(0,torch.tensor(prefix))
            prefix.append(token)
        assert 0 in fn(0,torch.tensor(prefix))
    fn=json_constraint(data,schema)
    prefix=[200]
    for token in tokenizer.encode('{"areas":["'):
        fn(0,torch.tensor(prefix)); prefix.append(token)
    assert tokenizer.encode('Z')[0] not in fn(0,torch.tensor(prefix))


def test_trainable_audit_rejects_vision_bridge_and_output_head():
    valid='base_model.model.model.language_model.layers.0.q_proj.lora_A.default.weight'
    model=NS(named_parameters=lambda:[(valid,torch.nn.Parameter(torch.zeros(2,3))),
                                     ('model.visual.weight',torch.nn.Parameter(torch.zeros(3),requires_grad=False))])
    assert trainable_audit(model)['trainable_parameters']==6
    for name in ['base_model.model.model.visual.q_proj.lora_A.default.weight',
                 'base_model.model.lm_head.weight','base_model.model.model.language_model.embed_tokens.weight']:
        model.named_parameters=lambda:[(name,torch.nn.Parameter(torch.zeros(1)))]
        with pytest.raises(ValueError,match='non-language'): trainable_audit(model)
    model.named_parameters=lambda:[]
    with pytest.raises(ValueError,match='empty'): trainable_audit(model)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(0.3))

    def forward(self, **batch):
        target=batch['labels'][batch['labels']!=-100].float().mean()
        return NS(loss=(self.weight-target).square())

    def save_pretrained(self,path,**kwargs):
        path.mkdir(exist_ok=True,parents=True)
        torch.save({'weight':self.weight.detach().clone()},path/'adapter.pt')

    def generate(self, **kwargs):
        return torch.cat((kwargs['input_ids'],torch.tensor([[1,2,0]])),dim=1)


def trainer(monkeypatch):
    import peft
    import peft.utils.save_and_load as persistence
    monkeypatch.setattr(persistence,'load_peft_weights',lambda path,**kwargs:torch.load(path+'/adapter.pt',weights_only=True))
    monkeypatch.setattr(peft,'set_peft_model_state_dict',lambda model,state:model.load_state_dict(state))
    renderer=NS(image_processor=Pixels(),tokenizer=CharTokenizer())
    return TorchTrainer(TinyModel(),renderer,recipe()['training'],audit={'verified':True})


def test_optimizer_resume_matches_uninterrupted_and_handles_partial_batch(tmp_path,monkeypatch):
    first=trainer(monkeypatch)
    a,b=example((2,3)),example((4,5,6,7))
    step=first.train_batch([a,b])
    expected=((0.3-2.5)**2*2+(0.3-5.5)**2*4)/6
    assert step['loss']==pytest.approx(expected)
    assert step['supervised_tokens']==6
    first.save(tmp_path)
    first.train_batch([a])
    second=trainer(monkeypatch)
    second.restore(tmp_path)
    second.train_batch([a])
    assert torch.equal(first.model.weight,second.model.weight)
    assert first.optimizer.state_dict()['state'][0]['step']==second.optimizer.state_dict()['state'][0]['step']
    prediction=second.predict(a,{'max_tokens':20},{'type':'object'})
    assert prediction['usage']['completion_tokens']==3
    assert prediction['text']=='0 '
    second.model.forward=lambda **kwargs:NS(loss=second.model.weight*float('nan'))
    with pytest.raises(ValueError,match='nonfinite'): second.train_batch([a])


def test_trainer_default_audit_and_cuda_rng_branches(tmp_path,monkeypatch):
    import edugraph_classify.self_hosted_training as module
    monkeypatch.setattr(module,'trainable_audit',lambda model:{'verified':True})
    engine=trainer(monkeypatch)
    other=TorchTrainer(TinyModel(),engine.renderer,recipe()['training'],constraints=engine.constraints)
    assert other.audit['verified']
    engine.device=torch.device('cuda')
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[torch.tensor([1],dtype=torch.uint8)])
    restored=Mock()
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',restored)
    engine.save(tmp_path)
    engine.restore(tmp_path)
    assert restored.call_args.args[0][0].tolist()==[1]
