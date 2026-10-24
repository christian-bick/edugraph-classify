from __future__ import annotations

import json
import random
import shutil
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save_file

from edugraph_classify import runpod_results as results
from edugraph_classify.checkpoint_storage import pack_files
from edugraph_classify.dataset import file_sha256
from edugraph_classify.evaluation import evaluate
from edugraph_classify.learning_data import write_json
from edugraph_classify.ontology import OntologyCatalog
from edugraph_classify.self_hosted_run import resume_identity


class Store:
    def __init__(self, root):
        self.root, self.completion, self.downloads = root, None, 0
        root.mkdir()

    def publish(self, directory, kind, manifest):
        archive = self.root / f"{kind}.tar"
        digest = pack_files(directory, [p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()], archive)
        archive.rename(self.root / f"{digest}.tar")
        return dict(run_id=manifest['run_id'], sha256=digest, bytes=(self.root / f'{digest}.tar').stat().st_size,
                    uri=f"{manifest['recipe']['artifacts']['root_uri']}/{kind}/{manifest['run_id']}/{digest}.tar")

    def read_json(self, uri):
        return self.completion

    def download(self, reference, path):
        self.downloads += 1
        shutil.copyfile(self.root / f"{reference['sha256']}.tar", path)


@pytest.fixture
def completed(tmp_path):
    catalog = OntologyCatalog.load(version('edugraph-py'))
    run, model, checkpoint = [tmp_path / name for name in ('run', 'model', 'checkpoint')]
    for directory in (run, model, checkpoint):
        directory.mkdir()
    recipe = {'run_id': 'repeatable-run', 'ontology_version': catalog.package_version,
              'dataset': {'release': 'test-release'}, 'model': {'hf_revision': 'a' * 40},
              'selection': {'train': 4}, 'evaluation': {'max_tokens': 512},
              'training': {'epochs': 2, 'batch_size': 2, 'early_stop_patience': 1},
              'execution': {'gpu_count': 1}, 'artifacts': {'root_uri': 'gs://bucket/runpod'}}
    gold = catalog.labels(['Addition']).to_mapping()
    examples = [dict(id=f'{split}-{i}', split=split, solution=bool(i % 2), gold=gold,
                     train_diagnostic=split == 'train' and i < 2)
                for split, count in [('train', 4), ('validation', 2), ('final_validation', 2)] for i in range(count)]
    write_json(run / 'recipe.json', recipe)
    write_json(run / 'examples.json', examples)
    for name in ['prompt.json', 'schema.json', 'closed_schema.json', 'chat_template.jinja', 'processor/config.json']:
        (run / name).parent.mkdir(exist_ok=True)
        (run / name).write_text('{}')
        (model / name).parent.mkdir(exist_ok=True)
        shutil.copyfile(run / name, model / name)
    manifest = {'run_id': recipe['run_id'], 'code_commit': 'c' * 40, 'recipe': recipe,
                'uv_lock_sha256': 'd' * 64, 'runtime_versions': {}, 'language_target_audit': {'language_linear_modules': 1},
                'ontology_snapshot_sha256': catalog.snapshot_sha256,
                'files_sha256': {p.relative_to(run).as_posix(): file_sha256(p) for p in run.rglob('*') if p.is_file()}}
    for directory in (run, model, checkpoint):
        write_json(directory / 'manifest.json', manifest)
    names = [f'base_model.model.model.language_model.layers.0.q_proj.lora_{part}.weight' for part in ('A', 'B')]
    weights = {names[0]: torch.ones(2, 3), names[1]: torch.ones(3, 2)}
    for directory, value in [(model / 'adapter', 1.0), (checkpoint / 'best-adapter', 1.0), (checkpoint / 'adapter', 2.0)]:
        directory.mkdir()
        save_file({name: tensor * value for name, tensor in weights.items()}, directory / 'adapter_model.safetensors')
    audit = {'trainable_parameters': 12, 'trainable_tensors': 2, 'vision_or_bridge_trainable_parameters': 0,
             'parameter_names': [name.replace('.weight', '.default.weight') for name in names]}
    (model / 'reports').mkdir()
    write_json(model / 'reports/runtime-audit.json', audit)
    write_json(checkpoint / 'trainable-audit.json', audit)
    cohorts = {name: [r for r in examples if r['split'] == name] for name in ('train', 'validation', 'final_validation')}
    cohorts['train_diagnostic'] = [r for r in cohorts['train'] if r['train_diagnostic']]
    metrics = {}
    for phase in ['baseline', 'epoch-1', 'epoch-2', 'final']:
        cohort_names = ['final_validation'] if phase == 'final' else ['validation', 'train_diagnostic']
        predictions = [dict(id=row['id'], text=json.dumps(gold if phase != 'baseline' else {'areas': [], 'scopes': [], 'abilities': []}),
                            latency_seconds=0.1) for name in cohort_names for row in cohorts[name]]
        metrics[phase] = {name: evaluate(cohorts[name], [p for p in predictions if p['id'] in {r['id'] for r in cohorts[name]}],
                                          cohorts['train'], catalog) for name in cohort_names}
        write_json(model / f'reports/{phase}-metrics.json', metrics[phase])
        write_json(model / f'reports/{phase}-predictions.json', predictions)
    state = {'identity': resume_identity(manifest), 'run_id': manifest['run_id'], 'phase': 'final',
             'epoch': 2, 'step': 4, 'offset': 0, 'best': {'epoch': 1, 'score': [1.0, 1.0]}}
    write_json(checkpoint / 'state.json', state)
    optimizer = {'optimizer': {'param_groups': [{'params': [0, 1]}],
                               'state': {i: dict(step=torch.tensor(4.0), exp_avg=tensor, exp_avg_sq=tensor)
                                         for i, tensor in enumerate(weights.values())}},
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': [torch.get_rng_state()], 'python_rng': random.getstate()}
    torch.save(optimizer, checkpoint / 'optimizer.pt')
    shutil.copytree(model / 'reports', checkpoint / 'reports', ignore=shutil.ignore_patterns('final-*'))
    store = Store(tmp_path / 'store')
    continuation = store.publish(checkpoint, 'checkpoints', manifest)
    completion = {'status': 'completed', 'run_id': manifest['run_id'], 'identity': state['identity'],
                  'epochs_completed': 2, 'selected_epoch': 1, 'step': 4, 'continuation': continuation,
                  'baseline_validation': metrics['baseline']['validation']['explicit'],
                  'selected_validation': metrics['epoch-1']['validation']['explicit'],
                  'validation_delta': {'exact_set_match': 1.0, 'f1': 1.0}, 'quality_promotion': False}
    write_json(model / 'result.json', completion)
    completion['model'] = store.publish(model, 'models', manifest)
    store.completion = completion
    return NS(manifest=manifest, path=run / 'manifest.json', run=run, model=model, checkpoint=checkpoint,
              completion=completion, store=store, state=state, optimizer=optimizer, names=names)


def verify(fixture):
    return results.verify_result(fixture.path, fixture.model, fixture.checkpoint, fixture.completion)


def test_retrieval_is_repeatable_and_verification_is_offline(completed, tmp_path):
    output = tmp_path / 'retrieved'
    retrieved = results.retrieve_result(completed.path, output, completed.store)
    assert completed.store.downloads == 2
    assert results.retrieve_result(completed.path, output, completed.store, completion=completed.completion) == retrieved
    assert completed.store.downloads == 2
    assert json.loads((output / 'completion.json').read_text()) == completed.completion
    assert json.loads((output / 'retrieval.json').read_text()) == retrieved
    report = results.verify_retrieved(completed.path, output)
    assert report['selected_epoch'] == 1 and report['epochs_completed'] == 2 and report['optimizer_steps'] == 4
    assert report['adapter_tensors'] == report['optimizer_states'] == 2
    assert report['all_metrics_reproduced'] and report['continuation_verified']
    assert not report['termination_verified'] and not report['gpu_resume_executed']
    # Verification reconstructs paths from immutable marker references, not a mutable receipt.
    write_json(output / 'retrieval.json', {'model_dir': 'untrusted'})
    assert results.verify_retrieved(completed.path, output) == report
    (Path(retrieved['checkpoint_dir']) / 'state.json').write_text('{}')
    with pytest.raises(ValueError, match='content differs'): results.verify_retrieved(completed.path, output)


@pytest.mark.parametrize('change', ['missing', 'identity', 'counter', 'epoch', 'run', 'hash', 'size', 'prefix'])
def test_completion_rejects_wrong_identity_and_references(completed, change):
    marker = deepcopy(completed.completion)
    if change == 'missing': marker = None
    elif change == 'identity': marker['identity'] = 'wrong'
    elif change == 'counter': marker['step'] = True
    elif change == 'epoch': marker['selected_epoch'] = 3
    elif change == 'run': marker['model']['run_id'] = 'other'
    elif change == 'hash': marker['model']['sha256'] = 'bad'
    elif change == 'size': marker['model']['bytes'] = -1
    else: marker['model']['uri'] = marker['model']['uri'].replace('/models/', '/inputs/')
    with pytest.raises(ValueError): results.validate_completion(completed.manifest, marker)


@pytest.mark.parametrize('tamper', ['archive', 'extracted', 'extra', 'partial', 'download'])
def test_retrieval_rejects_corruption(completed, tmp_path, tamper):
    output = tmp_path / 'output'
    output.mkdir()
    ref = completed.completion['model']
    if tamper == 'partial':
        (output / f"model-{ref['sha256']}.partial").write_text('old attempt')
    elif tamper == 'download':
        completed.store.download = lambda ref, target: target.write_bytes(b'corrupt')
    else:
        result = results.retrieve_result(completed.path, output, completed.store)
        if tamper == 'archive': Path(result['bundles']['model']['archive']).write_bytes(b'corrupt')
        if tamper == 'extracted': (Path(result['model_dir']) / 'prompt.json').write_text('changed')
        if tamper == 'extra': (Path(result['model_dir']) / 'extra').write_text('unknown')
    with pytest.raises(ValueError): results.retrieve_result(completed.path, output, completed.store)
    if tamper == 'download': assert not list(output.glob('*.partial'))


@pytest.mark.parametrize('tamper', ['manifest', 'result', 'portable', 'position', 'audit', 'snapshot', 'choice', 'history'])
def test_verification_binds_inputs_state_and_selection(completed, tamper, monkeypatch):
    if tamper == 'manifest': write_json(completed.model / 'manifest.json', {})
    if tamper == 'result': write_json(completed.model / 'result.json', {})
    if tamper == 'portable': (completed.model / 'prompt.json').write_text('changed')
    if tamper == 'position': write_json(completed.checkpoint / 'state.json', {**completed.state, 'step': 5})
    if tamper == 'audit':
        audit = results._read(completed.model / 'reports/runtime-audit.json')
        write_json(completed.model / 'reports/runtime-audit.json', {**audit, 'vision_or_bridge_trainable_parameters': 1})
    if tamper == 'snapshot':
        monkeypatch.setattr(results.OntologyCatalog, 'load', lambda version: NS(snapshot_sha256='wrong'))
    if tamper == 'history': write_json(completed.checkpoint / 'reports/epoch-1-predictions.json', [])
    if tamper == 'choice':
        completed.completion['selected_epoch'] = 2
        write_json(completed.model / 'result.json', {k: v for k, v in completed.completion.items() if k != 'model'})
        write_json(completed.checkpoint / 'state.json', {**completed.state, 'best': {'epoch': 2, 'score': [1.0, 1.0]}})
    with pytest.raises(ValueError): verify(completed)


@pytest.mark.parametrize('tamper', ['names', 'nonfinite', 'best', 'inventory'])
def test_verification_checks_real_adapter_tensors(completed, tamper):
    path = completed.model / 'adapter/adapter_model.safetensors'
    tensors = {name: torch.ones(2, 3) if i == 0 else torch.ones(3, 2) for i, name in enumerate(completed.names)}
    if tamper == 'names': tensors = {'vision.weight': torch.ones(2)}
    if tamper == 'nonfinite': tensors[completed.names[0]][0, 0] = float('nan')
    if tamper == 'best': tensors[completed.names[0]][0, 0] = 99
    if tamper == 'inventory': path = completed.checkpoint / 'adapter/adapter_model.safetensors'; tensors.pop(completed.names[0])
    save_file(tensors, path)
    with pytest.raises(ValueError): verify(completed)


@pytest.mark.parametrize('tamper', ['groups', 'states', 'step', 'nonfinite', 'shape', 'rng'])
def test_verification_checks_optimizer_and_rng(completed, tamper):
    saved = completed.optimizer
    if tamper == 'groups': saved['optimizer']['param_groups'][0]['params'] = [0]
    if tamper == 'states': saved['optimizer']['state'].pop(1)
    if tamper == 'step': saved['optimizer']['state'][0]['step'] = torch.tensor(3.)
    if tamper == 'nonfinite': saved['optimizer']['state'][0]['exp_avg'][0, 0] = float('inf')
    if tamper == 'shape':
        saved['optimizer']['state'][0].update(exp_avg=torch.ones(5), exp_avg_sq=torch.ones(5))
    if tamper == 'rng': saved['cuda_rng'] = []
    torch.save(saved, completed.checkpoint / 'optimizer.pt')
    with pytest.raises(ValueError): verify(completed)


@pytest.mark.parametrize('tamper', ['phase', 'predictions', 'metrics'])
def test_verification_reproduces_every_evaluation_phase(completed, tamper):
    if tamper == 'phase': (completed.model / 'reports/epoch-1-metrics.json').unlink()
    if tamper == 'predictions': write_json(completed.model / 'reports/final-predictions.json', [])
    if tamper == 'metrics':
        path = completed.model / 'reports/final-metrics.json'
        metrics = results._read(path)
        metrics['final_validation']['explicit']['f1'] = 0.0
        write_json(path, metrics)
    with pytest.raises(ValueError): verify(completed)


def test_smoke_without_final_cohort_and_early_stop(completed):
    examples = [r for r in results._read(completed.run / 'examples.json') if r['split'] != 'final_validation']
    write_json(completed.run / 'examples.json', examples)
    completed.manifest['files_sha256']['examples.json'] = file_sha256(completed.run / 'examples.json')
    completed.manifest['recipe']['training']['epochs'] = 5  # Two completed epochs, then patience-one stop.
    write_json(completed.run / 'recipe.json', completed.manifest['recipe'])
    completed.manifest['files_sha256']['recipe.json'] = file_sha256(completed.run / 'recipe.json')
    identity = resume_identity(completed.manifest)
    for directory in (completed.run, completed.model, completed.checkpoint): write_json(directory / 'manifest.json', completed.manifest)
    completed.completion['identity'] = completed.state['identity'] = identity
    write_json(completed.checkpoint / 'state.json', completed.state)
    write_json(completed.model / 'result.json', {k: v for k, v in completed.completion.items() if k != 'model'})
    for path in (completed.model / 'reports').glob('final-*'): path.unlink()
    report = verify(completed)
    assert report['epochs_completed'] == 2 and 'final' not in report['metrics']


def test_continuation_retains_previous_selection_and_adds_completed_phases(completed):
    completed.manifest['recipe']['training']['epochs'] = 3
    write_json(completed.run / 'recipe.json', completed.manifest['recipe'])
    completed.manifest['files_sha256']['recipe.json'] = file_sha256(completed.run / 'recipe.json')
    for directory in (completed.run, completed.model, completed.checkpoint):
        write_json(directory / 'manifest.json', completed.manifest)
    for directory in (completed.model, completed.checkpoint):
        for suffix in ('metrics', 'predictions'):
            shutil.copyfile(directory / f'reports/epoch-2-{suffix}.json', directory / f'reports/epoch-3-{suffix}.json')
    completed.state.update(epoch=3, step=6, source_run_id='earlier-run')
    completed.completion.update(epochs_completed=3, step=6)
    write_json(completed.checkpoint / 'state.json', completed.state)
    write_json(completed.model / 'result.json', {k: v for k, v in completed.completion.items() if k != 'model'})
    for item in completed.optimizer['optimizer']['state'].values(): item['step'] = torch.tensor(6.)
    torch.save(completed.optimizer, completed.checkpoint / 'optimizer.pt')
    report = verify(completed)
    assert report['selected_epoch'] == 1 and report['optimizer_steps'] == 6
    assert set(report['metrics']) == {'baseline', 'epoch-1', 'epoch-2', 'epoch-3', 'final'}


def test_offline_verification_rechecks_archives(completed, tmp_path):
    output = tmp_path / 'retrieved'
    receipt = results.retrieve_result(completed.path, output, completed.store)
    Path(receipt['bundles']['model']['archive']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum or size'): results.verify_retrieved(completed.path, output)
