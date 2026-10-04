from __future__ import annotations

import io
import json
import tarfile
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, MagicMock

import httpx
import pytest

from edugraph_classify import cli, checkpoint_storage as storage, runpod_cli, runpod_config, runpod_worker
from edugraph_classify import self_hosted_run as run
from edugraph_classify import training_tracking as tracking
from edugraph_classify.learning_data import prepare_learning, write_json
from edugraph_classify.providers.runpod import RunpodClient, RunpodPodClient, RunpodError, public_status
from test_learning import GOLD, ROOT, png, renderer_factory
from test_gcs_artifacts import FakeClient


def recipe():
    return runpod_config.load_runpod_recipe(ROOT / runpod_cli.DEFAULT_RECIPE)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    config = recipe()
    config['selection'].update(train=4, validation=4, train_diagnostic=2, final_validation=2)
    config['training'].update(batch_size=2, max_context_tokens=50000)
    config['execution']['checkpoint_steps'] = 1
    processor = tmp_path / 'processor'
    processor.mkdir()
    (processor / 'config.json').write_text('{}')
    monkeypatch.setattr('edugraph_classify.learning_data.audit_qwen_language_targets', lambda *a: {'fake': True})
    def fetch(url):
        if url.endswith('metadata.jsonl'):
            return '\n'.join(json.dumps(dict(file_name=f'{n}.png', labels=['Addition'], solution=bool(n % 2))) for n in range(6)).encode()
        n = int(url.rsplit('/', 1)[-1].split('.')[0])
        return png(n + (20 if '/validation/' in url else 0))
    directory = tmp_path / 'prepared'
    manifest = prepare_learning(config, ROOT, directory, processor, 'a' * 40, fetch, {}, renderer_factory,
                                runtime_packages=('pillow',))
    return NS(run=directory, path=directory/'manifest.json', manifest=manifest, config=config)


class Store:
    def __init__(self, root):
        self.root, self.references, self.completions = root, [], []
        root.mkdir()

    def publish(self, path, prefix, run_id):
        digest = storage.file_sha256(path)
        (self.root / (digest + '.tar')).write_bytes(path.read_bytes())
        reference = dict(uri=prefix+'/'+digest+'.tar', sha256=digest, run_id=run_id)
        self.references.append(reference)
        return reference

    def download(self, ref, path):
        path.write_bytes((self.root / (ref['sha256']+'.tar')).read_bytes())

    def complete(self, reference, uri):
        self.completions.append((reference, uri))

    def read_json(self, uri):
        return next((value for value, key in reversed(self.completions) if key == uri), None)


class Engine:
    def __init__(self, *a):
        self.steps = 0
        self.audit = {'vision_or_bridge_trainable_parameters': 0}
        self.seen = []

    def train_batch(self, examples):
        self.steps += 1
        self.seen.extend(e.fingerprint() for e in examples)
        return {'loss': 1 / self.steps}

    def predict(self, example, evaluation, schema):
        return dict(text=json.dumps(GOLD if self.steps else {'areas': [], 'scopes': [], 'abilities': []}),
                    latency_seconds=0.1, usage={'prompt_tokens': 10, 'completion_tokens': 5})

    def save_adapter(self, path):
        path.mkdir(exist_ok=True, parents=True)
        write_json(path/'weights.json', {'steps': self.steps})

    def load_adapter(self, path):
        self.steps = json.loads((path/'weights.json').read_text())['steps']

    def save(self, directory):
        self.save_adapter(directory/'adapter')
        (directory/'optimizer.pt').write_text(str(self.steps))

    def restore(self, directory):
        self.load_adapter(directory/'adapter')
        assert self.steps == int((directory/'optimizer.pt').read_text())


def execute(prepared, work, store, engine=None, **kwargs):
    tracker = Mock()
    result = run.execute_self_hosted(prepared.path, 'a'*40, work, lambda *a: engine or Engine(),
        lambda *a, **k: tracker, store, renderer_factory=renderer_factory, **kwargs)
    return result, tracker


def test_learning_selection_export_and_interrupted_recovery(prepared, tmp_path):
    store = Store(tmp_path/'store')
    full_engine = Engine()
    comparison_store = Store(tmp_path/'comparison-store')
    full, tracker = execute(prepared, tmp_path/'full', comparison_store, full_engine)
    assert full['selected_epoch'] == 1 and full['step'] == 4
    assert comparison_store.completions and tracker.finish.call_args.args == (True,)
    assert full_engine.steps == 2  # Best adapter, not last optimizer position.
    paused_engine = Engine()
    paused, _ = execute(prepared, tmp_path/'partial', store, paused_engine, stop_requested=lambda: True)
    assert paused['status'] == 'paused' and paused['step'] == 1
    ref = paused['checkpoint']
    restored = tmp_path/'restore'
    storage.unpack_verified(store.root/(ref['sha256']+'.tar'), ref['sha256'], restored)
    resumed_engine = Engine()
    resumed, _ = execute(prepared, tmp_path/'resumed', store, resumed_engine, resume=restored)
    assert resumed['step'] == full['step']
    assert paused_engine.seen + resumed_engine.seen == full_engine.seen
    assert json.loads((tmp_path/'resumed/reports/final-metrics.json').read_text())['final_validation']['explicit']['exact_set_match'] == 1
    final_snapshot = tmp_path/'final-snapshot'
    ref = full['continuation']
    storage.unpack_verified(comparison_store.root/(ref['sha256']+'.tar'), ref['sha256'], final_snapshot)
    assert json.loads((final_snapshot/'adapter/weights.json').read_text())['steps'] == 4
    assert json.loads((final_snapshot/'best-adapter/weights.json').read_text())['steps'] == 2
    retry, _ = execute(prepared, tmp_path/'retry-final', store, resume=final_snapshot)
    assert retry['step'] == 4
    assert not (tmp_path/'retry-final').exists()  # Already durable: no evaluation is repeated.
    saved_completions = store.completions
    store.completions = []
    retry, _ = execute(prepared, tmp_path/'missing-final', store, resume=final_snapshot)
    assert retry['step'] == 4
    store.completions = [({**saved_completions[-1][0], 'identity':'wrong'}, saved_completions[-1][1])]
    with pytest.raises(ValueError,match='different training identity'):
        execute(prepared,tmp_path/'collision',store,resume=final_snapshot)
    # A new W&B run can extend the total epoch ceiling with coherent state.
    extended = deepcopy(prepared.manifest)
    extended['run_id'] = extended['recipe']['run_id'] = 'continued-run'
    with pytest.raises(ValueError, match='larger total'):
        run.verify_resume(extended, final_snapshot)
    extended['recipe']['training']['epochs'] = 3
    assert run.verify_resume(extended, final_snapshot)['epoch'] == 2
    continuation_dir = tmp_path/'continued-inputs'
    shutil.copytree(prepared.run,continuation_dir)
    write_json(continuation_dir/'recipe.json',extended['recipe'])
    extended['files_sha256']['recipe.json']=storage.file_sha256(continuation_dir/'recipe.json')
    write_json(continuation_dir/'manifest.json',extended)
    next_input=NS(path=continuation_dir/'manifest.json')
    continued,_=execute(next_input,tmp_path/'continued-training',store,resume=final_snapshot)
    assert continued['epochs_completed']==3 and continued['step']==6
    extended['recipe']['training']['learning_rate'] *= 2
    with pytest.raises(ValueError, match='incompatible'):
        run.verify_resume(extended, final_snapshot)
    (restored/'optimizer.pt').unlink()
    with pytest.raises(ValueError, match='optimizer'):
        run.verify_resume(prepared.manifest, restored)


def test_training_failures_are_not_reported_as_success(prepared, tmp_path, monkeypatch):
    store = Store(tmp_path/'store')
    engine, tracker = Engine(), Mock()
    engine.train_batch = Mock(side_effect=ValueError('bad loss'))
    with pytest.raises(ValueError, match='bad loss'):
        run.execute_self_hosted(prepared.path, 'a'*40, tmp_path/'failed', lambda *a: engine,
            lambda *a, **k: tracker, store, renderer_factory=renderer_factory)
    tracker.finish.assert_called_once_with(False)
    for key, value, message in [('training_data_conflicts', ['conflict'], 'conflicting'),
        ('runtime_versions', {'pillow':'0.0'}, 'installed packages'), ('ontology_snapshot_sha256','bad','ontology changed')]:
        manifest = deepcopy(prepared.manifest)
        manifest[key] = value
        monkeypatch.setattr(run, 'verify_prepared', lambda *a, m=manifest: (m, []))
        with pytest.raises(ValueError, match=message):
            execute(prepared, tmp_path/'unused', store)


def test_smoke_omits_final_assessment_and_uses_sparse_checkpoints(prepared,tmp_path):
    manifest=deepcopy(prepared.manifest)
    manifest['recipe']['execution']['checkpoint_steps']=50
    manifest['recipe']['training']['epochs']=1
    examples=json.loads((prepared.run/'examples.json').read_text())
    examples=[row for row in examples if row['split']!='final_validation']
    write_json(prepared.run/'examples.json',examples)
    write_json(prepared.run/'recipe.json',manifest['recipe'])
    for name in ('examples.json','recipe.json'):
        manifest['files_sha256'][name]=storage.file_sha256(prepared.run/name)
    write_json(prepared.path,manifest)
    store=Store(tmp_path/'store')
    result,_=execute(prepared,tmp_path/'smoke',store)
    assert result['epochs_completed']==1
    assert not (tmp_path/'smoke/reports/final-metrics.json').exists()


def test_recipes_and_secret_free_pod_request(prepared):
    staged = {'run_id':prepared.config['run_id'], 'uri':'gs://edugraph-classify/runpod/inputs/a.tar','sha256':'b'*64,
              'manifest_identity':runpod_config.manifest_identity(prepared.manifest)}
    image = 'ghcr.io/christian-bick/edugraph-classify-trainer@sha256:'+'c'*64
    request = runpod_config.pod_request(prepared.manifest, image, staged, resume=staged, attempt='recovery1')
    assert request['cloudType'] == 'SECURE' and request['gpuCount'] == 1
    assert request['env']['WANDB_API_KEY'] == '{{ RUNPOD_SECRET_WANDB_API_KEY }}'
    assert 'RUNPOD_API_KEY' not in request['env']
    with pytest.raises(ValueError,match='fresh --attempt'):
        runpod_config.pod_request(prepared.manifest,image,staged,resume=staged)
    with pytest.raises(ValueError,match='attempt name'):
        runpod_config.pod_request(prepared.manifest,image,staged,attempt='bad/name')
    for img, ref, res in [('bad:latest', staged, None), (image, {**staged, 'run_id':'bad'}, None),
                           (image,{**staged,'uri':'gs://other-bucket/a'},None), (image,staged,{'uri':staged['uri'],'sha256':'bad'})]:
        with pytest.raises(ValueError):
            runpod_config.pod_request(prepared.manifest,img,ref,resume=res)


@pytest.mark.parametrize('section,key,value', [
    ('model','provider','fireworks'),('execution','cloud_type','COMMUNITY'),('execution','cloud_type','ALL'),('execution','gpu_type','bad/'),
    ('execution','gpu_count',2),('execution','minimum_ram_gib',False),('execution','max_hours',0),
    ('execution','completion_action','keep'),('evaluation','output_mode','raw'),
    ('training','expected_target_modules',[]),('training','train_attn',False),('training','beta1',1),
    ('training','eps',0),('training','max_grad_norm',5),('tracking','project','space bad'),
    ('tracking','entity','bad/'),('artifacts','root_uri','gs://bucket/../escape'),
    ('pricing','estimated_hours',30),('pricing','compute_hour_usd',100),
])
def test_bad_policy_rejected(section,key,value):
    config=recipe()
    config[section][key]=value
    with pytest.raises(ValueError):
        runpod_config.validate_runpod_recipe(config)


def test_extra_policy_keys_rejected():
    c=recipe()
    c['tracking']['key_value']='secret'
    with pytest.raises(ValueError,match='fields'):
        runpod_config.validate_runpod_recipe(c)


def test_bundle_roundtrip_and_path_rejection(tmp_path,monkeypatch):
    root=tmp_path/'root'; root.mkdir()
    (root/'a').write_text('payload')
    archive=tmp_path/'bundle.tar'
    digest=storage.pack_files(root,['a'],archive)
    assert storage.pack_files(root,['a'],archive)==digest
    storage.unpack_verified(archive,digest,tmp_path/'unpacked')
    assert (tmp_path/'unpacked/a').read_text()=='payload'
    with pytest.raises(ValueError,match='checksum'):
        storage.unpack_verified(archive,'bad',tmp_path/'bad')
    with pytest.raises(ValueError,match='relative'):
        storage.pack_files(root,['../outside'],archive)
    with monkeypatch.context() as m:
        m.setattr(Path,'is_symlink',lambda self:True)
        with pytest.raises(ValueError,match='links'):
            storage.pack_files(root,['a'],archive)
    for names,kind in [(['../escape'],tarfile.REGTYPE),(['a','a'],tarfile.REGTYPE),(['link'],tarfile.SYMTYPE)]:
        with tarfile.open(archive,'w') as tar:
            for name in names:
                info=tarfile.TarInfo(name); info.type=kind
                tar.addfile(info)
        with pytest.raises(ValueError):
            storage.unpack_verified(archive,storage.file_sha256(archive),tmp_path/('case'+str(len(names))+str(kind)))


def test_gcs_immutable_publish_download_credentials(tmp_path,monkeypatch):
    fake=FakeClient()
    store=storage.GcsBundles(fake)
    path=tmp_path/'x'; path.write_bytes(b'checkpoint')
    ref=store.publish(path,'gs://bucket/run','run')
    assert store.publish(path,'gs://bucket/run','run')==ref
    store.complete(ref,'gs://bucket/run/completed.json')
    blob=Mock(); blob.download_to_filename.side_effect=lambda p,**k:Path(p).write_bytes(b'checkpoint')
    store.client=Mock(); store.client.bucket.return_value.blob.return_value=blob
    store.download(ref,tmp_path/'out')
    with pytest.raises(ValueError,match='checksum'):
        store.download({**ref,'sha256':'bad'},tmp_path/'out')
    blob.exists.return_value=False
    assert store.read_json('gs://bucket/missing') is None
    blob.exists.return_value=True
    blob.download_as_bytes.return_value=b'{"status":"completed"}'
    assert store.read_json('gs://bucket/complete')['status']=='completed'
    import google.cloud.storage as gcs
    import google.oauth2.service_account as sa
    monkeypatch.setattr(gcs,'Client',Mock(return_value=fake))
    monkeypatch.delenv('EDUGRAPH_GCS_CREDENTIALS_JSON',raising=False)
    assert storage.GcsBundles.from_environment().client is fake
    monkeypatch.setenv('EDUGRAPH_GCS_CREDENTIALS_JSON',json.dumps({'type':'authorized_user'}))
    with pytest.raises(ValueError,match='personal ADC'):
        storage.GcsBundles.from_environment()
    monkeypatch.setattr(sa.Credentials,'from_service_account_info',Mock(return_value='scoped-credentials'))
    monkeypatch.setenv('EDUGRAPH_GCS_CREDENTIALS_JSON',json.dumps({'type':'service_account','project_id':'project'}))
    assert storage.GcsBundles.from_environment().client is fake


def test_runpod_adapter_suppresses_provider_secrets():
    requests=[]
    def handler(req):
        requests.append(req)
        return httpx.Response(200,json={'id':'pod1','env':{'secret':'hidden'}})
    client=RunpodClient('private',httpx.Client(base_url='https://test',transport=httpx.MockTransport(handler)))
    client.create({'name':'run'}); client.get('pod1'); client.stop('pod1'); client.terminate('pod1'); client.list()
    assert [r.method for r in requests]==['POST','GET','POST','DELETE','GET']
    assert requests[1].url.params['includeMachine']=='true'
    assert 'env' not in public_status(client.get('pod1'))
    with pytest.raises(ValueError): client.get('../bad')
    client.client=Mock(); client.client.request.return_value=httpx.Response(403,text='secret')
    with pytest.raises(RuntimeError,match='body suppressed'): client.list()
    client.client.request.side_effect=httpx.ConnectError('secret')
    with pytest.raises(RuntimeError,match='inspect Pod state'): client.list()
    client.client.request.side_effect=None
    client.client.request.return_value=httpx.Response(204)
    assert client.call('DELETE','/pods/pod1')=={}


def test_wandb_records_only_explicit_metrics_and_gcs_reference(prepared,monkeypatch):
    sdk=Mock(); sdk.init.return_value.id='run1'
    sdk.Artifact.return_value=MagicMock()
    tracker=tracking.WandbTracker(prepared.manifest,sdk=sdk)
    assert sdk.init.call_args.kwargs['entity']=='edugraph-io'
    assert sdk.init.call_args.kwargs['resume']=='never'
    tracker.log({'validation':{'f1':0.8,'labels':['Addition']},'secret':'ignored'},5)
    sdk.init.return_value.log.assert_called_once_with({'validation/f1':0.8,'optimizer_step':5})
    tracker.checkpoint({'uri':'gs://bucket/snapshot','sha256':'a'*64})
    tracker.checkpoint({'uri':'gs://bucket/model','sha256':'b'*64},final=True)
    sdk.Artifact.return_value.add_reference.assert_not_called()
    written=sdk.Artifact.return_value.new_file.return_value.__enter__.return_value.write.call_args.args[0]
    assert json.loads(written)=={'uri':'gs://bucket/model','sha256':'b'*64}
    tracker.finish(True)
    monkeypatch.setitem(__import__('sys').modules,'wandb',sdk)
    tracking.WandbTracker(prepared.manifest,resume=True)
    assert sdk.init.call_args.kwargs['resume']=='allow'


def test_real_wandb_reference_artifact_needs_no_google_credentials(tmp_path,monkeypatch):
    import google.auth
    import wandb
    monkeypatch.setenv('WANDB_DATA_DIR',str(tmp_path/'wandb'))
    monkeypatch.setattr(google.auth,'default',Mock(side_effect=AssertionError('must not initialize ADC')))
    tracker=object.__new__(tracking.WandbTracker)
    tracker.sdk=wandb
    tracker.run=Mock(id='offline-reference-check')
    reference={'run_id':'smoke','uri':'gs://bucket/run/checkpoint.tar','sha256':'a'*64}
    tracker.checkpoint(reference)
    artifact=tracker.run.log_artifact.call_args.args[0]
    assert list(artifact.manifest.entries)==['gcs-reference.json']
    entry=artifact.manifest.entries['gcs-reference.json']
    assert json.loads(Path(entry.local_path).read_text())==reference
    assert entry.size < 1024


def test_pod_scoped_client_uses_graphql_for_read_and_lifecycle():
    requests=[]
    def handler(request):
        payload=json.loads(request.content)
        requests.append(payload)
        assert request.url.path=='/graphql'
        assert payload['variables']=={'id':'pod1'}
        query=payload['query']
        field='podTerminate' if 'podTerminate' in query else 'podStop' if 'podStop' in query else 'pod'
        return httpx.Response(200,json={'data':{field:None if field=='podTerminate' else
            {'id':'pod1','costPerHr':0.49,'machine':{'secureCloud':True}}}})
    client=RunpodPodClient('pod-scoped-key',httpx.Client(base_url='https://test',transport=httpx.MockTransport(handler)))
    assert client.get('pod1')['machine']['secureCloud'] is True
    client.stop('pod1')
    client.terminate('pod1')
    assert len(requests)==3
    with pytest.raises(ValueError):client.get('../other')
    with pytest.raises(RunpodError,match='details'):
        client.graphql=lambda *args:None
        client.get('pod1')


@pytest.mark.parametrize('body', [{'errors':[{'message':'secret-value'}]}, {'data':None}, {'data':{}}])
def test_graphql_errors_do_not_expose_response_values(body):
    client=RunpodPodClient('pod-scoped-key',httpx.Client(base_url='https://test',
        transport=httpx.MockTransport(lambda request:httpx.Response(200,json=body))))
    with pytest.raises(RunpodError,match='body suppressed') as error:client.get('pod1')
    assert 'secret-value' not in str(error.value)
