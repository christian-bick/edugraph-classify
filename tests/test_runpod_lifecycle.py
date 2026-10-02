from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from edugraph_classify import cli, runpod_cli, runpod_worker, checkpoint_storage as storage
from edugraph_classify.learning_data import write_json
from test_runpod import Store, prepared, recipe


def test_launch_journal_replay_guard_and_price_guard(prepared,tmp_path):
    client=Mock()
    client.list.return_value=[]
    client.create.return_value={'id':'pod1','name':prepared.config['run_id'],'costPerHr':0.33,'env':{'key':'private'}}
    client.get.return_value={**client.create.return_value,'costPerHr':'0.53','machine':{'secureCloud':True}}
    request={'name':prepared.config['run_id']}
    result=runpod_cli.launch_pod(prepared.manifest,request,prepared.run,client)
    assert result['status']=='created' and 'env' not in result['pod']
    with pytest.raises(FileExistsError): runpod_cli.launch_pod(prepared.manifest,request,prepared.run,client)
    assert client.create.call_count==1
    client.list.return_value=[{'name':prepared.config['run_id']}]
    with pytest.raises(ValueError,match='already exists'): runpod_cli.launch_pod(prepared.manifest,request,prepared.run,client)
    client.list.return_value=[]
    client.get.return_value.update(costPerHr=3)
    other=tmp_path/'expensive'; other.mkdir()
    with pytest.raises(ValueError,match='ceiling'):runpod_cli.launch_pod(prepared.manifest,request,other,client)
    client.stop.assert_called_once_with('pod1')
    assert json.loads((other/'pod-launch.json').read_text())['status']=='stopped_price_guard'


def test_cli_preparation_staging_render_launch_status_stop(prepared,tmp_path,monkeypatch,capsys):
    parser=cli.build_parser()
    args=parser.parse_args(['runpod','check-config'])
    assert runpod_cli.run_runpod_command(args,None)[1]['status']=='valid'
    prepare=parser.parse_args(['runpod','prepare','--runs-root',str(tmp_path)])
    with pytest.raises(ValueError,match='clean'):runpod_cli.run_runpod_command(prepare,None)
    monkeypatch.setattr(runpod_cli,'prepare_runpod',lambda *a:prepared.manifest)
    assert runpod_cli.run_runpod_command(prepare,'a'*40)[1]['status']=='prepared'
    store=Store(tmp_path/'store')
    monkeypatch.setattr(storage.GcsBundles,'from_environment',lambda:store)
    monkeypatch.setattr(runpod_cli,'load_local_environment',lambda p:None)
    stage=parser.parse_args(['runpod','stage','--manifest',str(prepared.path),'--confirm-run-id',prepared.config['run_id']])
    with pytest.raises(ValueError,match='clean'):runpod_cli.run_runpod_command(stage,None)
    stage.confirm_run_id='wrong'
    with pytest.raises(ValueError,match='confirmation'):runpod_cli.run_runpod_command(stage,'a'*40)
    stage.confirm_run_id=prepared.config['run_id']
    ref=runpod_cli.run_runpod_command(stage,'a'*40)[1]
    image='ghcr.io/owner/trainer@sha256:'+'b'*64
    render=parser.parse_args(['runpod','render','--manifest',str(prepared.path),'--image',image])
    assert runpod_cli.run_runpod_command(render,'a'*40)[1]['request']['cloudType']=='SECURE'
    resume=tmp_path/'resume.json'; write_json(resume,ref)
    render.resume_reference=str(resume)
    render.attempt='recover-1'
    assert 'EDUGRAPH_RESUME_URI' in runpod_cli.run_runpod_command(render,'a'*40)[1]['request']['env']
    provider=Mock(); provider.list.return_value=[]
    provider.create.return_value={'id':'pod1','name':prepared.config['run_id'],'costPerHr':0.33}
    provider.get.return_value={**provider.create.return_value,'machine':{'secureCloud':True}}
    monkeypatch.setattr(runpod_cli,'RunpodClient',lambda key:provider)
    monkeypatch.setenv('RUNPOD_API_KEY','not-a-real-key')
    launch=parser.parse_args(['runpod','launch','--manifest',str(prepared.path),'--image',image,'--confirm-run-id',prepared.config['run_id']])
    assert runpod_cli.run_runpod_command(launch,'a'*40)[1]['status']=='created'
    record=str(prepared.run/'pod-launch.json')
    status=parser.parse_args(['runpod','status','--launch-record',record])
    assert runpod_cli.run_runpod_command(status,None)[1]['id']=='pod1'
    stop=parser.parse_args(['runpod','stop','--launch-record',record,'--confirm-run-id','wrong'])
    with pytest.raises(ValueError,match='confirmation'):runpod_cli.run_runpod_command(stop,None)
    stop.confirm_run_id=prepared.config['run_id']
    runpod_cli.run_runpod_command(stop,None)
    provider.stop.assert_called_once_with('pod1')
    assert cli.main(['runpod','check-config'])==0
    monkeypatch.setattr(cli,'_code_commit',lambda p:'a'*40)
    monkeypatch.setattr(cli,'run_runpod_command',lambda *a:(0,{'status':'prepared'}))
    assert cli.main(['runpod','prepare'])==0


def test_resume_is_checked_before_paid_creation(prepared,tmp_path,monkeypatch):
    from test_runpod import execute
    store=Store(tmp_path/'store')
    paused,_=execute(prepared,tmp_path/'partial',store,stop_requested=lambda:True)
    state=runpod_cli.check_resume_reference(prepared.manifest,paused['checkpoint'],prepared.run,store)
    assert state['step']==1
    monkeypatch.setattr(storage.GcsBundles,'from_environment',lambda:store)
    monkeypatch.setattr(runpod_cli,'load_local_environment',lambda p:None)
    monkeypatch.setenv('RUNPOD_API_KEY','fake')
    parser=cli.build_parser()
    stage=parser.parse_args(['runpod','stage','--manifest',str(prepared.path),'--confirm-run-id',prepared.config['run_id']])
    runpod_cli.run_runpod_command(stage,'a'*40)
    reference=tmp_path/'resume.json'; write_json(reference,paused['checkpoint'])
    client=Mock()
    client.list.return_value=[{'name':prepared.config['run_id'],'desiredStatus':'EXITED'}]
    client.create.return_value={'id':'newpod','costPerHr':0.33}
    client.get.return_value={**client.create.return_value,'machine':{'secureCloud':True}}
    monkeypatch.setattr(runpod_cli,'RunpodClient',lambda key:client)
    args=parser.parse_args(['runpod','launch','--manifest',str(prepared.path),'--confirm-run-id',prepared.config['run_id'],
        '--image','ghcr.io/owner/trainer@sha256:'+'b'*64,'--resume-reference',str(reference),'--attempt','recover-1'])
    assert runpod_cli.run_runpod_command(args,'a'*40)[1]['status']=='created'
    assert (prepared.run/'pod-launch-recover-1.json').exists()
    client.list.return_value=[{'name':prepared.config['run_id']+'--recover-1','desiredStatus':'RUNNING'}]
    with pytest.raises(ValueError,match='already exists'):
        runpod_cli.launch_pod(prepared.manifest,{'name':prepared.config['run_id']+'--recover-2'},prepared.run,client,attempt='recover-2')
    full,_=execute(prepared,tmp_path/'full',store)
    with pytest.raises(ValueError,match='already has a completed'):
        runpod_cli.check_resume_reference(prepared.manifest,full['continuation'],prepared.run,store)
    with pytest.raises(ValueError,match='already has a completed'):
        runpod_cli.run_runpod_command(args,'a'*40)


def test_prepare_downloads_only_public_pinned_inputs(tmp_path,monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub,'snapshot_download',Mock())
    response=Mock(); response.content=b'public'
    client=Mock(); client.__enter__=Mock(return_value=client); client.__exit__=Mock(return_value=False)
    client.get.return_value=response
    monkeypatch.setattr(runpod_cli.httpx,'Client',lambda **k:client)
    def prepare(config, root, run, processor, commit, fetch, capabilities, **kwargs):
        assert fetch('https://public.example/data')==b'public'
        assert kwargs['runtime_packages']==runpod_cli.RUNTIME_PACKAGES
        return {'prepared':True}
    monkeypatch.setattr(runpod_cli,'prepare_learning',prepare)
    assert runpod_cli.prepare_runpod(recipe(),tmp_path,'a'*40)=={'prepared':True}
    kwargs=huggingface_hub.snapshot_download.call_args.kwargs
    assert kwargs['token'] is False and len(kwargs['revision'])==40


@pytest.fixture
def worker(prepared,tmp_path,monkeypatch):
    root=tmp_path/'worker'; root.mkdir()
    store=Store(tmp_path/'store')
    archive=tmp_path/'bundle.tar'
    storage.pack_files(prepared.run,['manifest.json',*prepared.manifest['files_sha256']],archive)
    ref=store.publish(archive,'gs://edugraph-classify/runpod/inputs',prepared.config['run_id'])
    for name,value in {'RUNPOD_POD_ID':'pod1','RUNPOD_API_KEY':'pod-scoped-fake','WANDB_API_KEY':'fake-wandb',
        'EDUGRAPH_MAX_HOURS':'12','EDUGRAPH_EXPECTED_COMMIT':'a'*40,'EDUGRAPH_CODE_COMMIT':'a'*40,
        'EDUGRAPH_BUNDLE_URI':ref['uri'],'EDUGRAPH_BUNDLE_SHA256':ref['sha256'],
        'EDUGRAPH_RUN_ID':prepared.config['run_id']}.items():monkeypatch.setenv(name,value)
    monkeypatch.delenv('EDUGRAPH_RESUME_URI',raising=False)
    monkeypatch.setattr(runpod_worker,'file_sha256',lambda p:prepared.manifest['uv_lock_sha256'])
    monkeypatch.setattr(runpod_worker.signal,'signal',Mock())
    client=Mock(); client.get.return_value={'costPerHr':'0.53','machine':{'secureCloud':True}}
    timer=Mock()
    execute=Mock(return_value={'status':'completed','model':ref})
    kwargs=dict(store_factory=lambda:store,client_factory=lambda key:client,
                timer_factory=Mock(return_value=timer),execute=execute)
    return NS(root=root,store=store,ref=ref,client=client,timer=timer,execute=execute,kwargs=kwargs)


def test_worker_stops_after_failure_or_pause_and_terminates_after_durable_success(worker,monkeypatch):
    result=runpod_worker.run_worker(worker.root,**worker.kwargs)
    assert result['status']=='completed'
    worker.client.terminate.assert_called_once_with('pod1')
    worker.timer.cancel.assert_called_once()
    worker.kwargs['timer_factory'].call_args.args[1]()
    worker.client.stop.assert_called_once_with('pod1')
    handler=runpod_worker.signal.signal.call_args.args[1]
    handler(None,None)
    assert worker.execute.call_args.kwargs['stop_requested']()


@pytest.mark.parametrize('failure', ['wandb','commit','identity','lock','price','cloud','training','paused'])
def test_worker_failure_guards(worker,monkeypatch,failure):
    if failure=='wandb':monkeypatch.delenv('WANDB_API_KEY')
    if failure=='commit':monkeypatch.setenv('EDUGRAPH_CODE_COMMIT','wrong')
    if failure=='identity':monkeypatch.setenv('EDUGRAPH_RUN_ID','wrong')
    if failure=='lock':monkeypatch.setattr(runpod_worker,'file_sha256',lambda p:'wrong')
    if failure=='price':worker.client.get.return_value['costPerHr']=5
    if failure=='cloud':worker.client.get.return_value['machine']['secureCloud']=False
    if failure=='training':worker.execute.side_effect=ValueError('failed')
    if failure=='paused':worker.execute.return_value={'status':'paused'}
    if failure=='paused':runpod_worker.run_worker(worker.root,**worker.kwargs)
    else:
        with pytest.raises(ValueError):runpod_worker.run_worker(worker.root,**worker.kwargs)
    worker.client.stop.assert_called_once_with('pod1')
    worker.client.terminate.assert_not_called()


def test_worker_restores_reference_and_main_sanitizes_exceptions(worker,monkeypatch,capsys):
    monkeypatch.setenv('EDUGRAPH_RESUME_URI',worker.ref['uri'])
    monkeypatch.setenv('EDUGRAPH_RESUME_SHA256',worker.ref['sha256'])
    runpod_worker.run_worker(worker.root,**worker.kwargs)
    assert worker.execute.call_args.kwargs['resume']==worker.root/'resume'
    monkeypatch.setattr(runpod_worker,'run_worker',Mock(return_value={}))
    assert runpod_worker.main()==0
    monkeypatch.setattr(runpod_worker,'run_worker',Mock(side_effect=RuntimeError('secret-value')))
    assert runpod_worker.main()==1
    assert 'secret-value' not in capsys.readouterr().out


@pytest.mark.parametrize('machine', [None, {}, {'secureCloud':False}, {'secureCloud':'true'}])
def test_launch_refuses_unverified_secure_allocation(prepared,machine):
    client=Mock()
    client.list.return_value=[]
    client.create.return_value={'id':'pod1'}
    client.get.return_value={'id':'pod1','costPerHr':'0.49','machine':machine}
    with pytest.raises(ValueError,match='Secure Cloud'):
        runpod_cli.launch_pod(prepared.manifest,{'name':prepared.config['run_id']},prepared.run,client)
    client.stop.assert_called_once_with('pod1')
    assert json.loads((prepared.run/'pod-launch.json').read_text())['status']=='stopped_allocation_guard'


def test_launch_stops_if_allocation_cannot_be_read(prepared):
    client=Mock()
    client.list.return_value=[]
    client.create.return_value={'id':'pod1'}
    client.get.side_effect=RuntimeError('provider unavailable')
    with pytest.raises(RuntimeError):
        runpod_cli.launch_pod(prepared.manifest,{'name':prepared.config['run_id']},prepared.run,client)
    client.stop.assert_called_once_with('pod1')


@pytest.mark.parametrize('raw', [None, True, False, '', 'bad', 'NaN', 'Infinity', float('nan'), -1, 0, '0.61', {}])
def test_unknown_or_excessive_rate_rejected(raw):
    from edugraph_classify.providers.runpod import checked_hourly_rate
    with pytest.raises(ValueError,match='ceiling'):
        checked_hourly_rate({'costPerHr':raw},0.6)


@pytest.mark.parametrize('raw', [0.49, '0.49', 0.6, '0.600'])
def test_numeric_and_string_rates(raw):
    from edugraph_classify.providers.runpod import checked_hourly_rate
    assert checked_hourly_rate({'costPerHr':raw},0.6)==float(raw)
