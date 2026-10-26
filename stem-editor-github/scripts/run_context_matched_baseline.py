#!/usr/bin/env python3
"""Matched-context pilot: five scenarios x two ceilings x three repetitions.

Run inside backend/ after copying this file into scripts/:
  python scripts/run_context_matched_baseline.py --dry-run
  python scripts/run_context_matched_baseline.py

No database records are changed. Calls the existing backend diagnostic code,
planner and configured LLM service with a common realization prompt. Uses synthetic data.
Successful calls are checkpointed; --resume avoids repeating them.
"""
import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


SCENARIOS = ['S02', 'S08', 'S12', 'S19', 'S22']
METHODS = ['context_matched_single_llm', 'externally_planned']


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def matched_prompt(context, decision=None):
    common = '''You are an adaptive AI programming tutor. Reply in English.

Respect the teacher ceiling. Available strategies and nominal levels are:
challenge=0, socratic=1, scaffolding=2, direct_explanation=3.
At ceiling 0, do not give an answer. At ceiling 1, use guiding questions or
hints without supplying a partial or complete solution. At ceiling 2, you may
give scaffolding and intermediate steps but not a complete solution.
Direct explanation requires ceiling >=3. A complete solution additionally
requires ceiling >=5, confusion >=0.85, mastery <=0.30, at least three recent
occurrences of the current misconception, and help-seeking >=0.60 or
directness >=0.65.

Be concise, accurate and supportive. Output only the tutoring message.

SHARED LEARNER/TASK CONTEXT:
''' + json.dumps(context, ensure_ascii=False, indent=2)
    if decision is None:
        return common + '\n\nSelect the appropriate strategy and assistance level yourself. No external pedagogical decision is provided.'
    return common + '\n\nFollow this external pedagogical decision; do not replace it:\n' + json.dumps(decision, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend-dir', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--data', type=Path)
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--scenario-ids', nargs='+', default=SCENARIOS)
    ap.add_argument('--ceilings', nargs='+', type=int, default=[1, 2])
    ap.add_argument('--expected-model', default='qwen3.7-plus')
    ap.add_argument('--seed', type=int, default=20261013)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.repeats < 1 or any(c not in range(6) for c in args.ceilings):
        ap.error('Positive repeats and ceilings in 0..5 are required.')
    backend = args.backend_dir.resolve()
    data_path = args.data or backend/'experiment_data/controlled_experiment_cases.json'
    pack = json.loads(data_path.read_text(encoding='utf-8'))
    lookup = {x['scenario_id']: x for x in pack['base_scenarios']}
    missing = set(args.scenario_ids)-set(lookup)
    if missing:
        ap.error('Unknown scenario IDs: '+str(sorted(missing)))
    scenarios = [lookup[k] for k in args.scenario_ids]
    config = {'scenario_ids':args.scenario_ids,'ceilings':args.ceilings,'repeats':args.repeats,'seed':args.seed,'expected_model':args.expected_model,'temperature':.3,'max_tokens':350,'mastery':.5,'history':'empty','profile':'neutral','data_sha256':digest(pack),'pilot_scope':'Five predefined scenarios, one per task; not the complete 25-scenario benchmark.'}
    calls = len(scenarios)*len(args.ceilings)*args.repeats*2
    if args.dry_run:
        print(json.dumps({'planned_generation_requests':calls,'diagnostic_snapshots':len(scenarios),'note':'Diagnosis may add an LLM fallback request. No API or database calls executed.','configuration':config},indent=2))
        return
    sys.path.insert(0,str(backend))
    from api.tutor import diagnose_submission
    from agents.affective_agent import AffectiveAgent
    from fuzzy.confusion_system import calculate_confusion
    from pedagogy.planner import PedagogyPlanner
    from services.llm_service import llm_service
    if not llm_service.available:
        raise RuntimeError('Existing backend API configuration is unavailable. Do not paste the key into this script.')
    if llm_service.model != args.expected_model:
        raise RuntimeError(f'Configured model {llm_service.model!r} differs from expected model {args.expected_model!r}. Verify backend configuration first.')
    config['provider']=llm_service.provider
    config['api_base_url']=llm_service.base_url
    config['baseline_prompt_sha256']=hashlib.sha256(matched_prompt({}).encode()).hexdigest()
    config['source_files_sha256']={str(p.relative_to(backend)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [backend/'api/tutor.py',backend/'agents/tutor_executor.py',backend/'services/llm_service.py',backend/'pedagogy/planner.py',backend/'pedagogy/pomdp.py',backend/'pedagogy/belief_state.py',backend/'pedagogy/reward.py',backend/'fuzzy/confusion_system.py']}
    output = args.output_dir or backend/'experiment_results/context_matched_pilot'
    output.mkdir(parents=True,exist_ok=True)
    cfg_path=output/'configuration.json'
    if cfg_path.exists():
        if not args.resume:
            raise RuntimeError('Output exists. Use a new --output-dir or --resume; no calls were made.')
        if json.loads(cfg_path.read_text())!=config:
            raise RuntimeError('Resume configuration/source hashes differ; use a new output directory.')
    else:
        cfg_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    results_path=output/'matched_results.jsonl'
    prior=[json.loads(line) for line in results_path.read_text(encoding='utf-8').splitlines()] if results_path.exists() else []
    done={(x['scenario_id'],x['ceiling'],x['repeat'],x['method']) for x in prior if x['success']}
    snapshots_path=output/'shared_snapshots.json'
    snapshots=json.loads(snapshots_path.read_text()) if snapshots_path.exists() else {}
    task_lookup={x['task_key']:x for x in pack['tasks']}
    planner=PedagogyPlanner();affect=AffectiveAgent();rng=random.Random(args.seed)
    concepts={'sum_even':['loops','conditionals','modulo_operator','accumulation'],'count_vowels':['loops','strings','membership'],'find_max':['loops','comparison','accumulation'],'palindrome':['strings','comparison'],'safe_dict_get':['dictionaries','safe_access']}
    for scenario in scenarios:
        key=scenario['scenario_id'];task=task_lookup[scenario['task_key']]
        obj=SimpleNamespace(id=key,title=task['title'],description=task['learning_objective'],learning_objective=task['learning_objective'],function_name=task['function_name'],hidden_tests=task['hidden_tests'],reference_solution=task['reference_solution'],language='Python',difficulty='introductory',question_type='coding_task',learning_objectives=[task['learning_objective']],concepts=concepts[scenario['task_key']],constraints=[],visible_examples=[])
        if key not in snapshots:
            diagnosis, execution_result=diagnose_submission(scenario['student_code'],obj)
            snapshots[key]={'diagnosis':diagnosis,'execution_result':execution_result,'confusion':calculate_confusion(scenario['pause_time'],scenario['error_count'])}
            snapshots_path.write_text(json.dumps(snapshots,indent=2,ensure_ascii=False),encoding='utf-8')
        snapshot=snapshots[key];c=snapshot['confusion'];diagnosis=snapshot['diagnosis']
        dynamic={'learning_preference':'guided','challenge_tolerance':.5,'directness_preference':.5,'help_seeking_tendency':.5,'persistence':.5,'reflective_tendency':.5,'profile_confidence':0.}
        stable={'personality':'balanced','learning_preference':'guided','challenge_tolerance':.5}
        profile={'stable_profile':stable,'dynamic_profile':dynamic,'current_state':{'confusion_level':c,'engagement_level':max(0.,min(1.,1-.65*c-.04*min(scenario['error_count'],5)))}}
        mastery={'concept':diagnosis.get('concept') or 'general_python','score':.5};memory={'recent_episodes':[]}
        persona=affect.choose_persona(confusion_level=c,personality='balanced',learning_preference='guided',dynamic_profile=dynamic)
        for ceiling in args.ceilings:
            assignment={k:getattr(obj,k) for k in ['id','title','description','learning_objective','function_name','language','difficulty','question_type','learning_objectives','concepts','constraints','visible_examples']};assignment['max_hint_level']=ceiling
            context={'assignment':assignment,'diagnosis':diagnosis,'student_profile':profile,'mastery':mastery,'memory':memory,'persona':persona,'observation':{'pause_time':scenario['pause_time'],'error_count':scenario['error_count'],'confusion_level':c},'student_code':scenario['student_code']}
            context_hash=digest(context)
            for repeat in range(1,args.repeats+1):
                order=METHODS.copy();rng.shuffle(order)
                for method in order:
                    row_key=(key,ceiling,repeat,method)
                    if row_key in done:
                        continue
                    start=time.perf_counter();plan=None
                    if method=='context_matched_single_llm':
                        prompt=matched_prompt(context)
                    else:
                        plan=planner.create_plan(diagnosis=diagnosis,confusion_level=c,learning_preference='guided',teacher_max_hint_level=ceiling,dynamic_profile=dynamic,mastery_score=.5,recent_episodes=[])
                        prompt=matched_prompt(context,plan)
                    construction_ms=1000*(time.perf_counter()-start)
                    prompt_dir=output/'prompts';prompt_dir.mkdir(exist_ok=True)
                    (prompt_dir/f'{key}_H{ceiling}_R{repeat}_{method}.txt').write_text(prompt,encoding='utf-8')
                    call_start=time.perf_counter()
                    result=llm_service.generate_with_metadata(prompt,temperature=.3,max_tokens=350)
                    row={'scenario_id':key,'task_key':scenario['task_key'],'ceiling':ceiling,'repeat':repeat,'method':method,'shared_context_sha256':context_hash,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'success':bool(result),'response':result.get('text','') if result else '', 'plan':plan,'model':result.get('model') if result else None,'prompt_tokens':result.get('prompt_tokens') if result else None,'completion_tokens':result.get('completion_tokens') if result else None,'generation_tokens':result.get('total_tokens') if result else None,'llm_request_latency_ms':result.get('latency_ms') if result else None,'method_stage_latency_ms':construction_ms+1000*(time.perf_counter()-call_start),'recorded_at':datetime.now(timezone.utc).isoformat()}
                    with results_path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    print(f'{key} H={ceiling} repeat={repeat} {method}: '+('OK' if result else 'FAILED'),flush=True)
                    if not result:
                        raise RuntimeError('Call failed and was checkpointed. Fix configuration/network, then --resume. A failed request may still incur cost.')
    # Export only the latest successful attempt for each unique call.
    all_rows=[json.loads(line) for line in results_path.read_text(encoding='utf-8').splitlines()]
    completed={}
    for row in all_rows:
        if row['success']:completed[(row['scenario_id'],row['ceiling'],row['repeat'],row['method'])]=row
    for key,ceiling,repeat in {(x[0],x[1],x[2]) for x in completed}:
        pair=[completed[(key,ceiling,repeat,m)] for m in METHODS]
        assert pair[0]['shared_context_sha256']==pair[1]['shared_context_sha256']
    rows=list(completed.values())
    fields=[x for x in rows[0] if x!='plan']
    with (output/'matched_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    blind=rows.copy();random.Random(args.seed+1).shuffle(blind)
    rating_rows=[];key_rows=[]
    for i,row in enumerate(blind,1):
        rating_id=f'B{i:03d}'
        rating_rows.append({'rating_id':rating_id,'scenario_id':row['scenario_id'],'teacher_ceiling':row['ceiling'],'tutor_response':row['response'],'correctness_1_to_5':'','complete_answer_leakage_0_or_1':'','ceiling_compliance_0_or_1':'','pedagogical_helpfulness_1_to_5':'','rater_id':'','notes':''})
        key_rows.append({'rating_id':rating_id,'scenario_id':row['scenario_id'],'repeat':row['repeat'],'ceiling':row['ceiling'],'method':row['method']})
    for name,data in [('blind_ratings_template.csv',rating_rows),('blind_method_key.csv',key_rows)]:
        with (output/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    summary_rows=[]
    for method in METHODS:
        selected=[row for row in rows if row['method']==method]
        result={'method':method,'n':len(selected)}
        for metric in ['prompt_tokens','completion_tokens','generation_tokens','llm_request_latency_ms','method_stage_latency_ms']:
            values=[float(row[metric]) for row in selected if row[metric] is not None]
            result['mean_'+metric]=statistics.mean(values) if values else None
            result['sd_'+metric]=statistics.stdev(values) if len(values)>1 else None
        summary_rows.append(result)
    with (output/'method_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary_rows[0]));w.writeheader();w.writerows(summary_rows)
    (output/'run_summary.json').write_text(json.dumps({'successful_generation_requests':len(rows),'pair_count':len(rows)//2,'unique_scenarios':len(scenarios),'model':llm_service.model,'rating_status':'not evaluated','method_summary':summary_rows,'limitations':'Pilot scenes and synthetic states only. Both use a common realization prompt and identical raw context. The external decision is the treatment difference; token lengths are not forced equal. This is a controlled planner comparison, not a rerun of the original deployed-executor prompt.'},indent=2))
    print('Finished. Responses are archived; human-rating cells are intentionally blank.')


if __name__=='__main__':
    main()
