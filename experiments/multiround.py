import json
import shutil
import pdb
from itertools import islice
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model, apply_memit2model_modified, upd_matrix_match_shape
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *
from copy import deepcopy
from experiments.util_TIES import *

from tqdm import tqdm

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def main(
        merging_method: str,
        alg_name: str,
        model_name: Union[str, Tuple],
        hparams_fname: str,
        ds_name: str,
        dataset_size_limit: int,
        continue_from_run: str,
        skip_generation_tests: bool,
        generation_test_interval: int,
        conserve_memory: bool,
        device: str,
        dir_name: str,
        # num_edits: int = 1,
        num_edits_pC_pR: int = 1,
        num_clients: int = 10,
        num_rounds: int = 1,
        use_cache: bool = False,
):
    device = "cuda"  # cuda / cpu
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # the whole number of editing in single rounds
    num_edits = num_edits_pC_pR * num_clients * 1

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
            continue_from_run is None
            or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # Iterate through dataset
    Rec_chunks = chunks(ds, 2 * num_edits)
    for record_chunks in Rec_chunks:
        case_result_template = str(run_dir / "{}_edits-case_{}.json")
        target_chunks = deepcopy(record_chunks[0:num_edits])
        noise_chunks = record_chunks[num_edits:]
        Rec_chunks = record_chunks[0:num_edits]
        # print(noise_chunks)
        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                    case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=(
                "cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(
            alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

        flag = 0

        for ___ in tqdm(range(1 + num_edits)):
            if flag:
                record_chunks = [noise_chunks[flag - 1]]
            # compute update
            if "global" in merging_method:
                start = time()
                delta_50, covs, __, ____ = apply_memit2model_modified(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **
                            record["requested_rewrite"]}
                        for record in record_chunks
                    ],
                    hparams,
                    device,
                    copy=False,
                    return_orig_weights=True,
                    **args_conserve_memory,
                    **etc_args,
                )

            else:
                slide_len = int(num_edits / 10)
                # records = []
                deltas = []
                all_kkts = []
                for i in range(10):
                    start = time()
                    records = record_chunks[slide_len * i:slide_len * (i + 1)]
                    delta_nx10, covs, kkts, ____ = apply_memit2model_modified(
                        model,
                        tok,
                        [
                            {"case_id": record["case_id"],
                                **record["requested_rewrite"]}
                            for record in records
                        ],
                        hparams,
                        device,
                        copy=False,
                        return_orig_weights=True,
                        **args_conserve_memory,
                        **etc_args,
                    )
                    all_kkts.append(kkts)
                    deltas.append(delta_nx10)
                covs = covs

            with torch.no_grad():
                # 10N x 1
                if "global" in merging_method:
                    upd_matrix_10Nx1 = None
                    upd_matrix_10Nx1_dict = {}
                    for w_name, (key_mat, val_mat) in delta_50.items():
                        key_mat, val_mat = key_mat.to(
                            "cuda"), val_mat.to("cuda")
                        upd_matrix_10Nx1 = val_mat @ key_mat.T
                        upd_matrix_10Nx1_dict[w_name] = upd_matrix_10Nx1

                        # generate 50x1 model and record original weights
                        w1 = nethook.get_parameter(model, w_name)
                        upd_matrix_10Nx1 = upd_matrix_match_shape(
                            upd_matrix_10Nx1, w1.shape)
                        # if w_name not in weights_copy1:
                        #     weights_copy1[w_name] = w1.detach().clone()
                        w1[...] += upd_matrix_10Nx1.float()

                # N x 10
                else:
                    upd_matrix_dict_list = []
                    # all_kkts = []
                    for delta in deltas:
                        upd_matrix_dict = {}
                        kkts = {}
                        for w_name, (key_mat, val_mat) in delta.items():
                            key_mat, val_mat = key_mat.to(
                                "cuda"), val_mat.to("cuda")
                            upd_matrix = val_mat @ key_mat.T
                            upd_matrix_dict[w_name] = upd_matrix
                            # kkts[w_name] = key_mat @ key_mat.T
                        upd_matrix_dict_list.append(upd_matrix_dict)
                        all_kkts.append(kkts)

                    del deltas
                    torch.cuda.empty_cache()

                    for w_name, delta_demo in upd_matrix_dict_list[0].items():
                        delta_demo = delta_demo.to(device)
                        delta_Nx10 = torch.zeros(delta_demo.shape).to(device)
                        # Using Task-Algorithm (TA)
                        if "task" in merging_method:
                            print("Using Task-Vector Merge")
                            for i in range(10):
                                delta_Nx10 = delta_Nx10 + \
                                    upd_matrix_dict_list[i][w_name].to(device)
                                upd_matrix_dict_list[i][w_name].cpu()
                                torch.cuda.empty_cache()
                            # delta_Nx10 = delta_Nx10
                        elif "average" in merging_method:
                            # Using Simple-Average (SA)
                            print("Using Average Merge")
                            for i in range(10):
                                delta_Nx10 = delta_Nx10 + \
                                    upd_matrix_dict_list[i][w_name].to(device)
                                upd_matrix_dict_list[i][w_name].cpu()
                                torch.cuda.empty_cache()
                            delta_Nx10 = delta_Nx10 / 10

                        elif "Nondestructive" in merging_method:      # Using our CollabEdit
                            print("Using Nondestructive Merge")
                            cov = covs[w_name].to(device)
                            A = deepcopy(cov)
                            for i in range(10):
                                all_kkt = all_kkts[i][w_name].to(device)
                                delta_Nx10 = delta_Nx10 + upd_matrix_dict_list[i][w_name].to(device) @ (
                                    all_kkt + cov)
                                A = A + all_kkt
                                upd_matrix_dict_list[i][w_name].cpu()
                                all_kkt.cpu()
                                covs[w_name].cpu()
                                torch.cuda.empty_cache()
                            delta_Nx10 = delta_Nx10 @ A.inverse()
                            A.cpu()
                            torch.cuda.empty_cache()
                        else:
                            print(
                                "Is not TA, average, nondestructive merging method")
                            break
                        # generate 5x10 model and record original weights
                        w2 = nethook.get_parameter(model2, w_name)
                        delta_Nx10 = upd_matrix_match_shape(
                            delta_Nx10, w2.shape)
                        w2[...] += delta_Nx10.float()

                        # clear GPU memory
                        delta_demo = delta_demo.cpu()
                        delta_Nx10 = delta_Nx10.cpu()
                        del delta_demo, delta_Nx10
                        torch.cuda.empty_cache()

                    if "ties" in merging_method:  # Using Ties-Merging
                        print("Using Ties-merging")
                        # flat_ft = torch.vstack([])
                        # pdb.set_trace()
                        reference_state_dict = upd_matrix_dict_list[0]
                        # reference_state = reference_state_dict[]
                        for w_name, _ in reference_state_dict.items():
                            reset_thresh = eval("20")
                            # flat_delta = torch.vstack([state_dict_to_vector(deita[w_name], []) for deita in upd_matrix_dict_list])
                            flat_delta = torch.vstack(
                                [torch.nn.utils.parameters_to_vector([deita[w_name].reshape(-1)]) for deita in
                                 upd_matrix_dict_list])
                            flat_delta = flat_delta.to(device)
                            updated_checks, *_ = topk_values_mask(
                                flat_delta, K=reset_thresh, return_mask=False
                            )

                            # reference_state_dict = upd_matrix_dict_list[0]
                            # flat_delta.cpu()
                            del flat_delta
                            torch.cuda.empty_cache()
                            # pdb.set_trace()
                            print("1. mem allocated in MB:",
                                  torch.cuda.memory_allocated() / 1024 ** 2)
                            resolve_method = "mass"
                            final_signs = resolve_sign(
                                updated_checks, resolve_method)
                            merge_func = "mean"
                            torch.cuda.empty_cache()
                            print("2. mem allocated in MB:",
                                  torch.cuda.memory_allocated() / 1024 ** 2)
                            merged_tv = disjoint_merge(
                                updated_checks, merge_func, final_signs)
                            lam = 0.4
                            merged_check_delta = lam * merged_tv
                            print("merged_check_delta shape:",
                                  merged_check_delta.shape)
                            # delta_Nx10_Wname = vector_to_state_dict(
                            #     merged_check_delta, reference_state_dict, remove_keys=[]
                            # )
                            torch.nn.utils.vector_to_parameters(merged_check_delta,
                                                                reference_state_dict[w_name])

                            # for w_name, _ in reference_state_dict.items():
                            w2 = nethook.get_parameter(model2, w_name)
                            w2 = w2.to(device)
                            delta_Nx10_Wname = upd_matrix_match_shape(
                                torch.Tensor(reference_state_dict[w_name]), w2.shape)
                            delta_Nx10_Wname = delta_Nx10_Wname.to(device)
                            w2[...] += delta_Nx10_Wname
                        del upd_matrix_dict_list
            flag = flag + 1
        print("models generated")

        # Evaluate new model
        start = time()
        gen_test_vars = [snips, vec]
        record_chunks = target_chunks
        for record in record_chunks:
            metrics1 = {}
            metrics2 = {}
            out_file = Path(case_result_template.format(
                num_edits, record["case_id"]))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue
            if "global" in merging_method:
                metrics1 = {
                    "case_id": record["case_id"],
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": record["requested_rewrite"],
                    "post": ds_eval_method(
                        model,
                        tok,
                        record,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),  # Only test generation every generation_test_interval cases
                    ),
                }
            else:
                metrics2 = {
                    "case_id": record["case_id"],
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": record["requested_rewrite"],
                    "post": ds_eval_method(
                        model,
                        tok,
                        record,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),  # Only test generation every generation_test_interval cases
                    ),
                }
            metrics = {
                "global_edit": metrics1,
                "collaborative_edit": metrics2
            }

            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)

        break  # Just edit 'num_edits' records (in a single round)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i: i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME", "FT", "MEND"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
             "where a new run_id is generated on each run. "
             "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large",
                 "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
             "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
             "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits_perClient_inSingleRound",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously for each client in single round.",
    )
    parser.add_argument(
        "--num_editRounds",
        type=int,
        default=1,
        help="Number of editing rounds.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--merging_method",
        type=str,
        default="global",
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Using GPU or CPU",
    )
    
    parser.add_argument(
        "--num_clients",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.merging_method,
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        args.device,
        dir_name=args.alg_name,
        # num_edits=args.num_edits,
        num_edits_pC_pR=args.num_edits_perClient_inSingleRound,
        num_clients=args.num_clients,
        num_rounds=args.num_editRounds,
        use_cache=args.use_cache,
    )