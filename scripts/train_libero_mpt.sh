#!/bin/bash
# train_libero_mpt.sh -- MPT on LIBERO, three desks (Grasp / Place / Articulate).
#
# The MPT recipe of train_mc_mpt.sh (sealed-bid auction on {0,.2,.4,.6,.8}, revenue-share
# settlement, profit - rent as reward, REINFORCE with a per-agent baseline, 4 act : 1 bid
# iterations with the counterfactual preference loss), on a robot-manipulation setting:
#   * the winner's instruction is executed by a frozen pi0.5 on eom-robotics' Stage 2 env server
#     (`python -m eom_robotics.stage2.server --backend libero --slots TBS*GROUP+2`, next to an openpi
#     policy server); reward = newly met BDDL goal conditions;
#   * the desks start from a base VLM (default Qwen3-VL-4B-Instruct): there is no format-SFT
#     checkpoint for these prompts yet, so check the bid/action parse rates first (DRY_RUN, then a
#     short run with LIBERO_CHAIN_TRACE set);
#   * EoM's population economy is available through agent.orchestra.libero_chain.economy.* and a
#     per-agent reward_rent list (see the yaml block); the defaults are MPT's fixed society.
#
# Usage:
#   LIBERO_ENV_URL=http://<host>:31700 BASE_MODEL=/path/to/Qwen3-VL-4B-Instruct bash scripts/train_libero_mpt.sh [hydra overrides...]
# Knobs (env, defaults in brackets):
#   BASE_MODEL [unset -> required]  M_GRASP/M_PLACE/M_ARTICULATE [$BASE_MODEL]  DATA [<repo>/data/libero]
#   NGPUS [4] TBS [8] GROUP [1] MAXSTEPS [12] EXP [libero_mpt] RENT [0.05] GPU_MEM [0.4]
#   SAVE_FREQ [5] KEEP_CKPT [2] LOGGER ['["console"]']
#   CKPT_DIR [$MPT_WORK/checkpoints/libero] LIBERO_ALT_COUNTER [$MPT_WORK/rl/.libero_alt_counter_$EXP]
#   LIBERO_CHAIN_TRACE [$MPT_WORK/rl/libero_chain_trace_$EXP.jsonl]  plus the common.sh variables.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"

LOGGER=${LOGGER:-'["console"]'}
if [ -z "${BASE_MODEL:-}" ]; then
  [ "${DRY_RUN:-0}" = "1" ] && BASE_MODEL=/path/to/Qwen3-VL-4B-Instruct || \
    { echo "set BASE_MODEL to a local VLM checkpoint dir (e.g. Qwen3-VL-4B-Instruct)"; exit 1; }
fi
M_GRASP=${M_GRASP:-$BASE_MODEL}
M_PLACE=${M_PLACE:-$BASE_MODEL}
M_ARTICULATE=${M_ARTICULATE:-$BASE_MODEL}
DATA=${DATA:-$REPO/data/libero}
NGPUS=${NGPUS:-4}
TBS=${TBS:-8}
GROUP=${GROUP:-1}
MAXSTEPS=${MAXSTEPS:-12}
EXP=${EXP:-libero_mpt}
RENT=${RENT:-0.05}
GPU_MEM=${GPU_MEM:-0.4}
SAVE_FREQ=${SAVE_FREQ:-5}
KEEP_CKPT=${KEEP_CKPT:-2}
CKPT_DIR=${CKPT_DIR:-$MPT_WORK/checkpoints/libero}

: "${LIBERO_ENV_URL:?set LIBERO_ENV_URL to the env server (e.g. http://host:31700)}"
case "$LIBERO_ENV_URL" in
  *,*) echo "LIBERO_ENV_URL must name exactly ONE env server for MPT training (counterfactual branches"
       echo "run on that server's spare slots); got: $LIBERO_ENV_URL"; exit 1;;
esac
export LIBERO_ENV_URL

export LIBERO_ALT_COUNTER=${LIBERO_ALT_COUNTER:-${MPT_ALT_COUNTER:-$MPT_WORK/rl/.libero_alt_counter_$EXP}}
export MPT_ALT_COUNTER="$LIBERO_ALT_COUNTER"
export LIBERO_CHAIN_TRACE=${LIBERO_CHAIN_TRACE:-$MPT_WORK/rl/libero_chain_trace_$EXP.jsonl}
[ "${DRY_RUN:-0}" = "1" ] || mkdir -p "$(dirname "$LIBERO_ALT_COUNTER")" "$(dirname "$LIBERO_CHAIN_TRACE")" "$CKPT_DIR"

# Re-synchronise the alternation counter with the checkpoint that resume_mode=auto will load
# (the reasoning is in train_mc_mpt.sh).
if [ "${DRY_RUN:-0}" != "1" ] && [ "${MPT_ALT_SYNC:-1}" = "1" ]; then
  _latest="$CKPT_DIR/$EXP/latest_checkpointed_iteration.txt"
  _ckpt_step=0
  if [ -s "$_latest" ]; then
    _ckpt_step=$(tr -dc '0-9' < "$_latest")
    _ckpt_step=${_ckpt_step:-0}
  fi
  echo "$_ckpt_step" > "$LIBERO_ALT_COUNTER"
fi

[ "${DRY_RUN:-0}" = "1" ] || [ -f "$DATA/train.parquet" ] || \
  { echo "missing $DATA/train.parquet (run scripts/make_libero_data.py)"; exit 1; }

if [ "${DRY_RUN:-0}" != "1" ]; then
  wait_healthy "$LIBERO_ENV_URL" 40 15 || { echo "env server never became healthy at $LIBERO_ENV_URL"; exit 1; }
  HEALTH=$(curl -sf --noproxy '*' --max-time 10 "$LIBERO_ENV_URL/health")
  NEED=$((TBS * GROUP + 2))
  "$PY" - "$HEALTH" "$NEED" <<'EOF' || exit 1
import json, sys
h = json.loads(sys.argv[1]); need = int(sys.argv[2])
slots = int(h.get("slots", 0))
if slots < need:
    sys.exit(f"env server has {slots} slots; MPT needs TBS*GROUP+2 = {need} (mainline batch + 2 branch slots)")
if str(h.get("snapshot")) != "1":
    sys.exit("env server does not report snapshot=1; the counterfactual bid phase needs it")
print(f"env server ok: slots={slots} (need {need}) backend={h.get('backend')}")
EOF
fi

echo "MPT LIBERO: exp=$EXP alt=4 act : 1 bid (CF preference) counter=$LIBERO_ALT_COUNTER"

run_or_echo "$PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_mean \
    algorithm.gamma=1.0 \
    algorithm.group_by_agent_id=True \
    data.train_files="$DATA/train.parquet" \
    data.train_batch_size="$TBS" \
    data.max_prompt_length=3072 \
    data.max_response_length=320 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TBS" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=1 \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    env.env_name=libero \
    env.max_steps="$MAXSTEPS" \
    env.rollout.n="$GROUP" \
    agent.multi_agent=True \
    'agent.agent_ids=["Grasp Desk (libero)","Place Desk (libero)","Articulate Desk (libero)"]' \
    "agent.model_ids=[\"$M_GRASP\",\"$M_PLACE\",\"$M_ARTICULATE\"]" \
    agent.model_sharing=False \
    agent.orchestra_type=libero_chain \
    +agent.mixed_modality=True \
    agent.orchestra.libero_chain.reward_rent="$RENT" \
    agent.orchestra.libero_chain.profit_as_reward=True \
    agent.orchestra.libero_chain.alt_act_iters=4 \
    agent.orchestra.libero_chain.alt_bid_iters=1 \
    agent.orchestra.libero_chain.cf_preference=True \
    actor_rollout_ref.actor.bid_pref_enable=True \
    actor_rollout_ref.actor.bid_pref_scope=answer \
    reward_model.reward_manager=episode \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.project_name=mpt-libero \
    trainer.experiment_name="$EXP" \
    trainer.logger="$LOGGER" \
    trainer.default_local_dir="$CKPT_DIR/$EXP" \
    trainer.resume_mode=auto \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.max_actor_ckpt_to_keep="$KEEP_CKPT" \
    trainer.max_critic_ckpt_to_keep="$KEEP_CKPT" \
    trainer.total_epochs=100 \
    "$@"
