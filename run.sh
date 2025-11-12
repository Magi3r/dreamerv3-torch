game=$1
cuda_device=$2

if [ -z "$game" ]; then
  echo "Usage: ./run.sh <atari_game_name> [cuda_device]"
  echo "Example: ./run.sh pong 0"
  exit 1
fi

if [ -z "$cuda_device" ]; then
  cuda_device=0
fi

# tmux new -s dreamer_$game "echo "======  DETACH WITH Ctrl+b d  ======" && CUDA_VISIBLE_DEVICES=$cuda_device uv run dreamer.py --configs atari100k --task atari_$game --logdir ./logdir/atari_$game"
CUDA_VISIBLE_DEVICES=$cuda_device uv run dreamer.py --configs atari100k --task atari_$game --logdir ./logdir/atari_$game --parallel True --envs 8