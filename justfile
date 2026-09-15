demo instruction:
  uv run mjpython run.py --viewer --instruction {{quote(instruction)}}

teleop *args:
  uv run mjpython teleop.py {{args}}

generate *args:
  uv run mjpython generate_training_data.py {{args}}

replay *args:
  uv run replay.py {{args}}

dataset *args:
  uv run dataset.py {{args}}

ffmpeg7 := `brew --prefix ffmpeg@7 2>/dev/null || echo /nonexistent`

check *args:
  DYLD_LIBRARY_PATH="{{ffmpeg7}}/lib:${DYLD_LIBRARY_PATH:-}" uv run check_dataset.py {{args}}
