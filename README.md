# [CVPR 2026] Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing

[![Website](https://img.shields.io/badge/Website-Visit-blue)](https://seongraenoh.github.io/edit-as-act/)  
[![arXiv](https://img.shields.io/badge/arXiv-2603.17583-b31b1b.svg)](https://arxiv.org/abs/2603.17583)

## Requirements

- Ubuntu 24.04
- `uv`
- Blender 4.2

## Installation

1. Clone this repository.
2. Sync the environment with `uv`.

```bash
git clone https://github.com/SeongRaeNoh/Edit-As-Act.git
cd EditAsAct
uv sync
```

## Dataset



Download the dataset from Google Drive:
[Dataset Download](https://drive.google.com/file/d/1Pn41pI-txhvoqjLxLaLSJ9j2n93xNWtz/view?usp=sharing)

Important: The dataset link was updated on March 27, 2026. If you downloaded the dataset before this update, please re-download it.

After downloading, place the dataset under the `code/dataset/` directory so that the repository layout looks like this:

```text
EditAsAct/
├── code/
│   ├── dataset/
│   │   └── dataset/
│   │       ├── bedroom/
│   │       ├── dining_room/
│   │       ├── bathroom/
│   │       └── ...
│   └── ...
└── ...
```

## Usage

Set your OpenAI API key before running any LLM-based step:

```bash
export OPENAI_API_KEY="your_api_key"
```

### 1. Run the benchmark

This runs the full benchmark pipeline for a scene: terminal condition extraction followed by regression planning.

```bash
cd code
uv run python run_benchmark.py \
  --scene dataset/dataset/bedroom/scene_layout_edited.json \
  --instructions dataset/dataset/user_instructions_edit/instruction_bedroom.json \
  --output results/bedroom_results.json \
  --verbose
```

### 2. Convert predicted plans into scene JSON files

After the benchmark run, convert the predicted plans into edited scene layout JSON files:

```bash
cd code
uv run python tools/apply_plan_to_scene.py \
  --scene dataset/dataset/bedroom/scene_layout_edited.json \
  --plans results/bedroom_results.json \
  --outdir dataset/dataset/bedroom/plans_applied
```

This produces per-instruction scene files such as:

```text
code/dataset/dataset/bedroom/plans_applied/scene_layout_instruction_1_add.json
```

### 3. Visualize the edited scene in Blender

To visualize a scene JSON produced in Step 2, use the Blender import script:

1. Open `scene.blend`.
2. Go to the **Scripting** workspace.
3. Paste the contents of `tools/blender_scene_import.py` into the script editor.
4. Set `JSON_PATH` to the path of the scene JSON you want to visualize.
5. Click **Run Script**.

This will load the specified scene layout into Blender for visualization.

### Optional: Export a Blender scene to our dataset JSON format

If you want to export a Blender scene as a `scene_layout.json` file—for example, to test your own scene—use the export script:

1. Open `scene.blend`.
2. Go to the **Scripting** workspace.
3. Paste the contents of `tools/blender_scene_export.py` into the script editor.
4. Set `EXPORT_DIR` to your desired output directory.
5. Click **Run Script**.

## Todo

- [x] Dataset
- [x] Model
- [x] Visualization code 


The visualization code will be released by ~ 3/28

## BibTeX

```bibtex
@misc{noh2026editasactgoalregressiveplanningopenvocabulary,
      title={Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing}, 
      author={Seongrae Noh and SeungWon Seo and Gyeong-Moon Park and HyeongYeop Kang},
      year={2026},
      eprint={2603.17583},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.17583}, 
}
