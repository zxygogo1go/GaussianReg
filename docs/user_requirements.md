# User Requirements

## Research direction

- Build a fully Gaussian-native head-and-neck medical image registration model.
- Preserve hierarchical Gaussian representation, Gaussian correspondence,
  Gaussian velocity synthesis, and diffeomorphic integration as the model core.
- Add Small-Organ-Adaptive Gaussian Refinement as the third paper innovation.
- The third stage is motivated by the user's own MUSA extension, which produced
  a substantial small-organ improvement in prior experiments.

## Implementation constraints

- Work locally; the user synchronizes the repository to the school Linux server.
- Target hardware is one 40 GiB NVIDIA A100.
- Use PyTorch and remain compatible with the existing SACB conda environment.
- Fast local unit/smoke tests may run automatically; full training runs on the
  school server.
- Give explicit Python launch commands instead of creating shell scripts.
- Preserve existing V12 behavior and configurations.
- Do not use fixed target labels as model inference inputs.
- Training labels may supervise the small-organ Gaussian priority mechanism.
- Keep the user's untracked `小器官精修模块/` prototype unchanged.

### Document Preferences

- Language: Chinese explanations with existing repository documentation style.
- README location: repository root.
- Commands: explicit multi-line shell commands, no new launch scripts.
- Data format section: retain existing dataset documentation.
- Ablations: list each architectural component separately.
