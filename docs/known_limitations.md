# Known limitations and unfinished external setup

Updated 2026-10-26. These limits are part of the project result and should be
included in presentations and technical handoffs.

## Model and evaluation

- The model is a student-project demo, not validated for safety-critical,
  autonomous-driving, traffic-enforcement, or other high-impact use.
- Target evaluation reached 92.40% on this fixed 2,500-image dataset. On the
  mapped MIO-TCD stress set it reached only 43.24%; performance is poor for
  Bus, Truck, and Bike there. The external result shows limited generalization.
- MIO-TCD pedestrian/background categories were mapped to `None`, despite
  imperfect semantic equivalence. `None` is not a calibrated open-set rejector.
- Source archives, accepted licenses, original sample IDs, and source-to-curated
  lineage are not fully verified. Raw source dumps are therefore not included
  in the shared DVC set.
- There is no calibrated confidence threshold, fairness audit, geographic or
  temporal evaluation, or labeled production feedback loop.

## Reproducibility and operations

- DVC storage uses a local RustFS instance. It is free but local to the host;
  it is not a remote team service and is not accessible to collaborators until
  networking and credentials are provisioned. The Phase 8 repeat clean-clone
  retrieval exceeded this machine's temporary storage quota.
- MLflow uses local SQLite and local artifact storage by default. Runs and the
  model registry alias are not shared automatically with teammates.
- Evaluation reports contain a whole-worktree dirty flag inside `environment`
  and a code/config-only dirty flag at `run.git_working_tree_dirty`; generated
  reports and DVC lock updates can make those values differ. The target and
  OOD runs also recorded different DVC lock checksums because DVC updates the
  lock after each metrics stage. Compare their shared Git SHA, source snapshot,
  checkpoint SHA, and data-manifest SHA for model/data identity; interpret the
  lock checksum as the version observed when that individual stage started.
- Local measurements use synthetic traffic in kind on the same host. They are
  neither a production capacity estimate nor an SLO. Prediction-distribution
  counts have no labels and cannot detect accuracy loss by themselves.
- The API image serves CPU inference. Training configurations may select a
  GPU when run with a compatible CUDA-enabled PyTorch environment; this does
  not mean the current inference container uses the host GPU.
- GitHub Actions definitions exist, but repository-side branch protection,
  hosted workflow proof, and a published GHCR image have not been completed.
  No version tag was created, per the current project decision.
- Kubernetes was exercised locally with kind. Jenkins has not deployed an
  immutable GHCR digest because its agent, kubeconfig credential, and release
  digest require account-side setup.

## What would close the remaining proof gaps

1. Repeat a clean clone and `dvc pull` on a machine with adequate free disk,
   then verify the manifest SHA and validation report.
2. Decide on a reachable, affordable DVC remote and securely share access with
   a teammate; verify from their machine.
3. Configure GitHub required status checks and, when release is desired, publish
   a versioned GHCR image. Do not create a release tag until intentionally
   ready to publish.
4. Configure Jenkins agent and least-privilege kubeconfig credentials, then
   deploy that immutable image digest and capture its successful and failed
   rollout/rollback evidence.
5. Repeat load testing on the intended serving hardware/network and collect
   representative labeled feedback before proposing any service or model
   quality objective.
