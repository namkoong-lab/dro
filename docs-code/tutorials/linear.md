# Linear Models

These optimization problems are solved exactly (or approximately, e.g., kernel) through solvers.

Each method below links to its conceptual guide and literature references, its
estimator API documentation (where available), and a runnable notebook.

## $f$-divergence DRO

See the {doc}`$f$-divergence DRO guide and references <linear_fdro>` for the
standard and partial-shift formulations. Estimator references are available for
{doc}`KL-DRO <../api/apis/kldro_linear>`,
{doc}`chi-square DRO <../api/apis/chi2dro_linear>`,
{doc}`CVaR-DRO <../api/apis/cvardro_linear>`,
{doc}`TV-DRO <../api/apis/tvdro>`,
{doc}`conditional CVaR-DRO <../api/apis/conditionaldro>`, and
{doc}`marginal DRO <../api/apis/marginaldro>`. See the
{doc}`$f$-divergence DRO notebook <../api/notebooks/f_dro_tutorial>` for
runnable standard and partial-shift linear examples.

## Wasserstein DRO

See the {doc}`Wasserstein DRO guide and references <linear_wdro>` for the
standard and robust-satisficing formulations, the
{doc}`Wasserstein DRO API reference <../api/apis/wdro_linear>` for estimator
details, and the
{doc}`Wasserstein DRO notebook <../api/notebooks/WassersteinDRO_tutorial>` for
linear classification, regression, and robust-satisficing examples.

## Group DRO

See the {doc}`Group DRO guide and references <linear_groupdro>` for the
formulation, supported losses, and implementation details. The
{doc}`Group DRO notebook <../api/notebooks/groupdro_tutorial>` provides an
end-to-end linear and neural example.

## MMD-DRO

See the {doc}`MMD-DRO guide and references <linear_mmddro>` for the formulation
and kernel guidance, the {doc}`MMD-DRO API reference <../api/apis/mmddro>` for
estimator details, and the
{doc}`kernel DRO notebook <../api/notebooks/kernel_dro_tutorial>` for a runnable
example and ambiguity-set kernel selection guidance.

## Bayesian (Parametric) DRO

See the {doc}`Bayesian DRO guide and references <linear_pdro>` for the
frequentist and Bayesian formulations, the
{doc}`Bayesian DRO API reference <../api/apis/bayesian_dro>` for estimator
details, and the {doc}`Bayesian DRO notebook <../api/notebooks/param_dro>` for
runnable frequentist and Bayesian parametric examples.

## Mixed-distance DRO

See the {doc}`mixed-distance DRO guide and references <linear_mixdro>` for the
Sinkhorn, MOT, outlier-robust Wasserstein, and Holistic DRO formulations.
Estimator references are available for
{doc}`Sinkhorn DRO <../api/apis/sinkhorn_dro>`,
{doc}`MOT DRO <../api/apis/motdro>`,
{doc}`outlier-robust Wasserstein DRO <../api/apis/or_wdro>`, and
{doc}`Holistic Robust DRO <../api/apis/hrdro>`. See the
{doc}`mixed-distance DRO notebook <../api/notebooks/mixed_dro_tutorial>` for
runnable examples of all four methods.
