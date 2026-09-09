# Representative Prevalence Panel

The original factorial panel is label-stratified. It remains useful for rare
class transition analysis, but its marginal label shares cannot be compared
with the natural SFW distribution.

This second panel is the prevalence panel. It samples matched raw-parent
records by simple random sampling from the full eligible bridge, without using
old labels, class quotas, or a role cap. Every selected parent retains every
linked atomic child. The same four prompt-by-model conditions will be applied
to both raw parents and atomic children.

This design answers whether the observed historical approximately 24% non-E0
rate and the newer approximately 67% non-E0 rate repeat under controlled
prompt/model/grain conditions. The comparison is valid only when both figures
are computed at the same grain and with the same E23 merge rule.

The current run's label-stratified responses are preserved as
`results/factorial_restart_v1/`. The representative panel will be stored under
`results/factorial_representative_v1/`, with its own manifest and response
checkpoint. No old or new label is included in the outbound manifest.

The final analysis will report natural prevalence estimates, uncertainty, and
the factorial effects separately. It will not combine the two sampling designs
into one unweighted headline percentage.
