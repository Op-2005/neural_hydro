# Dataset — provenance & citation (supplementary material)

Grounded against disk (`datasets/camels_us/`) and the NeuralHydrology docs. Every file listed
below is confirmed present in this repo. Nothing recalled.

## What we use

**CAMELS US catchment attributes v2.0** — static basin attributes for 671 contiguous U.S.
catchments, plus the daily meteorological forcings and observed streamflow, as loaded by the
NeuralHydrology `camelsus` dataset class.

### Attribute package (on disk: `datasets/camels_us/camels_attributes_v2.0/`)
The v2.0 attributes folder, confirmed present, contains:
`camels_clim.txt`, `camels_geol.txt`, `camels_hydro.txt`, `camels_name.txt`, `camels_soil.txt`,
`camels_topo.txt`, `camels_vege.txt` (plus our generated `camels_topology.txt` for the ablation's
topology features, and `readme-2.txt`).

Of these, the study reads: `camels_topo.txt` (drainage area, elevation, slope — used both as static
inputs and to area-weight the `upstream_q` aggregation) and the auto-loaded generated
`camels_topology.txt`. The 5 static attributes fed to every model are `elev_mean`, `area_gages2`,
`slope_mean`, `p_mean`, `pet_mean` (from `camels_topo.txt` / `camels_clim.txt`).

### Forcings & streamflow
Daily Maurer-product forcings (`PRCP`, `SRAD`, `Tmax`, `Tmin`, `Vp`) and observed discharge
(`QObs(mm/d)`, area-normalized) per basin, 1980–2014, as distributed with CAMELS.

## Citations (both required)

**Attributes dataset (the citation for the v2.0 attribute package we use):**
> Addor, N., Newman, A. J., Mizukami, N., & Clark, M. P. (2017). The CAMELS data set: catchment
> attributes and meteorology for large-sample studies. *Hydrology and Earth System Sciences*, 21,
> 5293–5313. DOI:10.5194/hess-21-5293-2017.
— bib key `addor2017`. This is the **primary dataset citation** for our static attributes.

**Forcing/timeseries dataset + benchmark (cite alongside for the forcings):**
> Newman, A. J., Clark, M. P., Sampson, K., Wood, A., Hay, L. E., Bock, A., et al. (2015).
> Development of a large-sample watershed-scale hydrometeorological data set for the contiguous
> USA. *Hydrology and Earth System Sciences*, 19, 209–223. DOI:10.5194/hess-19-209-2015.
— bib key `newman2015`.

**Data product (the downloadable archive — cite if the venue wants the data DOI):**
> Newman, A., Sampson, K., Clark, M. P., Bock, A., Viger, R. J., & Blodgett, D. (2014). A
> large-sample watershed-scale hydrometeorological dataset for the contiguous USA [data set].
> UCAR/NCAR. DOI:10.5065/D6MW2F4D.
— bib key `newman2014data`. Verified to resolve. Some venues want the data-product DOI in addition
to the describing papers; keep it, cite it in Data availability.

**Software (the codebase):**
> Kratzert, F., Gauch, M., Nearing, G., & Klotz, D. (2022). NeuralHydrology — A Python library for
> Deep Learning research in hydrology. *Journal of Open Source Software*, 7(71), 4050.
> DOI:10.21105/joss.04050.
— bib key `kratzert2022joss`. From the repo's own `CITATION.cff`.

## Ready-to-use dataset sentence (for the Data section)

> We use the CAMELS US catchment attributes v2.0 dataset~\cite{addor2017}, which provides static
> basin attributes for 671 contiguous U.S.\ catchments, together with the associated daily
> Maurer-product forcings and observed streamflow~\cite{newman2015}. Our study network is a
> connected 183-basin eastern-U.S.\ sub-network (Component~0).

## Landing pages (for Data availability / reproducibility, not for citing in place of the papers)
- NCAR CAMELS attributes: ral.ucar.edu/solutions/products/camels
- Zenodo CAMELS record: (as noted by the data host)

## Notes
- The **attributes** citation is `addor2017`; the **forcings** citation is `newman2015`; the
  **data-product DOI** is `newman2014data`. Large-sample hydrology papers commonly cite all three
  — attributes + forcings papers in Data/Methods, data-product DOI in Data availability.
- Source of this provenance: NeuralHydrology docs (camelsus dataset class) + files verified on disk.
