# Paper References — verified collection

Every entry below is either (a) already confirmed in `research_papers.md`, or (b) newly found and
**verified via web search / arXiv** this session (title, authors, venue, year checked against the
source — per the citation-integrity rule, nothing fabricated). Items needing a final page/DOI check
before camera-ready are flagged `[verify]`.

---

## A. Core references (from research_papers.md — already confirmed)

1. **kratzert2019** — Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., & Nearing,
   G. (2019). *Towards learning universal, regional, and local hydrological behaviors via machine
   learning applied to large-sample datasets.* Hydrology and Earth System Sciences, 23, 5089–5110.
   — **Role:** the strong multi-basin LSTM baseline our L condition instantiates.

2. **kirschstein2024** — Kirschstein, N., & Sun, Y. (2024). *The merit of river network topology for
   neural flood forecasting.* ICML 2024, PMLR 235, pp. 24713–24725.
   — **Role:** the GNN-topology null we explain (topology-as-label inert; topology-as-flow specific).

3. **jiang2025** — Jiang, H., Wang, J., Zhu, X., & He, Y. (2025). *Topology-aware neural flux
   prediction guided by physics.* ICML 2025, PMLR 267.
   — **Role:** physics-aware directional operator; we operationalize the direction as a plain feature.

4. **nearing2021** — Nearing, G.S., Kratzert, F., Sampson, A.K., Pelissier, C.S., Klotz, D., Frame,
   J.M., Prieto, C., & Gupta, H.V. (2021). *What role does hydrological science play in the age of
   machine learning?* Water Resources Research, 57(3), e2020WR028091.
   — **Role:** framing — ML vs process understanding in hydrology.

5. **kipf2017** — Kipf, T.N., & Welling, M. (2017). *Semi-supervised classification with graph
   convolutional networks.* ICLR 2017. arXiv:1609.02907.
   — **Role:** the canonical GCN; structure-helps-in-can't-memorize-regime framing.

6. **topping2022** — Topping, J., Di Giovanni, F., Chamberlain, B.P., Dong, X., & Bronstein, M.M.
   (2022). *Understanding over-squashing and bottlenecks on graphs via curvature.* ICLR 2022.
   arXiv:2111.14522.
   — **Role:** GNN limitation (over-squashing) — why message passing struggles on deep trees.

7. **rusch2022** — Rusch, T.K., Chamberlain, B.P., Rowbottom, J., Mishra, S., & Bronstein, M.M.
   (2022). *Graph-coupled oscillator networks.* ICML 2022, PMLR 162. arXiv:2202.02296.
   — **Role:** GNN over-smoothing mitigation (optional; discussion depth).

8. **bodnar2022** — Bodnar, C., Di Giovanni, F., Chamberlain, B.P., Liò, P., & Bronstein, M.M.
   (2022). *Neural sheaf diffusion: A topological perspective on heterophily and oversmoothing in
   GNNs.* NeurIPS 2022. arXiv:2202.04579.
   — **Role:** GNN limitation theory (optional; discussion depth).

---

## B. New references (verified this session — ≥5 as requested)

9. **kratzert2018** — Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M. (2018).
   *Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks.* Hydrology and Earth
   System Sciences, 22, 6005–6022. DOI:10.5194/hess-22-6005-2018.
   — **Role:** the foundational LSTM-for-rainfall-runoff paper; establishes LSTMs match/beat
     conceptual models (SAC-SMA + Snow-17) on CAMELS. Missing from our set; a reviewer expects it.
     Source: hess.copernicus.org/articles/22/6005/2018/.

10. **addor2017** — Addor, N., Newman, A.J., Mizukami, N., & Clark, M.P. (2017). *The CAMELS data
    set: catchment attributes and meteorology for large-sample studies.* Hydrology and Earth System
    Sciences, 21, 5293–5313. DOI:10.5194/hess-21-5293-2017.
    — **Role:** the CAMELS **attributes** dataset citation (our static attributes + basin set).
      Source: hess.copernicus.org/articles/21/5293/2017/.

11. **newman2015** — Newman, A.J., Clark, M.P., Sampson, K., Wood, A., Hay, L.E., Bock, A., et al.
    (2015). *Development of a large-sample watershed-scale hydrometeorological data set for the
    contiguous USA: data set characteristics and assessment of regional variability in hydrologic
    model performance.* Hydrology and Earth System Sciences, 19, 209–223.
    DOI:10.5194/hess-19-209-2015.
    — **Role:** the CAMELS **forcing/timeseries** dataset citation (Daymet/Maurer/NLDAS forcings we
      use; the SAC-SMA benchmark). Source: hess.copernicus.org/articles/19/209/2015/.

11b. **newman2014data** — Newman, A., Sampson, K., Clark, M.P., Bock, A., Viger, R.J., & Blodgett,
    D. (2014). *A large-sample watershed-scale hydrometeorological dataset for the contiguous USA*
    [data set]. Boulder, CO: UCAR/NCAR. **DOI:10.5065/D6MW2F4D** (verified: resolves live).
    — **Role:** the **CAMELS data PRODUCT** citation — the actual multi-GB downloadable archive
      (forcings + USGS streamflow) we run on, distinct from the two describing HESS papers. Data
      hosts request this dataset DOI be cited when the files are used. Source:
      ral.ucar.edu/solutions/products/camels ; DOI dx.doi.org/10.5065/D6MW2F4D.

12b. **kratzert2022joss** — Kratzert, F., Gauch, M., Nearing, G., & Klotz, D. (2022).
    *NeuralHydrology — A Python library for Deep Learning research in hydrology.* Journal of Open
    Source Software, 7(71), 4050. **DOI:10.21105/joss.04050**.
    — **Role:** the **codebase/software** citation. Source: the repo's own `CITATION.cff` +
      `README.md` (authoritative — the maintainers' requested citation). All experiments run on
      stock NeuralHydrology `cudalstm`, so this must be cited.

12. **hoedt2021** — Hoedt, P.-J., Kratzert, F., Klotz, D., Halmich, C., Holzleitner, M., Nearing,
    G., Hochreiter, S., & Klambauer, G. (2021). *MC-LSTM: Mass-Conserving LSTM.* arXiv:2101.05186.
    (ICML 2021.) `[verify: confirm ICML 2021 proceedings vs arXiv-only before camera-ready]`
    — **Role:** the mass-conserving, physically-grounded LSTM the hydrology audience expects as a
      comparison point (our FORWARD_PLAN flagged mclstm). Source: arxiv.org/abs/2101.05186.

13. **zhang2021magnet** — Zhang, X., He, Y., Brugnone, N., Perlmutter, M., & Hirn, M. (2021).
    *MagNet: A Neural Network for Directed Graphs.* NeurIPS 2021. arXiv:2102.11391.
    — **Role:** directed-GNN theory — the magnetic Laplacian encodes edge *direction*; supports the
      Discussion point that standard (symmetric) message passing is direction-insensitive, which
      is exactly the failure mode our directed flow feature sidesteps. Source: arxiv.org/abs/2102.11391.

---

## C. Optional / discussion-depth (verified topic, cite only if the section needs them)

- **Directed GNNs beyond MagNet** (DirGNN / edge-directionality-improves-learning, Rossi et al.
  2023, arXiv:2305.10498) `[verify authors/venue]` — if the Discussion goes deeper on why
  directionality matters for message passing. Only if it earns its place.
- Recent GNN-for-streamflow works (spatio-temporal GNNs, 2023–2024) exist and could contextualize
  Related Work, but most are station-graph forecasting, not the topology-merit question — cite
  sparingly and only where they clarify difference, not as a dump.

---

## Notes
- **MC-LSTM venue:** arXiv confirms Hoedt et al. 2101.05186; commonly cited as ICML 2021 — verify
  the proceedings entry before camera-ready (flagged above).
- **MagNet arXiv ID** verified as 2102.11391 (NeurIPS 2021), not the id a coarse search suggested.
- Every DOI/page above was read from the source this session; still do a final pass at camera-ready.
