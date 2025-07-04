
# Type 1 Corrected PF Missing pT

## Overview

This is a documentation issue describing the computation of the **Type 1 corrected PF missing pT** in the standard reconstruction.

---

## NANOAOD branch

This section provides [explanations](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD) for certain variables in the Run 2 NanoAOD dataset to support the use and validation of our correction code.

`RawMET`: raw (i.e., uncorrected) PF MET

`MET`: Type-1 corrected PF MET, it does not include other corrections such as the $\phi$ modulation or propagation of JER smearing. 

`Jet`: slimmedJets, i.e. ak4 PFJets CHS with JECs applied, after basic selection (pt > 15)

`CorrT1METJet`: Additional low-pt jets for Type-1 MET re-correction

---
## Type-I Correction 

The [Type-I correction](https://twiki.cern.ch/twiki/bin/viewauth/CMS/METType1Type2Formulae#3_The_Type_I_correction) is the most popular MET correction in CMS. This correction is a propagation of the jet energy corrections (JEC) to MET. The Type-I correction replaces the vector sum of transverse momenta of particles which can be clustered as jets with the vector sum of the transverse momenta of the jets to which JEC is applied. 

### Implementation Details:

The jet collection used in [Type-I corrections](https://gitlab.cern.ch/hgao/coffea/-/blob/master/jetmet_tools/CorrectedMETFactory.py?ref_type=heads#L7-13) for PF  MET is AK4PFchs jets with JEC corrected Pt>15 GeV (using the L1L2L3(+Res) -L1 scheme):

$$
\vec{p}_T^{\,\text{miss,type-1}} \leftarrow \vec{p}_T^{\,\text{miss}} - \sum_{\text{jets}} \left( \vec{p}_T^{\,\text{L1L2L3}} - \vec{p}_T^{\,\text{L1}} \right)
$$
The jets that are used to correct MET are also required to have electromagnetic energy fraction [smaller than 0.9](https://gitlab.cern.ch/hgao/smqawa/-/blob/master/src/qawa/jme_gh.py?ref_type=heads#L122):
```cpp
jet.chargedEmEnergyFraction() + jet.neutralEmEnergyFraction() > 0.9
```
And not to be overlapping with the pf muon candidate, scaling the jet pT by `(1 − Jet_muonSubtrFactor)`

Therefore, the final formula is:

$$\begin{split} 
\vec{E}{\rm{_T^{Type-I}}} &= \vec{E}{\rm{_T^{raw}}}+ \vec{C}{\rm{_T^{Type-I}}} \\
&= \vec{E}{\rm{_T^{raw}}} -\sum\limits_{\rm jet}{\left(\left(f_{JEC}^{L123} - f_{JEC}^{L1}\right)\cdot \left(1-f_{\rm raw}\right)\cdot \left(1-f_{\rm muon}\right)  \cdot \vec{p}_{\rm T,\ jet} \right)} 
\end{split}$$

In this formula, the term **jet** refers to all slimmedJets `Jet_*` and soft jets `CorrT1METJet_*` that pass the selection criteria described above.

Branches [`CorrT1METJet_*`](https://gitlab.cern.ch/hgao/smqawa/-/blob/master/src/qawa/process/zz2l2nu_vbs.py?ref_type=heads#L1079) provide a reduced set of properties of jets with corrected ￼jet_pt < 15 GeV. They are needed to propagate JEC uncertainties or JER smearing into missing ET.

The resulting $\vec{E}{\rm{_T^{Type-I}}}$ from the above formula can be compared against the `MET` in the NanoAOD, and the pt difference is less than 1%.
## [MET Type-I Smear correction](https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETRun2Corrections#Type_I_Correction_Propagation_of)
Jets in simulation can be smeared (as shown in [JetResolution twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetResolution)) to achieve better agreement with data. This correction is a propagation of the smeared such jets to MET.  The Smeared MET correction replaces the vector sum of transverse momenta of particles which can be clustered as jets with the vector sum of the transverse momenta of the jets to which smearing is applied. 

The procedure for Jet Energy Resolution ([JER](https://gitlab.cern.ch/hgao/coffea/-/blob/master/jetmet_tools/CorrectedJetsFactory.py?ref_type=heads#L269-271)) smearing appears to be **identical** to the above Type 1 correction, except that it:

- Starts from jets with **smearing applied**.
- Only rescales the [**jet momentum**](https://gitlab.cern.ch/hgao/smqawa/-/blob/master/src/qawa/jme_gh.py?ref_type=heads#L83).
- As a result, **JER smearing is incorrectly propagated into the L1-corrected momentum**, which introduces inconsistencies.

The [JER propagation](https://gitlab.cern.ch/hgao/smqawa/-/blob/master/src/qawa/process/zz2l2nu_vbs.py?ref_type=heads#L1093) formula is:

$$\begin{split} 
\vec{E}{\rm{_T^{Type-I}}} &= \vec{E}{\rm{_T^{raw}}}+ \vec{C}{\rm{_T^{Type-I}}} \\
&= \vec{E}{\rm{_T^{raw}}} -\sum\limits_{\rm jet}{\left(\left(f_{JEC}^{L123}\cdot f_{JER} - f_{JEC}^{L1}\right)\cdot \left(1-f_{\rm raw}\right)\cdot \left(1-f_{\rm muon}\right)  \cdot \vec{p}_{\rm T,\ jet} \right)} 
\end{split}$$
