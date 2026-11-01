<div align="center">

# ⚗️ Cheminformatics Virtual Laboratory

### SRM Institute of Science and Technology, Tiruchirappalli

**Faculty of Engineering and Technology | School of Sciences | Division of Chemistry**

</div>

> *From molecular structure to drug-likeness, explore chemistry through code.*

---

## 🎯 Aim

To build a strong foundation in **Cheminformatics and computer-aided drug design** by using computational tools to represent, visualize, analyze, screen and compare chemical structures, and to relate molecular structure to physicochemical properties and biological behaviour.

---

## 📚 Learning Objectives

After completing this virtual laboratory, students will be able to:

1. **Represent molecules** using SMILES, InChI and InChIKey, and convert between these formats.
2. **Visualize structures** in 2D and 3D, including highlighting functional groups and substructures.
3. **Calculate molecular descriptors** (physicochemical, topological and structural) for any valid molecule.
4. **Evaluate drug-likeness** using Lipinski's Rule of Five, Veber's rules and the Ghose filter.
5. **Predict ADME properties**, including absorption, solubility and membrane permeability.
6. **Perform substructure and pattern searches** using SMARTS queries to identify functional groups.
7. **Compute molecular fingerprints** (Morgan/ECFP, MACCS keys) and measure similarity using the Tanimoto coefficient.
8. **Screen and compare compound libraries**, ranking molecules against property criteria.
9. **Generate and optimize 3D conformers** and understand conformational flexibility.
10. **Understand structure–property relationships** and how small structural changes affect activity and drug-likeness.
11. **Import, clean and export chemical data** in SDF and CSV formats for downstream analysis.

---

## 🧬 Molecular Descriptors

The application calculates the following descriptors, grouped by purpose.

### Physicochemical Properties
| Descriptor | Significance |
|---|---|
| Molecular Weight (MW) | Size of the molecule; key for absorption |
| LogP (Lipophilicity) | Balance between water and fat solubility |
| Molar Refractivity (MR) | Molecular volume and polarizability |
| Heavy Atom Count | Number of non-hydrogen atoms |

### Polarity and Permeability
| Descriptor | Significance |
|---|---|
| Topological Polar Surface Area (TPSA) | Predicts membrane and blood–brain barrier permeability |
| Hydrogen Bond Donors (HBD) | Affects solubility and binding |
| Hydrogen Bond Acceptors (HBA) | Affects solubility and binding |

### Structural and Topological Features
| Descriptor | Significance |
|---|---|
| Rotatable Bonds | Molecular flexibility; oral bioavailability |
| Ring Count | Overall ring complexity |
| Aromatic Ring Count | Presence of aromatic systems |
| Fraction Csp3 | Saturation and 3D character |
| Formal Charge | Net electrical charge |

### Drug-Likeness and Similarity Metrics
| Metric | Significance |
|---|---|
| Lipinski Rule of Five violations | Oral drug-likeness screen |
| Veber rule compliance | Oral bioavailability (rotatable bonds, TPSA) |
| QED (Quantitative Estimate of Drug-likeness) | Single score from 0 to 1 |
| Tanimoto similarity | Structural similarity between two molecules |

---

## 🛠️ Installation

Follow the steps below to set up the laboratory on your computer.

**1. Clone or download the project**

```bash
git clone <your-repository-url>
cd cheminformatics-studio
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

**3. Install the required packages**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
python app.py
```

> 💡 **Note:** Python 3.8 or higher is required. If you use Conda, you can install RDKit with `conda install -c conda-forge rdkit`.
