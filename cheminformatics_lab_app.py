"""
Cheminformatics Virtual Laboratory
SRM Institute of Science and Technology, Tiruchirappalli
Faculty of Engineering and Technology | School of Sciences | Division of Chemistry

Developed by:
    VISMITHA PARASA  (RA2611030050047)
    VARSHINI A       (RA2611030050040)

LOGO: place your college logo in the same folder as this file and name it
one of:  logo.png | logo.jpg | logo.jpeg | srm_logo.png | srm_logo.jpg | srm_logo.jpeg
It will appear (slightly faded) at the top-left corner of the app and on PDF reports.
"""

import os
import base64
import tempfile
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

# RDKit
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Draw,
    AllChem,
    Descriptors,
    Crippen,
    Lipinski,
    rdMolDescriptors,
    rdFingerprintGenerator,
    MACCSkeys,
    QED,
)

# 3D visualization
import py3Dmol

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    Preformatted,
)


# ============================================================
# PROJECT CONSTANTS
# ============================================================

APP_TITLE = "Cheminformatics Virtual Laboratory"
INSTITUTE = "SRM Institute of Science and Technology, Tiruchirappalli"
FACULTY_LINE = (
    "Faculty of Engineering and Technology | School of Sciences | Division of Chemistry"
)

DEVELOPERS = [
    {"name": "VISMITHA PARASA", "reg": "RA2611030050047"},
    {"name": "VARSHINI A", "reg": "RA2611030050040"},
]

# Theme colours (deep ocean teal + amber accent)
C_DARK = "#0B3C49"
C_TEAL = "#1B998B"
C_AMBER = "#F4A259"
C_LIGHT = "#E6F4F1"

LOGO_CANDIDATES = [
    "logo.png", "logo.jpg", "logo.jpeg",
    "srm_logo.png", "srm_logo.jpg", "srm_logo.jpeg",
]

EXAMPLE_MOLECULES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
}

FUNCTIONAL_GROUPS = {
    "Alcohol (–OH)": "[CX4][OX2H]",
    "Phenol (Ar–OH)": "c[OX2H]",
    "Carboxylic acid (–COOH)": "[CX3](=O)[OX2H1]",
    "Ester (–COO–)": "[#6][CX3](=O)[OX2H0][#6]",
    "Amide (–CONH–)": "[NX3][CX3](=[OX1])[#6]",
    "Amine (primary/secondary)": "[NX3;H2,H1;!$(NC=O)]",
    "Aldehyde (–CHO)": "[CX3H1](=O)[#6]",
    "Ketone (C=O)": "[#6][CX3](=O)[#6]",
    "Ether (C–O–C)": "[OD2]([#6])[#6]",
    "Benzene ring": "c1ccccc1",
    "Halogen (F, Cl, Br, I)": "[F,Cl,Br,I]",
    "Nitro (–NO2)": "[N+](=O)[O-]",
    "Nitrile (–C≡N)": "C#N",
    "Thiol (–SH)": "[SX2H]",
    "Alkene (C=C)": "[CX3]=[CX3]",
    "Alkyne (C≡C)": "C#C",
}


# ============================================================
# PAGE CONFIGURATION AND THEME
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚗️",
    layout="wide",
)

st.markdown(
    f"""
<style>
.lab-banner {{
    background: linear-gradient(120deg, {C_DARK} 0%, {C_TEAL} 100%);
    border-radius: 16px;
    padding: 26px 30px;
    margin-bottom: 18px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(11, 60, 73, 0.25);
}}
.lab-banner h1 {{
    color: #FFFFFF !important;
    font-size: 36px;
    margin: 0 0 6px 0;
    padding: 0;
}}
.lab-banner .inst {{
    color: #FFFFFF;
    font-size: 21px;
    font-weight: 700;
}}
.lab-banner .fac {{
    color: {C_AMBER};
    font-size: 16px;
    font-weight: 600;
    margin-top: 4px;
}}
.page-banner {{
    background: linear-gradient(120deg, {C_DARK} 0%, {C_TEAL} 100%);
    border-radius: 14px;
    padding: 16px 24px;
    margin-bottom: 16px;
}}
.page-banner h2 {{
    color: #FFFFFF !important;
    margin: 0;
    padding: 0;
    font-size: 28px;
}}
.page-banner p {{
    color: {C_LIGHT};
    margin: 4px 0 0 0;
    font-size: 15px;
}}
.info-card {{
    background: {C_LIGHT};
    color: {C_DARK};
    border-left: 6px solid {C_AMBER};
    border-radius: 10px;
    padding: 12px 18px;
    margin: 8px 0;
}}
.dev-card {{
    background: #FFFFFF;
    color: {C_DARK};
    border: 2px solid {C_TEAL};
    border-radius: 14px;
    padding: 14px 18px;
    text-align: center;
    margin: 6px 0;
}}
.dev-card .nm {{ font-size: 20px; font-weight: 800; color: {C_DARK}; }}
.dev-card .rg {{ font-size: 15px; color: {C_TEAL}; font-weight: 700; }}
[data-testid="stMetric"] {{
    background: #FFFFFF;
    border-left: 5px solid {C_TEAL};
    border-radius: 10px;
    padding: 10px 14px;
    box-shadow: 0 1px 6px rgba(0, 0, 0, 0.10);
}}
[data-testid="stMetric"] * {{ color: {C_DARK} !important; }}
.stButton > button, .stDownloadButton > button {{
    background-color: {C_TEAL};
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background-color: {C_DARK};
    color: #FFFFFF;
}}
.logo-box img {{
    width: 115px;
    opacity: 0.65;
    display: block;
    margin: 0 0 6px 0;
    transition: opacity 0.3s;
}}
.logo-box img:hover {{ opacity: 1; }}
</style>
""",
    unsafe_allow_html=True,
)


def page_header(title, subtitle=""):
    """Themed banner shown at the top of every page."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="page-banner"><h2>{title}</h2>{sub}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# LOGO HELPERS
# ============================================================

def find_logo_path():
    base = os.path.dirname(os.path.abspath(__file__))
    for fname in LOGO_CANDIDATES:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            return path
    return None


def get_logo_html():
    path = find_logo_path()
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<div class="logo-box"><img src="data:{mime};base64,{b64}" alt="College logo"/></div>'


# ============================================================
# SHARED MOLECULE STATE
# ============================================================

if "shared_smiles" not in st.session_state:
    st.session_state.shared_smiles = "CCO"


def get_molecule(smiles):
    """Convert SMILES into an RDKit molecule (None if invalid)."""
    if not smiles or not str(smiles).strip():
        return None
    try:
        return Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None


def molecule_input(prefix):
    """SMILES input + example loader that shares the molecule across pages."""
    key = f"smiles_{prefix}"
    ex_key = f"example_{prefix}"

    if key not in st.session_state:
        st.session_state[key] = st.session_state.shared_smiles

    def _sync():
        st.session_state.shared_smiles = st.session_state[key]

    def _load_example():
        smi = EXAMPLE_MOLECULES[st.session_state[ex_key]]
        st.session_state[key] = smi
        st.session_state.shared_smiles = smi

    col1, col2 = st.columns([2, 1])
    with col1:
        smiles = st.text_input("Enter SMILES", key=key, on_change=_sync)
    with col2:
        st.selectbox("Choose Example Molecule", list(EXAMPLE_MOLECULES.keys()), key=ex_key)
        st.button("Load Example", on_click=_load_example, key=f"load_{prefix}")

    st.session_state.shared_smiles = smiles
    return smiles


# ============================================================
# CORE CHEMISTRY FUNCTIONS
# ============================================================

def calculate_bond_information(mol):
    """Heavy-atom bonds, sigma bonds, pi bonds and total atoms including H."""
    heavy_atom_bonds = mol.GetNumBonds()

    kekule_mol = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule_mol, clearAromaticFlags=True)
    except Exception:
        pass

    pi_bonds = 0
    for bond in kekule_mol.GetBonds():
        bt = bond.GetBondType()
        if bt == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif bt == Chem.BondType.TRIPLE:
            pi_bonds += 2
        elif bt == Chem.BondType.QUADRUPLE:
            pi_bonds += 3
        elif bond.GetIsAromatic():
            pi_bonds += 0.5

    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_with_h.GetNumBonds()

    return {
        "Heavy Atom Bonds": heavy_atom_bonds,
        "Sigma Bonds": sigma_bonds,
        "Pi Bonds": pi_bonds,
        "Total Bonds": sigma_bonds + pi_bonds,
        "Total Atoms Including H": mol_with_h.GetNumAtoms(),
    }


def lipinski_violations(mw, logp, hbd, hba):
    """Return the number of Lipinski Rule-of-Five violations and the details."""
    checks = [
        ("Molecular Weight ≤ 500", mw, mw <= 500),
        ("LogP ≤ 5", logp, logp <= 5),
        ("H-Bond Donors ≤ 5", hbd, hbd <= 5),
        ("H-Bond Acceptors ≤ 10", hba, hba <= 10),
    ]
    violations = sum(1 for _, _, ok in checks if not ok)
    return violations, checks


def calculate_properties(mol):
    """Complete descriptor set, including drug-likeness metrics."""
    bond_info = calculate_bond_information(mol)

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    mr = Crippen.MolMR(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = Lipinski.NumRotatableBonds(mol)

    violations, _ = lipinski_violations(mw, logp, hbd, hba)
    veber = "Pass" if (rot <= 10 and tpsa <= 140) else "Fail"
    total_atoms_h = bond_info["Total Atoms Including H"]
    ghose = (
        "Pass"
        if (160 <= mw <= 480 and -0.4 <= logp <= 5.6 and 40 <= mr <= 130
            and 20 <= total_atoms_h <= 70)
        else "Fail"
    )

    try:
        qed_value = round(QED.qed(mol), 3)
    except Exception:
        qed_value = float("nan")

    return {
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": round(mw, 3),
        "Exact Molecular Weight": round(Descriptors.ExactMolWt(mol), 4),
        "LogP": round(logp, 3),
        "Molar Refractivity": round(mr, 3),
        "TPSA": round(tpsa, 3),
        "Hydrogen Bond Donors": hbd,
        "Hydrogen Bond Acceptors": hba,
        "Rotatable Bonds": rot,
        "Ring Count": Lipinski.RingCount(mol),
        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Aliphatic Rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "Number of Atoms": mol.GetNumAtoms(),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Heavy Atom Bonds": bond_info["Heavy Atom Bonds"],
        "Sigma (σ) Bonds": bond_info["Sigma Bonds"],
        "Pi (π) Bonds": bond_info["Pi Bonds"],
        "Total Bonds (σ + π)": bond_info["Total Bonds"],
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "Formal Charge": Chem.GetFormalCharge(mol),
        "Lipinski Violations": violations,
        "Veber Rule": veber,
        "Ghose Filter": ghose,
        "QED": qed_value,
    }


def get_identifiers(mol):
    """SMILES, canonical SMILES, InChI and InChIKey."""
    try:
        inchi = Chem.MolToInchi(mol)
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchi, inchikey = "Not available", "Not available"
    return {
        "Canonical SMILES": Chem.MolToSmiles(mol),
        "InChI": inchi or "Not available",
        "InChIKey": inchikey or "Not available",
    }


def generate_2d_image(mol, size=(700, 500), highlight_atoms=None):
    if highlight_atoms:
        return Draw.MolToImage(mol, size=size, highlightAtoms=list(highlight_atoms))
    return Draw.MolToImage(mol, size=size)


def generate_3d_molecule(mol):
    """Single optimized 3D conformer."""
    try:
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol3d, params) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol3d)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol3d)
            except Exception:
                pass
        return mol3d
    except Exception:
        return None


def generate_conformers(mol, n_confs=10):
    """Generate and optimize several conformers; return (mol, table DataFrame)."""
    molh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    cids = list(AllChem.EmbedMultipleConfs(molh, numConfs=n_confs, params=params))
    if not cids:
        return None, None

    energies = None
    method = "MMFF94"
    try:
        res = AllChem.MMFFOptimizeMoleculeConfs(molh, maxIters=2000)
        energies = [r[1] for r in res]
        if all(e < 0 and e == -1.0 for e in energies):
            raise ValueError("MMFF unavailable")
    except Exception:
        try:
            res = AllChem.UFFOptimizeMoleculeConfs(molh, maxIters=2000)
            energies = [r[1] for r in res]
            method = "UFF"
        except Exception:
            energies = [float("nan")] * len(cids)
            method = "None"

    df = pd.DataFrame({"Conformer ID": cids, "Energy (kcal/mol)": np.round(energies, 3)})
    if df["Energy (kcal/mol)"].notna().any():
        df["Relative Energy (kcal/mol)"] = (
            df["Energy (kcal/mol)"] - df["Energy (kcal/mol)"].min()
        ).round(3)
    df.attrs["method"] = method
    return molh, df


def display_3d_structure(mol3d, style, show_surface=False, conf_id=-1):
    mol_block = Chem.MolToMolBlock(mol3d, confId=conf_id)
    view = py3Dmol.view(width=900, height=600)
    view.addModel(mol_block, "mol")

    if style == "Ball and Stick":
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.30}})
    elif style == "Stick":
        view.setStyle({"stick": {"radius": 0.20}})
    elif style == "Space Filling":
        view.setStyle({"sphere": {"scale": 1.0}})
    elif style == "Wireframe":
        view.setStyle({"line": {"linewidth": 2}})
    elif style == "Cartoon":
        view.setStyle({"stick": {}, "sphere": {"scale": 0.20}})

    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.7})

    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=620)


# ---------- Fingerprints and similarity ----------

def get_fingerprint(mol, fp_type, radius=2, n_bits=2048):
    if fp_type == "MACCS Keys":
        return MACCSkeys.GenMACCSKeys(mol)
    if fp_type == "RDKit Topological":
        return Chem.RDKFingerprint(mol, fpSize=n_bits)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprint(mol)


def similarity_label(value):
    if value >= 0.85:
        return "Very high similarity (near-identical scaffold)"
    if value >= 0.6:
        return "High similarity"
    if value >= 0.4:
        return "Moderate similarity"
    if value >= 0.2:
        return "Low similarity"
    return "Very low similarity (structurally different)"


# ---------- Library helpers ----------

def library_row(name, smiles, mol):
    p = calculate_properties(mol)
    return {
        "Name": name,
        "SMILES": smiles,
        "MW": p["Molecular Weight"],
        "LogP": p["LogP"],
        "TPSA": p["TPSA"],
        "HBD": p["Hydrogen Bond Donors"],
        "HBA": p["Hydrogen Bond Acceptors"],
        "RotB": p["Rotatable Bonds"],
        "Rings": p["Ring Count"],
        "Aromatic Rings": p["Aromatic Rings"],
        "Fsp3": p["Fraction Csp3"],
        "QED": p["QED"],
        "Lipinski Violations": p["Lipinski Violations"],
        "Veber": p["Veber Rule"],
        "Ghose": p["Ghose Filter"],
    }


def dataframe_to_sdf(df):
    sio = StringIO()
    writer = Chem.SDWriter(sio)
    for _, row in df.iterrows():
        m = Chem.MolFromSmiles(str(row["SMILES"]))
        if m is None:
            continue
        m.SetProp("_Name", str(row["Name"]))
        for col in df.columns:
            if col not in ("Name", "SMILES"):
                m.SetProp(str(col), str(row[col]))
        writer.write(m)
    writer.close()
    return sio.getvalue()


# ============================================================
# INTERPRETATION HELPERS
# ============================================================

def get_property_interpretation(name, value, properties=None):
    if name == "Molecular Formula":
        return "Elemental composition of the molecule."
    if name == "Molecular Weight":
        return ("Average molecular mass. Drug-like oral molecules are generally "
                "below 500 g/mol (Lipinski).")
    if name == "Exact Molecular Weight":
        return "Monoisotopic mass, useful when comparing mass-spectrometry data."
    if name == "LogP":
        if value < 0:
            return "Negative value: strongly hydrophilic, prefers the aqueous phase."
        if value < 1:
            return "Low lipophilicity; favours polar environments."
        if value < 3:
            return "Moderate lipophilicity; a balance of polar and non-polar character."
        if value <= 5:
            return "Relatively high lipophilicity, but still within the Lipinski limit (≤ 5)."
        return "Very high lipophilicity (> 5); may cause poor solubility."
    if name == "Molar Refractivity":
        return "Related to molecular volume and polarizability (Ghose range 40–130)."
    if name == "TPSA":
        if value < 60:
            return "Low polarity; favourable for membrane and blood–brain barrier crossing."
        if value <= 140:
            return "Moderate polarity; generally compatible with oral absorption (≤ 140 Å²)."
        return "High polarity (> 140 Å²); membrane permeability is likely poor."
    if name == "Hydrogen Bond Donors":
        return f"{value} donor site(s); Lipinski limit is ≤ 5."
    if name == "Hydrogen Bond Acceptors":
        return f"{value} acceptor site(s); Lipinski limit is ≤ 10."
    if name == "Rotatable Bonds":
        if value == 0:
            return "No rotatable bonds; rigid framework."
        return f"{value} rotatable bond(s); Veber rule suggests ≤ 10 for good oral bioavailability."
    if name == "Ring Count":
        return f"{value} ring(s) contribute to shape and rigidity."
    if name == "Aromatic Rings":
        return f"{value} aromatic ring(s) with delocalized π-electrons."
    if name == "Aliphatic Rings":
        return f"{value} non-aromatic ring(s)."
    if name == "Number of Atoms":
        return "Atoms explicitly present in the RDKit graph (implicit hydrogens excluded)."
    if name == "Heavy Atoms":
        return "All non-hydrogen atoms; a simple size measure."
    if name == "Heavy Atom Bonds":
        return "Bonds between non-hydrogen atoms."
    if name == "Sigma (σ) Bonds":
        return "Every bonded atom pair has one σ bond (hydrogens included)."
    if name == "Pi (π) Bonds":
        return "π bonds come from double/triple bonds and aromatic systems."
    if name == "Total Bonds (σ + π)":
        return "Sum of σ and π bond components."
    if name == "Fraction Csp3":
        if value < 0.25:
            return "Mostly planar / unsaturated character."
        if value < 0.60:
            return "Balanced saturated and unsaturated character."
        return "Highly saturated, more three-dimensional framework."
    if name == "Formal Charge":
        return "Net electrical charge of the molecule."
    if name == "Lipinski Violations":
        if value == 0:
            return "No violations: fully complies with the Rule of Five."
        if value == 1:
            return "One violation is generally tolerated for oral drug-likeness."
        return "Two or more violations: oral bioavailability is likely to be poor."
    if name == "Veber Rule":
        return "Pass means ≤ 10 rotatable bonds and TPSA ≤ 140 Å² (good oral bioavailability)."
    if name == "Ghose Filter":
        return "Pass means MW 160–480, LogP –0.4 to 5.6, MR 40–130, 20–70 atoms."
    if name == "QED":
        if value >= 0.67:
            return "High drug-likeness (favourable)."
        if value >= 0.35:
            return "Moderate drug-likeness."
        return "Low drug-likeness."
    return "Additional descriptor describing molecular structure."


def build_key_interpretation_points(p):
    points = [
        f"<b>• Identity:</b> Formula <b>{p['Molecular Formula']}</b>, molecular weight "
        f"<b>{p['Molecular Weight']}</b> g/mol.",
        f"<b>• Lipophilicity:</b> LogP = <b>{p['LogP']}</b>. "
        f"{get_property_interpretation('LogP', p['LogP'])}",
        f"<b>• Polarity:</b> TPSA = <b>{p['TPSA']}</b> Å². "
        f"{get_property_interpretation('TPSA', p['TPSA'])}",
        f"<b>• Hydrogen bonding:</b> {p['Hydrogen Bond Donors']} donor(s) and "
        f"{p['Hydrogen Bond Acceptors']} acceptor(s).",
        f"<b>• Flexibility:</b> {p['Rotatable Bonds']} rotatable bond(s).",
        f"<b>• Rings and aromaticity:</b> {p['Ring Count']} ring(s), of which "
        f"{p['Aromatic Rings']} are aromatic.",
        f"<b>• Bonding:</b> {p['Sigma (σ) Bonds']} σ bond(s) and {p['Pi (π) Bonds']} π bond(s).",
        f"<b>• Drug-likeness:</b> Lipinski violations = <b>{p['Lipinski Violations']}</b>, "
        f"Veber rule = <b>{p['Veber Rule']}</b>, Ghose filter = <b>{p['Ghose Filter']}</b>, "
        f"QED = <b>{p['QED']}</b>.",
        "<b>• Conclusion:</b> These descriptors should be read together with the 2D/3D "
        "structure to understand composition, polarity, flexibility and drug-likeness.",
    ]
    return points


# ============================================================
# PDF REPORT HELPERS
# ============================================================

def save_2d_structure_png(mol):
    img = Draw.MolToImage(mol, size=(700, 500))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    img.save(tmp.name)
    return tmp.name


def save_3d_structure_png(mol):
    """Clean ball-and-stick projection of the 3D geometry for the PDF."""
    mol3d = generate_3d_molecule(mol)
    if mol3d is None:
        return None

    conf = mol3d.GetConformer()
    n_atoms = mol3d.GetNumAtoms()
    coords = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(n_atoms)],
        dtype=float,
    )

    centered = coords - coords.mean(axis=0)
    if n_atoms >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T
    else:
        projected = centered[:, :2]

    cpk = {"H": "#FFFFFF", "C": "#3F3F3F", "N": "#2F5BFF", "O": "#E53935",
           "F": "#65C466", "P": "#F39C12", "S": "#F4D03F", "Cl": "#27AE60",
           "Br": "#8E3B2F", "I": "#8E44AD"}
    sizes = {"H": 70, "C": 240, "N": 270, "O": 285, "F": 260, "P": 300,
             "S": 310, "Cl": 310, "Br": 330, "I": 350}

    fig, ax = plt.subplots(figsize=(8.8, 6.4), facecolor="white")
    ax.set_facecolor("white")

    for bond in mol3d.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        x1, y1 = projected[i]
        x2, y2 = projected[j]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length == 0:
            continue
        px, py = -dy / length, dx / length
        bt = bond.GetBondType()
        offsets = (-0.035, 0.035) if bt == Chem.BondType.DOUBLE else (
            (-0.055, 0.0, 0.055) if bt == Chem.BondType.TRIPLE else (0.0,))
        for off in offsets:
            ax.plot([x1 + px * off, x2 + px * off], [y1 + py * off, y2 + py * off],
                    color="#707070", linewidth=3.0, solid_capstyle="round", zorder=1)

    for i, atom in enumerate(mol3d.GetAtoms()):
        sym = atom.GetSymbol()
        x, y = projected[i]
        ax.scatter(x, y, s=sizes.get(sym, 250), c=cpk.get(sym, "#A0A0A0"),
                   edgecolors="#222222", linewidths=1.0, zorder=3)
        if sym not in ("H", "C"):
            ax.text(x, y, sym, ha="center", va="center", fontsize=8,
                    fontweight="bold", color="black", zorder=4)

    xmin, ymin = projected.min(axis=0)
    xmax, ymax = projected.max(axis=0)
    pad = 0.22 * max(max(xmax - xmin, 1.0), max(ymax - ymin, 1.0))
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("3D Molecular Structure (Ball-and-Stick)", fontsize=18,
                 fontweight="bold", pad=18)
    fig.tight_layout(pad=1.2)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(tmp.name, dpi=320, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    return tmp.name


def pdf_styles():
    styles = getSampleStyleSheet()
    return {
        "base": styles,
        "title": ParagraphStyle("T", parent=styles["Title"], fontSize=19, leading=23,
                                alignment=TA_CENTER, textColor=HexColor(C_DARK)),
        "heading": ParagraphStyle("H", parent=styles["Heading1"], fontSize=13, leading=16,
                                  textColor=HexColor(C_DARK), spaceBefore=7, spaceAfter=7),
        "body": ParagraphStyle("B", parent=styles["BodyText"], fontSize=9, leading=12,
                               alignment=TA_JUSTIFY),
        "small": ParagraphStyle("S", parent=styles["BodyText"], fontSize=7.2, leading=8.8),
        "header": ParagraphStyle("Hd", parent=styles["BodyText"], fontSize=7.5, leading=9,
                                 textColor=colors.white, alignment=TA_CENTER),
        "inst": ParagraphStyle("I", parent=styles["BodyText"], fontSize=9, leading=11,
                               alignment=TA_CENTER, textColor=HexColor(C_TEAL)),
    }


def pdf_header_block(s, report_title):
    """Logo (top-left, if available), title and institution lines."""
    block = []
    logo_path = find_logo_path()
    if logo_path:
        try:
            iw, ih = ImageReader(logo_path).getSize()
            w = 0.9 * inch
            img = Image(logo_path, width=w, height=w * ih / iw)
            img.hAlign = "LEFT"
            block.append(img)
        except Exception:
            pass
    block += [
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", s["title"]),
        Paragraph(INSTITUTE, s["inst"]),
        Paragraph(FACULTY_LINE, s["inst"]),
        Spacer(1, 0.08 * inch),
        Paragraph(report_title, s["heading"]),
    ]
    return block


def pdf_table_style(header_color=C_TEAL, pad=4):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), pad),
    ])


def developer_credit_text():
    return " &nbsp;|&nbsp; ".join(f"{d['name']} ({d['reg']})" for d in DEVELOPERS)


def generate_molecule_pdf(input_smiles, mol, properties):
    """Single-molecule properties report."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=42, rightMargin=42,
                            topMargin=40, bottomMargin=40)
    s = pdf_styles()
    story = pdf_header_block(s, "MOLECULAR PROPERTIES REPORT")

    ids = get_identifiers(mol)
    basic = [
        ["Input SMILES", input_smiles],
        ["Canonical SMILES", ids["Canonical SMILES"]],
        ["InChIKey", ids["InChIKey"]],
        ["Molecular Formula", properties["Molecular Formula"]],
    ]
    t = Table([[Paragraph(str(a), s["small"]), Paragraph(str(b), s["small"])] for a, b in basic],
              colWidths=[1.8 * inch, 4.9 * inch])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (0, 0), (0, -1), HexColor(C_LIGHT)),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("PADDING", (0, 0), (-1, -1), 5)]))
    story += [t, Spacer(1, 0.15 * inch), Paragraph("Molecular Properties and Interpretation", s["heading"])]

    data = [[Paragraph("<b>Property</b>", s["header"]),
             Paragraph("<b>Value</b>", s["header"]),
             Paragraph("<b>Interpretation</b>", s["header"])]]
    for k, v in properties.items():
        data.append([Paragraph(str(k), s["small"]), Paragraph(str(v), s["small"]),
                     Paragraph(get_property_interpretation(k, v, properties), s["small"])])
    pt = Table(data, colWidths=[1.55 * inch, 0.95 * inch, 4.2 * inch], repeatRows=1)
    pt.setStyle(pdf_table_style())
    story += [pt, Spacer(1, 0.15 * inch), Paragraph("Key Interpretation", s["heading"])]
    for point in build_key_interpretation_points(properties):
        story += [Paragraph(point, s["body"]), Spacer(1, 0.04 * inch)]

    story += [Spacer(1, 0.12 * inch),
              Paragraph(f"<i>Generated by the {APP_TITLE}. Developed by {developer_credit_text()}.</i>",
                        s["small"])]
    doc.build(story)
    data_bytes = buffer.getvalue()
    buffer.close()
    return data_bytes


def generate_multi_final_report(student_name, registration_number, molecule_records):
    """One PDF containing every valid molecule."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=38, rightMargin=38,
                            topMargin=40, bottomMargin=40)
    s = pdf_styles()

    story = pdf_header_block(s, "MULTI-MOLECULE STUDENT ANALYSIS REPORT")
    story += [
        Paragraph(f"Student: {student_name or 'Not provided'} &nbsp;&nbsp;&nbsp; "
                  f"Registration Number: {registration_number or 'Not provided'}", s["body"]),
        Paragraph(f"Number of molecules analyzed: {len(molecule_records)}", s["body"]),
        Spacer(1, 0.12 * inch),
    ]

    temp_images = []
    for index, (input_smiles, mol, props) in enumerate(molecule_records, start=1):
        if index > 1:
            story.append(PageBreak())

        story.append(Paragraph(f"Molecule {index}: Molecular Information", s["heading"]))
        ids = get_identifiers(mol)
        basic = [
            ["Input SMILES", input_smiles],
            ["Canonical SMILES", ids["Canonical SMILES"]],
            ["InChIKey", ids["InChIKey"]],
            ["Molecular Formula", props["Molecular Formula"]],
            ["Molecular Weight", f"{props['Molecular Weight']} g/mol"],
        ]
        bt = Table([[Paragraph(str(a), s["small"]), Paragraph(str(b), s["small"])] for a, b in basic],
                   colWidths=[1.8 * inch, 4.9 * inch])
        bt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                ("BACKGROUND", (0, 0), (0, -1), HexColor(C_LIGHT)),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("PADDING", (0, 0), (-1, -1), 5)]))
        story += [bt, Spacer(1, 0.10 * inch), Paragraph("Molecular Structures", s["heading"])]

        img2d = save_2d_structure_png(mol)
        img3d = save_3d_structure_png(mol)
        temp_images += [p for p in (img2d, img3d) if p]
        if img2d and img3d:
            st_table = Table([[Image(img2d, width=3.05 * inch, height=2.0 * inch),
                               Image(img3d, width=3.05 * inch, height=2.0 * inch)]],
                             colWidths=[3.25 * inch, 3.25 * inch])
            st_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                          ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            story += [st_table, Spacer(1, 0.10 * inch)]
        elif img2d:
            story += [Image(img2d, width=4.8 * inch, height=3.2 * inch), Spacer(1, 0.10 * inch)]

        story.append(Paragraph("Molecular Descriptors and Drug-likeness", s["heading"]))
        data = [[Paragraph("<b>Property</b>", s["header"]),
                 Paragraph("<b>Value</b>", s["header"]),
                 Paragraph("<b>Interpretation</b>", s["header"])]]
        for k, v in props.items():
            data.append([Paragraph(str(k), s["small"]), Paragraph(str(v), s["small"]),
                         Paragraph(get_property_interpretation(k, v, props), s["small"])])
        pt = Table(data, colWidths=[1.55 * inch, 0.9 * inch, 4.0 * inch], repeatRows=1)
        pt.setStyle(pdf_table_style(pad=3))
        story += [pt, Spacer(1, 0.10 * inch), Paragraph("Key Interpretation", s["heading"])]
        for point in build_key_interpretation_points(props):
            story += [Paragraph(point, s["body"]), Spacer(1, 0.04 * inch)]

    # Libraries + code
    story += [PageBreak(), Paragraph("Python Libraries Used and Their Importance", s["heading"])]
    libs = [
        ("Streamlit", "Interactive web application, navigation, input controls and downloads."),
        ("RDKit", "Parses SMILES/InChI/SDF, draws 2D structures, generates 3D conformers, "
                  "and calculates descriptors, drug-likeness metrics, fingerprints and SMARTS matches."),
        ("Pandas", "Organizes descriptor tables, filters compound libraries and exports CSV."),
        ("NumPy and Matplotlib", "Process 3D coordinates and generate plots and report images."),
        ("py3Dmol", "Interactive 3D molecular viewer in the web application."),
        ("ReportLab", "Builds the downloadable PDF reports."),
    ]
    lib_data = [[Paragraph("<b>Library</b>", s["header"]),
                 Paragraph("<b>Importance in this project</b>", s["header"])]]
    for a, b in libs:
        lib_data.append([Paragraph(a, s["small"]), Paragraph(b, s["small"])])
    lt = Table(lib_data, colWidths=[1.8 * inch, 4.9 * inch], repeatRows=1)
    lt.setStyle(pdf_table_style(pad=5))
    story += [lt, Spacer(1, 0.15 * inch), Paragraph("Illustrative Python Code", s["heading"])]

    code_style = ParagraphStyle("Code", parent=s["base"]["Code"], fontName="Courier",
                                fontSize=7, leading=8.5, leftIndent=4, rightIndent=4)
    code = "\n".join([
        "from rdkit import Chem, DataStructs",
        "from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdFingerprintGenerator",
        "",
        'mol_a = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")   # aspirin',
        'mol_b = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")      # paracetamol',
        "",
        "mw   = Descriptors.MolWt(mol_a)",
        "logp = Crippen.MolLogP(mol_a)",
        "hbd  = Lipinski.NumHDonors(mol_a)",
        "hba  = Lipinski.NumHAcceptors(mol_a)",
        "violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])",
        'print("Lipinski violations:", violations, "| QED:", round(QED.qed(mol_a), 3))',
        "",
        "gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)",
        "sim = DataStructs.TanimotoSimilarity(gen.GetFingerprint(mol_a), gen.GetFingerprint(mol_b))",
        'print("Tanimoto similarity:", round(sim, 3))',
    ])
    story += [Preformatted(code, code_style, maxLineLength=92), Spacer(1, 0.08 * inch),
              Paragraph("This compact example shows descriptor calculation, rule-based drug-likeness "
                        "screening and fingerprint similarity. Calculated descriptors are computational "
                        "estimates and should be interpreted alongside experimental data.", s["body"]),
              Spacer(1, 0.12 * inch),
              Paragraph(f"<i>Developed by {developer_credit_text()}.</i>", s["small"])]

    try:
        doc.build(story)
        return buffer.getvalue()
    finally:
        buffer.close()
        for p in temp_images:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# ============================================================
# SIDEBAR
# ============================================================

logo_html = get_logo_html()
if logo_html:
    st.sidebar.markdown(logo_html, unsafe_allow_html=True)

st.sidebar.title("⚗️ Virtual Lab")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📖 Theory",
        "🧬 Molecular Visualization",
        "📊 Descriptors & Drug-likeness",
        "🔍 Substructure Search (SMARTS)",
        "🧩 Fingerprints & Similarity",
        "🌀 Conformer Explorer",
        "🧪 Library Screening",
        "📈 Structure–Property Analysis",
        "📝 Assessment",
        "📄 Final Project Report",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### 👩‍🎓 Student Information")
student_name = st.sidebar.text_input("Student Name", key="student_name")
registration_number = st.sidebar.text_input("Registration Number", key="registration_number")

st.sidebar.markdown(
    """
### 🎯 Students can learn

- SMILES, InChI & InChIKey
- 2D and 3D visualization
- Molecular descriptors
- Drug-likeness (Lipinski, Veber, Ghose, QED)
- ADME-related properties
- SMARTS substructure search
- Fingerprints & Tanimoto similarity
- Library screening
- 3D conformers
- SDF / CSV data handling
- Structure–property relationships
"""
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        f"""
<div class="lab-banner">
  <h1>⚗️ {APP_TITLE}</h1>
  <div class="inst">{INSTITUTE}</div>
  <div class="fac">{FACULTY_LINE}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("## 🎯 Aim")
    st.markdown(
        '<div class="info-card">To build a strong foundation in <b>Cheminformatics and '
        'computer-aided drug design</b> by using computational tools to represent, visualize, '
        'analyze, screen and compare chemical structures, and to relate molecular structure to '
        'physicochemical properties and biological behaviour.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("## 📚 Learning Objectives")
    st.markdown(
        """
1. **Represent molecules** using SMILES, InChI and InChIKey and convert between formats.
2. **Visualize structures** in 2D and 3D, including highlighted functional groups.
3. **Calculate molecular descriptors** (physicochemical, topological and structural).
4. **Evaluate drug-likeness** with Lipinski's Rule of Five, Veber's rules and the Ghose filter.
5. **Predict ADME-related properties** such as absorption and membrane permeability.
6. **Perform substructure searches** using SMARTS patterns.
7. **Compute fingerprints** (Morgan/ECFP, MACCS) and measure **Tanimoto similarity**.
8. **Screen and compare compound libraries**, ranking molecules against property criteria.
9. **Generate and optimize 3D conformers** and understand flexibility.
10. **Understand structure–property relationships.**
11. **Import, clean and export chemical data** in SDF and CSV formats.
"""
    )

    st.markdown("## 🧪 Example Molecules")
    st.dataframe(
        pd.DataFrame({"Molecule": list(EXAMPLE_MOLECULES.keys()),
                      "SMILES": list(EXAMPLE_MOLECULES.values())}),
        hide_index=True,
    )

    st.divider()
    st.markdown("## 👩‍💻 Developed by")
    dev_cols = st.columns(len(DEVELOPERS))
    for col, dev in zip(dev_cols, DEVELOPERS):
        col.markdown(
            f'<div class="dev-card"><div class="nm">{dev["name"]}</div>'
            f'<div class="rg">Reg. No: {dev["reg"]}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption(f"{FACULTY_LINE} — {INSTITUTE}")


# ============================================================
# THEORY
# ============================================================

elif page == "📖 Theory":

    page_header("📖 Theory", "Key concepts behind the virtual laboratory")

    with st.expander("What is Cheminformatics?", expanded=True):
        st.markdown(
            """
Cheminformatics combines **Chemistry ⚗️, Computer Science 💻 and Data Analysis 📊** to store,
process, visualize and analyze chemical information, and is central to modern drug discovery.
"""
        )

    with st.expander("Molecular representations: SMILES, InChI, InChIKey"):
        st.markdown(
            """
**SMILES** (Simplified Molecular Input Line Entry System) writes a structure as a text string.
**InChI** is a standardized, layered identifier and **InChIKey** is its fixed-length hashed form,
useful for database searching.

| Molecule | SMILES |
|---|---|
| Ethanol | CCO |
| Benzene | c1ccccc1 |
| Acetic Acid | CC(=O)O |
"""
        )

    with st.expander("Molecular descriptors"):
        st.markdown(
            """
| Descriptor | Meaning |
|---|---|
| Molecular Weight | Molecular size |
| LogP | Lipophilicity (water/fat balance) |
| Molar Refractivity | Volume and polarizability |
| TPSA | Polar surface area; permeability |
| HBD / HBA | Hydrogen-bond donors / acceptors |
| Rotatable Bonds | Flexibility |
| Aromatic Ring Count, Fsp3 | Aromaticity and 3D character |
| QED | Overall drug-likeness (0–1) |
"""
        )

    with st.expander("Drug-likeness rules"):
        st.markdown(
            """
- **Lipinski's Rule of Five:** MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10 (at most one violation is tolerated).
- **Veber's rules:** Rotatable bonds ≤ 10 and TPSA ≤ 140 Å².
- **Ghose filter:** MW 160–480, LogP –0.4 to 5.6, MR 40–130, 20–70 atoms.
- **QED:** Combines several properties into one score between 0 and 1.
"""
        )

    with st.expander("SMARTS, fingerprints and similarity"):
        st.markdown(
            """
**SMARTS** is a pattern language used to search for substructures such as functional groups.
**Fingerprints** encode a molecule as a bit string; **Morgan (ECFP)** captures circular atom
environments and **MACCS** uses 166 predefined structural keys. The **Tanimoto coefficient**
(shared bits ÷ total bits) ranges from 0 (different) to 1 (identical).
"""
        )

    with st.expander("3D models and conformers"):
        st.markdown(
            """
**Ball and Stick** shows atoms and bonds, **Stick** emphasizes bonding, **Space Filling** shows
approximate volume and **Wireframe** gives a simplified view. A molecule can adopt many
**conformers**; lower-energy conformers are more stable and more likely to be populated.
"""
        )


# ============================================================
# MOLECULAR VISUALIZATION
# ============================================================

elif page == "🧬 Molecular Visualization":

    page_header("🧬 Molecular Visualization", "Representations, 2D drawing and interactive 3D models")
    smiles = molecule_input("viz")
    mol = get_molecule(smiles)

    if mol is None:
        st.error("❌ Invalid SMILES notation.")
    else:
        st.success("✅ Valid molecular structure!")
        st.session_state.latest_smiles = smiles

        ids = get_identifiers(mol)
        bond_info = calculate_bond_information(mol)

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("2D Molecular Structure")
            st.image(generate_2d_image(mol))
        with col2:
            st.subheader("Molecular Representations")
            st.write("**Input SMILES:**", smiles)
            st.write("**Canonical SMILES:**", ids["Canonical SMILES"])
            st.write("**Formula:**", rdMolDescriptors.CalcMolFormula(mol))
            st.write("**InChIKey:**", ids["InChIKey"])
            st.text_area("InChI", ids["InChI"], height=90)
            st.write("**Heavy atoms:**", mol.GetNumHeavyAtoms())
            st.write("**Total atoms (including H):**", bond_info["Total Atoms Including H"])
            st.write("**σ bonds:**", bond_info["Sigma Bonds"], "| **π bonds:**", bond_info["Pi Bonds"])

        mol_for_dl = Chem.Mol(mol)
        mol_for_dl.SetProp("_Name", "Molecule")
        st.download_button("⬇️ Download structure (MOL file)",
                           data=Chem.MolToMolBlock(mol_for_dl),
                           file_name="molecule.mol", mime="chemical/x-mdl-molfile")

        with st.expander("🔁 Convert InChI → SMILES"):
            inchi_in = st.text_input("Enter an InChI string", key="inchi_input",
                                     placeholder="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
            if inchi_in:
                m_from_inchi = Chem.MolFromInchi(inchi_in.strip())
                if m_from_inchi is None:
                    st.error("Could not interpret this InChI.")
                else:
                    st.success("Converted SMILES: " + Chem.MolToSmiles(m_from_inchi))
                    st.image(generate_2d_image(m_from_inchi, size=(450, 320)))

        st.divider()
        st.subheader("🌐 Interactive 3D Molecular Structure")
        style = st.selectbox("Select Molecular Model",
                             ["Ball and Stick", "Stick", "Space Filling", "Wireframe", "Cartoon"])
        surface = st.checkbox("Show van der Waals Surface")

        if st.button("🚀 Generate 3D Structure"):
            with st.spinner("Generating 3D molecular structure..."):
                mol3d = generate_3d_molecule(mol)
            if mol3d is None:
                st.error("Unable to generate the 3D structure.")
            else:
                display_3d_structure(mol3d, style, surface)


# ============================================================
# DESCRIPTORS AND DRUG-LIKENESS
# ============================================================

elif page == "📊 Descriptors & Drug-likeness":

    page_header("📊 Descriptors & Drug-likeness",
                "Complete descriptor report with Lipinski, Veber, Ghose and QED evaluation")
    smiles = molecule_input("desc")

    if smiles:
        mol = get_molecule(smiles)
        if mol is None:
            st.error("❌ Invalid SMILES notation. Please check the structure.")
        else:
            st.success("✅ Valid molecular structure detected!")
            props = calculate_properties(mol)
            ids = get_identifiers(mol)
            st.session_state.latest_smiles = smiles
            st.session_state.latest_properties = props

            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.subheader("🧬 2D Structure")
                st.image(generate_2d_image(mol))
            with col2:
                st.subheader("📋 Basic Information")
                st.write("**Canonical SMILES:**", ids["Canonical SMILES"])
                st.write("**InChIKey:**", ids["InChIKey"])
                st.write("**Formula:**", props["Molecular Formula"])
                st.write("**Heavy atoms:**", props["Heavy Atoms"])
                st.write("**σ bonds:**", props["Sigma (σ) Bonds"],
                         "| **π bonds:**", props["Pi (π) Bonds"])
                st.write("**Formal charge:**", props["Formal Charge"])

            st.divider()
            st.subheader("📊 Key Descriptors")
            c = st.columns(4)
            c[0].metric("Molecular Weight", f"{props['Molecular Weight']} g/mol")
            c[1].metric("LogP", props["LogP"])
            c[2].metric("TPSA", f"{props['TPSA']} Å²")
            c[3].metric("Molar Refractivity", props["Molar Refractivity"])
            c = st.columns(4)
            c[0].metric("H-Bond Donors", props["Hydrogen Bond Donors"])
            c[1].metric("H-Bond Acceptors", props["Hydrogen Bond Acceptors"])
            c[2].metric("Rotatable Bonds", props["Rotatable Bonds"])
            c[3].metric("Fraction Csp3", props["Fraction Csp3"])
            c = st.columns(4)
            c[0].metric("Ring Count", props["Ring Count"])
            c[1].metric("Aromatic Rings", props["Aromatic Rings"])
            c[2].metric("Heavy Atoms", props["Heavy Atoms"])
            c[3].metric("Formal Charge", props["Formal Charge"])

            st.divider()
            st.subheader("💊 Drug-likeness Evaluation")

            violations, checks = lipinski_violations(
                props["Molecular Weight"], props["LogP"],
                props["Hydrogen Bond Donors"], props["Hydrogen Bond Acceptors"])

            lc1, lc2 = st.columns([1.2, 1])
            with lc1:
                st.markdown("**Lipinski's Rule of Five**")
                st.dataframe(
                    pd.DataFrame([{"Criterion": n, "Value": round(float(v), 3),
                                   "Result": "✅ Pass" if ok else "❌ Violation"}
                                  for n, v, ok in checks]),
                    hide_index=True)
                if violations == 0:
                    st.success("No violations: the molecule complies with the Rule of Five.")
                elif violations == 1:
                    st.info("One violation: still generally acceptable for oral drugs.")
                else:
                    st.warning(f"{violations} violations: poor oral bioavailability is likely.")
            with lc2:
                st.metric("Veber Rule", props["Veber Rule"])
                st.metric("Ghose Filter", props["Ghose Filter"])
                st.metric("QED (0–1)", props["QED"])
                qed_val = props["QED"]
                if isinstance(qed_val, float) and not np.isnan(qed_val):
                    st.progress(float(min(max(qed_val, 0.0), 1.0)))

            st.divider()
            st.subheader("📑 Complete Descriptor Table")
            report_rows = [["Input SMILES", smiles], ["Canonical SMILES", ids["Canonical SMILES"]]]
            report_rows += [[k, str(v)] for k, v in props.items()]
            report_df = pd.DataFrame(report_rows, columns=["Property", "Value"])
            st.dataframe(report_df, hide_index=True)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("⬇️ Download Descriptors (CSV)",
                                   data=report_df.to_csv(index=False).encode("utf-8"),
                                   file_name="Molecular_Descriptors.csv", mime="text/csv")
            with dl2:
                if st.button("📄 Generate Molecular PDF Report"):
                    with st.spinner("Generating PDF report..."):
                        pdf = generate_molecule_pdf(smiles, mol, props)
                    st.download_button("⬇️ Download PDF", data=pdf,
                                       file_name="Molecular_Properties_Report.pdf",
                                       mime="application/pdf")

            st.divider()
            st.subheader("🌐 3D Visualization")
            model_style = st.selectbox("Select 3D Model",
                                       ["Ball and Stick", "Stick", "Space Filling", "Wireframe", "Cartoon"],
                                       key="desc_3d_style")
            surface = st.checkbox("Show van der Waals Surface", key="desc_surface")
            if st.button("🚀 Generate 3D Molecular Model", key="desc_generate_3d"):
                with st.spinner("Generating 3D model..."):
                    mol3d = generate_3d_molecule(mol)
                if mol3d is None:
                    st.error("Unable to generate the 3D structure.")
                else:
                    display_3d_structure(mol3d, model_style, surface)


# ============================================================
# SUBSTRUCTURE SEARCH
# ============================================================

elif page == "🔍 Substructure Search (SMARTS)":

    page_header("🔍 Substructure Search (SMARTS)",
                "Find functional groups and highlight them in the structure")
    smiles = molecule_input("sub")
    mol = get_molecule(smiles)

    if mol is None:
        st.error("❌ Invalid SMILES notation.")
    else:
        st.subheader("1. Functional Group Scan")
        scan_rows = []
        for gname, smarts in FUNCTIONAL_GROUPS.items():
            patt = Chem.MolFromSmarts(smarts)
            count = len(mol.GetSubstructMatches(patt)) if patt is not None else 0
            scan_rows.append({"Functional Group": gname, "SMARTS": smarts,
                              "Present": "✅ Yes" if count else "—", "Count": count})
        scan_df = pd.DataFrame(scan_rows)
        present = scan_df[scan_df["Count"] > 0]
        if present.empty:
            st.info("None of the listed functional groups were detected.")
        else:
            st.success("Detected: " + ", ".join(present["Functional Group"]))
        st.dataframe(scan_df, hide_index=True)

        st.divider()
        st.subheader("2. Highlight a Pattern")
        mode = st.radio("Pattern source", ["Choose functional group", "Custom SMARTS"], horizontal=True)
        if mode == "Choose functional group":
            chosen = st.selectbox("Functional group", list(FUNCTIONAL_GROUPS.keys()))
            smarts_query = FUNCTIONAL_GROUPS[chosen]
            st.code(smarts_query)
        else:
            smarts_query = st.text_input("Enter SMARTS", value="[CX3]=[OX1]")

        patt = Chem.MolFromSmarts(smarts_query) if smarts_query else None
        if patt is None:
            st.error("Invalid SMARTS pattern.")
        else:
            matches = mol.GetSubstructMatches(patt)
            if not matches:
                st.warning("No match found in this molecule.")
                st.image(generate_2d_image(mol))
            else:
                atoms = sorted({a for m in matches for a in m})
                st.success(f"{len(matches)} match(es) found; highlighted atoms: {atoms}")
                st.image(generate_2d_image(mol, highlight_atoms=atoms))


# ============================================================
# FINGERPRINTS AND SIMILARITY
# ============================================================

elif page == "🧩 Fingerprints & Similarity":

    page_header("🧩 Fingerprints & Similarity",
                "Encode molecules as bit vectors and measure Tanimoto similarity")

    st.markdown("**Molecule A (query)**")
    smiles_a = molecule_input("fp")
    smiles_b = st.text_input("Molecule B (SMILES)", value="CC(=O)Nc1ccc(O)cc1", key="fp_b")

    fp_type = st.radio("Fingerprint type", ["Morgan (ECFP)", "MACCS Keys", "RDKit Topological"],
                       horizontal=True)
    radius, n_bits = 2, 2048
    if fp_type == "Morgan (ECFP)":
        fc1, fc2 = st.columns(2)
        radius = fc1.slider("Morgan radius (2 = ECFP4)", 1, 4, 2)
        n_bits = fc2.select_slider("Bit length", options=[512, 1024, 2048, 4096], value=2048)

    mol_a, mol_b = get_molecule(smiles_a), get_molecule(smiles_b)

    if mol_a is None or mol_b is None:
        st.error("❌ One or both SMILES are invalid.")
    else:
        fp_a = get_fingerprint(mol_a, fp_type, radius, n_bits)
        fp_b = get_fingerprint(mol_b, fp_type, radius, n_bits)
        tani = DataStructs.TanimotoSimilarity(fp_a, fp_b)
        dice = DataStructs.DiceSimilarity(fp_a, fp_b)

        ic1, ic2 = st.columns(2)
        ic1.image(generate_2d_image(mol_a, size=(450, 330)), caption="Molecule A")
        ic2.image(generate_2d_image(mol_b, size=(450, 330)), caption="Molecule B")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tanimoto", f"{tani:.3f}")
        m2.metric("Dice", f"{dice:.3f}")
        m3.metric("Bits ON (A)", fp_a.GetNumOnBits())
        m4.metric("Bits ON (B)", fp_b.GetNumOnBits())
        st.progress(float(tani))
        st.info(f"**Interpretation:** {similarity_label(tani)}")

        st.divider()
        st.subheader("Similarity ranking against example molecules")
        rank_rows = []
        for nm, smi in EXAMPLE_MOLECULES.items():
            m = get_molecule(smi)
            if m is None:
                continue
            rank_rows.append({"Molecule": nm, "SMILES": smi,
                              "Tanimoto vs A": round(DataStructs.TanimotoSimilarity(
                                  fp_a, get_fingerprint(m, fp_type, radius, n_bits)), 3)})
        rank_df = pd.DataFrame(rank_rows).sort_values("Tanimoto vs A", ascending=False)
        st.dataframe(rank_df, hide_index=True)


# ============================================================
# CONFORMER EXPLORER
# ============================================================

elif page == "🌀 Conformer Explorer":

    page_header("🌀 Conformer Explorer",
                "Generate, optimize and compare 3D conformers by energy")
    smiles = molecule_input("conf")
    mol = get_molecule(smiles)

    if mol is None:
        st.error("❌ Invalid SMILES notation.")
    else:
        n_confs = st.slider("Number of conformers", 1, 30, 10)

        if st.button("🌀 Generate Conformers"):
            with st.spinner("Embedding and optimizing conformers..."):
                molh, df = generate_conformers(mol, n_confs)
            if molh is None:
                st.error("Unable to generate conformers for this molecule.")
            else:
                st.session_state.conf_mol = molh
                st.session_state.conf_df = df
                st.session_state.conf_smiles = smiles

        if st.session_state.get("conf_df") is not None and st.session_state.get("conf_smiles") == smiles:
            df = st.session_state.conf_df
            molh = st.session_state.conf_mol
            st.success(f"{len(df)} conformer(s) generated (force field: {df.attrs.get('method', 'MMFF94')}).")
            st.dataframe(df, hide_index=True)

            if "Relative Energy (kcal/mol)" in df.columns:
                st.bar_chart(df.set_index("Conformer ID")["Relative Energy (kcal/mol)"])
                best_row = df.loc[df["Energy (kcal/mol)"].idxmin()]
                st.info(f"Lowest-energy conformer: **ID {int(best_row['Conformer ID'])}** "
                        f"({best_row['Energy (kcal/mol)']} kcal/mol). Lower energy means a more stable geometry.")

            cid = st.selectbox("Select conformer to view", list(df["Conformer ID"]))
            style = st.selectbox("3D model", ["Ball and Stick", "Stick", "Space Filling", "Wireframe"],
                                 key="conf_style")
            display_3d_structure(molh, style, False, conf_id=int(cid))


# ============================================================
# LIBRARY SCREENING
# ============================================================

elif page == "🧪 Library Screening":

    page_header("🧪 Library Screening",
                "Import compound libraries, filter by drug-likeness and export results")

    source = st.radio("Data source", ["Paste (Name, SMILES)", "Upload CSV", "Upload SDF"], horizontal=True)
    records = []  # (name, smiles)
    skipped = 0

    if source == "Paste (Name, SMILES)":
        default_lib = "\n".join(f"{n},{s}" for n, s in EXAMPLE_MOLECULES.items())
        text = st.text_area("One compound per line: Name, SMILES", value=default_lib, height=220)
        for i, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            if "," in line:
                n, s = line.split(",", 1)
            else:
                n, s = f"Compound_{i}", line
            records.append((n.strip(), s.strip()))

    elif source == "Upload CSV":
        up = st.file_uploader("Upload a CSV with a SMILES column", type=["csv"])
        if up is not None:
            try:
                raw = pd.read_csv(up)
                smi_col = next((c for c in raw.columns
                                if c.lower() in ("smiles", "smile", "canonical_smiles")), None)
                name_col = next((c for c in raw.columns
                                 if c.lower() in ("name", "id", "molecule", "compound", "title")), None)
                if smi_col is None:
                    st.error("No SMILES column found (expected a column named 'SMILES').")
                else:
                    for i, row in raw.iterrows():
                        nm = str(row[name_col]) if name_col else f"Compound_{i + 1}"
                        records.append((nm, str(row[smi_col])))
            except Exception as e:
                st.error(f"Could not read the CSV file: {e}")

    else:
        up = st.file_uploader("Upload an SDF file", type=["sdf", "sd"])
        if up is not None:
            supplier = Chem.ForwardSDMolSupplier(BytesIO(up.getvalue()))
            for i, m in enumerate(supplier, start=1):
                if m is None:
                    skipped += 1
                    continue
                nm = m.GetProp("_Name") if m.HasProp("_Name") and m.GetProp("_Name") else f"Compound_{i}"
                records.append((nm, Chem.MolToSmiles(m)))

    if len(records) > 500:
        st.warning("Only the first 500 compounds are processed.")
        records = records[:500]

    if records:
        rows = []
        for nm, smi in records:
            m = get_molecule(smi)
            if m is None:
                skipped += 1
                continue
            rows.append(library_row(nm, smi, m))

        if skipped:
            st.warning(f"{skipped} entr(y/ies) could not be parsed and were skipped.")

        if rows:
            lib_df = pd.DataFrame(rows)
            st.success(f"{len(lib_df)} valid compound(s) processed.")

            st.subheader("🎛️ Filter Criteria")
            f1, f2, f3 = st.columns(3)
            max_viol = f1.selectbox("Max Lipinski violations", [0, 1, 2, 3, 4], index=1)
            need_veber = f2.checkbox("Require Veber pass", value=False)
            min_qed = f3.slider("Minimum QED", 0.0, 1.0, 0.0, 0.05)
            mw_lo, mw_hi = st.slider("Molecular weight range (g/mol)", 0, 1000, (0, 600))

            filt = lib_df[
                (lib_df["Lipinski Violations"] <= max_viol)
                & (lib_df["QED"] >= min_qed)
                & (lib_df["MW"].between(mw_lo, mw_hi))
            ]
            if need_veber:
                filt = filt[filt["Veber"] == "Pass"]

            sort_by = st.selectbox("Rank by", ["QED", "MW", "LogP", "TPSA", "Lipinski Violations"])
            ascending = st.checkbox("Ascending order", value=(sort_by == "Lipinski Violations"))
            filt = filt.sort_values(sort_by, ascending=ascending)

            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Total compounds", len(lib_df))
            sm2.metric("Pass filters", len(filt))
            sm3.metric("Pass rate", f"{100 * len(filt) / len(lib_df):.0f}%")

            st.dataframe(filt, hide_index=True)

            dl1, dl2 = st.columns(2)
            dl1.download_button("⬇️ Download filtered (CSV)",
                                data=filt.to_csv(index=False).encode("utf-8"),
                                file_name="screened_library.csv", mime="text/csv")
            dl2.download_button("⬇️ Download filtered (SDF)", data=dataframe_to_sdf(filt),
                                file_name="screened_library.sdf", mime="chemical/x-mdl-sdfile")

            st.subheader("📉 Chemical Space: MW vs LogP")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter(lib_df["MW"], lib_df["LogP"], c="#BBBBBB", s=50, label="All compounds")
            ax.scatter(filt["MW"], filt["LogP"], c=C_TEAL, s=70, edgecolors=C_DARK, label="Pass filters")
            ax.axvline(500, color=C_AMBER, linestyle="--", linewidth=1)
            ax.axhline(5, color=C_AMBER, linestyle="--", linewidth=1)
            ax.set_xlabel("Molecular Weight (g/mol)")
            ax.set_ylabel("LogP")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Dashed lines mark the Lipinski limits (MW = 500, LogP = 5).")


# ============================================================
# STRUCTURE–PROPERTY ANALYSIS
# ============================================================

elif page == "📈 Structure–Property Analysis":

    page_header("📈 Structure–Property Analysis",
                "Compare molecules and see how structure changes properties")
    st.write("Enter one molecule per line in the format: **Molecule Name, SMILES**")

    default_data = "\n".join(f"{n},{EXAMPLE_MOLECULES[n]}" for n in
                             ["Ethanol", "Benzene", "Phenol", "Acetic Acid", "Aspirin", "Paracetamol"])
    molecule_input_text = st.text_area("Enter Molecules", value=default_data, height=200)

    rows = []
    for line in molecule_input_text.splitlines():
        if "," not in line:
            continue
        name, smi = line.split(",", 1)
        m = get_molecule(smi)
        if m:
            p = calculate_properties(m)
            rows.append({
                "Molecule": name.strip(),
                "MW": p["Molecular Weight"],
                "LogP": p["LogP"],
                "TPSA": p["TPSA"],
                "HBD": p["Hydrogen Bond Donors"],
                "HBA": p["Hydrogen Bond Acceptors"],
                "Rotatable Bonds": p["Rotatable Bonds"],
                "Ring Count": p["Ring Count"],
                "Aromatic Rings": p["Aromatic Rings"],
                "Fsp3": p["Fraction Csp3"],
                "QED": p["QED"],
                "Lipinski Violations": p["Lipinski Violations"],
            })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)
        st.download_button("⬇️ Download Comparison Table (CSV)",
                           data=df.to_csv(index=False).encode("utf-8"),
                           file_name="structure_property_comparison.csv", mime="text/csv")

        numeric_cols = [c for c in df.columns if c != "Molecule"]
        st.subheader("📊 Property Bar Chart")
        prop_choice = st.selectbox("Property", numeric_cols, index=1)
        st.bar_chart(df.set_index("Molecule")[prop_choice])

        st.subheader("🔵 Property Scatter Plot")
        sc1, sc2 = st.columns(2)
        x_prop = sc1.selectbox("X axis", numeric_cols, index=0)
        y_prop = sc2.selectbox("Y axis", numeric_cols, index=1)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(df[x_prop], df[y_prop], c=C_TEAL, s=80, edgecolors=C_DARK)
        for _, r in df.iterrows():
            ax.annotate(r["Molecule"], (r[x_prop], r[y_prop]), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
        ax.set_xlabel(x_prop)
        ax.set_ylabel(y_prop)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        if len(df) >= 3 and df[x_prop].nunique() > 1 and df[y_prop].nunique() > 1:
            corr = float(np.corrcoef(df[x_prop], df[y_prop])[0, 1])
            st.info(f"Correlation between **{x_prop}** and **{y_prop}**: r = {corr:.2f}")
    else:
        st.warning("Please enter at least one valid molecule in the format: Molecule Name, SMILES")


# ============================================================
# ASSESSMENT
# ============================================================

elif page == "📝 Assessment":

    page_header("📝 Cheminformatics Assessment", "Test your understanding of the laboratory concepts")

    questions = [
        {"question": "What is the full form of SMILES?",
         "options": ["Simplified Molecular Input Line Entry System",
                     "Standard Molecular Information Language Encoding System",
                     "Simple Molecular Identification and Labeling System"],
         "answer": "Simplified Molecular Input Line Entry System"},
        {"question": "Which SMILES represents ethanol?",
         "options": ["CCO", "c1ccccc1", "CC(=O)O"], "answer": "CCO"},
        {"question": "Which SMILES represents benzene?",
         "options": ["c1ccccc1", "CCO", "O=C=O"], "answer": "c1ccccc1"},
        {"question": "In SMILES, parentheses are mainly used to indicate:",
         "options": ["Branches", "Molecular weight", "Atom colour"], "answer": "Branches"},
        {"question": "What is an InChIKey?",
         "options": ["A fixed-length hashed form of the InChI identifier",
                     "A 3D coordinate file", "A type of fingerprint"],
         "answer": "A fixed-length hashed form of the InChI identifier"},
        {"question": "Which property is commonly associated with lipophilicity?",
         "options": ["LogP", "TPSA", "HBD"], "answer": "LogP"},
        {"question": "What does TPSA describe?",
         "options": ["Topological Polar Surface Area", "Total Pi Surface Area",
                     "Thermal Property Surface Analysis"],
         "answer": "Topological Polar Surface Area"},
        {"question": "A double bond contains how many pi (π) bonds?",
         "options": ["1", "2", "0"], "answer": "1"},
        {"question": "A triple bond contains how many pi (π) bonds?",
         "options": ["2", "1", "3"], "answer": "2"},
        {"question": "According to Lipinski's Rule of Five, the molecular weight limit is:",
         "options": ["≤ 500 g/mol", "≤ 1000 g/mol", "≤ 100 g/mol"], "answer": "≤ 500 g/mol"},
        {"question": "Lipinski's rule allows at most how many hydrogen-bond donors?",
         "options": ["5", "10", "2"], "answer": "5"},
        {"question": "Veber's rules relate oral bioavailability to:",
         "options": ["Rotatable bonds and TPSA", "Ring count and charge", "Formula and colour"],
         "answer": "Rotatable bonds and TPSA"},
        {"question": "What is the range of the QED score?",
         "options": ["0 to 1", "0 to 100", "-5 to 5"], "answer": "0 to 1"},
        {"question": "SMARTS patterns are mainly used for:",
         "options": ["Substructure searching", "Drawing 3D models", "Measuring pH"],
         "answer": "Substructure searching"},
        {"question": "Which fingerprint uses 166 predefined structural keys?",
         "options": ["MACCS keys", "Morgan (ECFP)", "InChIKey"], "answer": "MACCS keys"},
        {"question": "A Tanimoto coefficient of 1.0 means the two fingerprints are:",
         "options": ["Identical", "Completely different", "Half similar"], "answer": "Identical"},
        {"question": "A higher Fraction Csp3 generally indicates:",
         "options": ["A more saturated, three-dimensional molecule",
                     "A more planar aromatic molecule", "A charged molecule"],
         "answer": "A more saturated, three-dimensional molecule"},
        {"question": "Which 3D model best shows approximate molecular volume?",
         "options": ["Space Filling", "Wireframe", "Text SMILES"], "answer": "Space Filling"},
        {"question": "Among several conformers, the most stable one has the:",
         "options": ["Lowest energy", "Highest energy", "Most rotatable bonds"],
         "answer": "Lowest energy"},
        {"question": "Which file format is commonly used to store many molecules with properties?",
         "options": ["SDF", "PDF", "MP4"], "answer": "SDF"},
    ]

    student_answers = []
    for i, q in enumerate(questions):
        ans = st.radio(f"{i + 1}. {q['question']}", q["options"], index=None, key=f"assessment_v3_{i}")
        student_answers.append(ans)

    if st.button("Submit Assessment"):
        if any(a is None for a in student_answers):
            st.warning("Please answer all questions before submitting the assessment.")
            st.stop()

        score = sum(1 for i, q in enumerate(questions) if student_answers[i] == q["answer"])
        pct = score / len(questions) * 100
        st.success(f"Your Score: {score}/{len(questions)}")
        st.metric("Percentage", f"{pct:.1f}%")

        if pct >= 80:
            st.balloons()
            st.success("🎉 Excellent performance!")
        elif pct >= 50:
            st.info("👍 Good! Continue practicing.")
        else:
            st.warning("📚 Please review the Theory section and try again.")

        wrong = [(i + 1, q["question"], q["answer"]) for i, q in enumerate(questions)
                 if student_answers[i] != q["answer"]]
        if wrong:
            with st.expander("Review incorrect answers"):
                for num, qt, ans in wrong:
                    st.write(f"**Q{num}.** {qt}  \n✅ Correct answer: **{ans}**")


# ============================================================
# FINAL PROJECT REPORT
# ============================================================

elif page == "📄 Final Project Report":

    page_header("📄 Final Student Molecular Analysis Report",
                "One downloadable PDF covering every valid molecule you enter")
    st.write("Enter one SMILES string per line. The report will include every valid molecule "
             "in a single PDF, with descriptors and drug-likeness evaluation.")

    name = st.text_input("Student Name for Report",
                         value=st.session_state.get("student_name", ""), key="final_report_name")
    reg = st.text_input("Registration Number for Report",
                        value=st.session_state.get("registration_number", ""), key="final_report_reg")

    default_smiles = st.session_state.get("latest_smiles", st.session_state.shared_smiles)
    multi_text = st.text_area("Enter SMILES (one molecule per line)",
                              value=st.session_state.get("final_report_smiles_multi", default_smiles),
                              height=150, key="final_report_smiles_multi",
                              help="Example: CCO on the first line, c1ccccc1 on the second line.")

    entries = [ln.strip() for ln in multi_text.splitlines() if ln.strip()]
    molecule_records, invalid = [], []
    for item in entries:
        m = get_molecule(item)
        if m is None:
            invalid.append(item)
            continue
        molecule_records.append((item, m, calculate_properties(m)))

    if invalid:
        st.warning("These SMILES could not be interpreted and will not be included: " + ", ".join(invalid))

    if molecule_records:
        st.success(f"{len(molecule_records)} valid molecule(s) ready for the report.")
        preview = [{"No.": i, "Input SMILES": smi, "Formula": p["Molecular Formula"],
                    "MW (g/mol)": p["Molecular Weight"], "Lipinski Violations": p["Lipinski Violations"],
                    "QED": p["QED"]}
                   for i, (smi, _, p) in enumerate(molecule_records, start=1)]
        st.dataframe(pd.DataFrame(preview), hide_index=True)

        if st.button("📄 Generate Single PDF for All Molecules"):
            with st.spinner(f"Generating PDF for {len(molecule_records)} molecule(s)..."):
                pdf = generate_multi_final_report(name, reg, molecule_records)
            st.success("✅ Combined molecular report generated successfully!")
            st.download_button("⬇️ Download Combined Molecular Report PDF", data=pdf,
                               file_name="Cheminformatics_Multi_Molecule_Report.pdf",
                               mime="application/pdf", key="download_multi_molecule_pdf")
    else:
        st.info("Enter at least one valid SMILES string to generate the report.")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    f"⚗️ {APP_TITLE} | {INSTITUTE} | Developed by "
    + " & ".join(f"{d['name']} ({d['reg']})" for d in DEVELOPERS)
    + " using Python, Streamlit and RDKit"
)
