from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "random_forest_clinvar.pkl"
LOOKUP_PATH = DATA_DIR / "variant_lookup.csv"
DATASET_PATH = DATA_DIR / "clinvar_labeled.csv"

REQUIRED_FEATURES = [
    "Gene", "Variant type", "Molecular consequence",
    "Protein_Position", "Original_AA", "New_AA",
]
LOOKUP_REQUIRED_COLUMNS = [
    "VariationID", "Accession", "Gene", "Name",
    "Protein change", "Germline classification", "Label",
    "Variant type", "Molecular consequence", "Protein_Position",
    "Original_AA", "New_AA",
]

GENES = ["CFTR", "BRCA1", "TP53", "HBB", "LDLR", "PAH"]
GENE_INFO = {
    "CFTR": "Cystic fibrosis",
    "BRCA1": "Hereditary breast and ovarian cancer susceptibility",
    "TP53": "Li-Fraumeni syndrome",
    "HBB": "Sickle cell disease and beta-thalassemia",
    "LDLR": "Familial hypercholesterolemia",
    "PAH": "Phenylketonuria",
}
AMINO_ACIDS = {
    "A": "Alanine", "R": "Arginine", "N": "Asparagine", "D": "Aspartic acid",
    "C": "Cysteine", "E": "Glutamic acid", "Q": "Glutamine", "G": "Glycine",
    "H": "Histidine", "I": "Isoleucine", "L": "Leucine", "K": "Lysine",
    "M": "Methionine", "F": "Phenylalanine", "P": "Proline", "S": "Serine",
    "T": "Threonine", "W": "Tryptophan", "Y": "Tyrosine", "V": "Valine",
}
AMINO_ACID_CLASSES = {
    "A": "Nonpolar", "V": "Nonpolar", "L": "Nonpolar", "I": "Nonpolar", "M": "Nonpolar",
    "G": "Nonpolar", "P": "Nonpolar", "F": "Aromatic", "W": "Aromatic", "Y": "Aromatic",
    "S": "Polar", "T": "Polar", "C": "Polar", "N": "Polar", "Q": "Polar",
    "K": "Positively charged", "R": "Positively charged", "H": "Positively charged",
    "D": "Negatively charged", "E": "Negatively charged",
}

# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Genetic Variant Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 3. CUSTOM DESIGN
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{ --ink:#071311; --panel:#0f211d; --line:#1f3a34; --text:#e8f3ef; --mute:#8eaaa1;
  --benign:#2dd4a7; --path:#ff647c; --amber:#f5b942; }
html, body, .stApp, [class*="css"]{ font-family:'IBM Plex Sans',sans-serif; }
.stApp{ background:radial-gradient(circle at 10% 0%, rgba(0,210,180,.09), transparent 30%), var(--ink); color:var(--text); }
.main .block-container{ max-width:1200px; padding-top:1.6rem; padding-bottom:4rem; }
h1,h2,h3,h4{ font-family:'Bricolage Grotesque',sans-serif !important; color:#effff9 !important; letter-spacing:-.01em; }
#MainMenu, footer, header{ visibility:hidden; }

.hero{ padding:32px 34px 28px; border:1px solid rgba(87,221,191,.20); border-radius:20px;
  background:linear-gradient(120deg, rgba(11,42,36,.95), rgba(8,24,23,.95)); margin-bottom:20px; }
.hero-title{ font-family:'Bricolage Grotesque',sans-serif; font-size:2.5rem; font-weight:800; letter-spacing:-.02em; margin:0; color:#edfff9; }
.hero-subtitle{ color:#9bbab0; font-size:1rem; margin:8px 0 0; max-width:640px; }
.bases{ display:flex; gap:5px; margin-top:22px; }
.bases span{ width:26px; height:7px; border-radius:4px; opacity:.25; animation:scan 3.2s ease-in-out infinite; }
.bases span:nth-child(4n+1){ background:#2dd4a7; } .bases span:nth-child(4n+2){ background:#f5b942; }
.bases span:nth-child(4n+3){ background:#ff647c; } .bases span:nth-child(4n+4){ background:#6aa7ff; }
@keyframes scan{ 0%,100%{opacity:.2; transform:scaleY(1);} 50%{opacity:1; transform:scaleY(1.9);} }
@media (prefers-reduced-motion: reduce){ .bases span, .gauge, .bar .fill{ animation:none !important; } .bases span{ opacity:.7; } .bar .fill{ width:var(--w); } .gauge{ --p:var(--target); } }

.stat{ background:var(--panel); border:1px solid var(--line); border-radius:15px; padding:18px 20px; }
.stat .k{ color:var(--mute); font-size:.85rem; }
.stat .v{ font-family:'Bricolage Grotesque',sans-serif; font-size:1.8rem; font-weight:700; margin-top:4px; }

button[data-baseweb="tab"]{ font-family:'Bricolage Grotesque',sans-serif; font-size:1.02rem; font-weight:700; color:#a8c4b9; }
button[data-baseweb="tab"][aria-selected="true"]{ color:#36dfb6; }
div[data-baseweb="tab-highlight"]{ background:var(--benign) !important; }
div[data-testid="stMetric"]{ background:var(--panel); border:1px solid var(--line); padding:15px; border-radius:13px; }
div[data-testid="stForm"]{ background:var(--panel); border:1px solid var(--line); border-radius:16px; }

div.stButton > button, div.stFormSubmitButton > button{ width:100%; height:48px; border-radius:12px; font-weight:600;
  background:var(--benign); color:#04201a; border:none; transition:transform .15s, box-shadow .15s; }
div.stButton > button:hover, div.stFormSubmitButton > button:hover{ transform:translateY(-1px);
  box-shadow:0 6px 22px rgba(45,212,167,.28); color:#04201a; border:none; }

.sub, .rec{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:20px 22px; margin:8px 0; }
.sub .row{ display:flex; align-items:center; justify-content:center; gap:18px; margin:6px 0 14px; }
.aa{ font-family:'Bricolage Grotesque',sans-serif; font-size:3.2rem; font-weight:800; line-height:1; width:88px; height:88px;
  display:flex; align-items:center; justify-content:center; border-radius:18px; border:1px solid var(--line); background:#0a1815; }
.aa.new{ border-color:var(--amber); color:var(--amber); }
.arrow{ font-size:1.8rem; color:var(--mute); }
.chips{ display:flex; gap:8px; flex-wrap:wrap; justify-content:center; }
.chip{ padding:4px 12px; border-radius:999px; font-size:.82rem; border:1px solid var(--line); color:var(--mute); }
.chip.warn{ color:var(--amber); border-color:var(--amber); }
.chip.ok{ color:var(--benign); border-color:var(--benign); }
.chip.bad{ color:var(--path); border-color:var(--path); }
.cap{ color:var(--mute); font-size:.85rem; text-align:center; margin-top:10px; }

.rec .grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:16px 28px; }
.rec .k{ color:var(--mute); font-size:.82rem; margin-bottom:2px; }
.rec .v{ font-size:1.02rem; font-weight:500; word-break:break-word; }

.result{ border-radius:18px; padding:26px; border:1px solid var(--line); background:var(--panel); margin:8px 0; }
.result.pathogenic{ border-color:var(--path); box-shadow:0 10px 40px rgba(255,100,124,.10); }
.result.benign{ border-color:var(--benign); box-shadow:0 10px 40px rgba(45,212,167,.10); }
.verdict{ font-family:'Bricolage Grotesque',sans-serif; font-size:2rem; font-weight:800; margin:0 0 4px; }
.verdict.pathogenic{ color:var(--path); } .verdict.benign{ color:var(--benign); }
.sub-verdict{ color:var(--mute); margin-bottom:18px; }
@property --p { syntax:'<number>'; inherits:false; initial-value:0; }
.gauge{ --p:0; --c:var(--path); width:170px; height:170px; border-radius:50%; margin:auto;
  background:conic-gradient(var(--c) calc(var(--p)*1%), #1a2f2a 0); display:flex; align-items:center; justify-content:center;
  animation:fill 1.1s ease-out forwards; }
@keyframes fill{ from{ --p:0; } to{ --p:var(--target); } }
.gauge .inner{ width:128px; height:128px; border-radius:50%; background:var(--panel); display:flex; flex-direction:column; align-items:center; justify-content:center; }
.gauge .num{ font-family:'Bricolage Grotesque',sans-serif; font-size:2.1rem; font-weight:800; }
.gauge .lbl{ color:var(--mute); font-size:.8rem; text-align:center; max-width:90px; }
.bar{ margin:12px 0; }
.bar .top{ display:flex; justify-content:space-between; font-size:.92rem; margin-bottom:5px; }
.bar .track{ height:10px; background:#1a2f2a; border-radius:6px; overflow:hidden; }
.bar .fill{ height:100%; border-radius:6px; width:0; animation:grow 1s ease-out forwards; }
@keyframes grow{ to{ width:var(--w); } }
.footer{ text-align:center; color:#76968a; font-size:.82rem; padding-top:30px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 4. DATA LOADING
# ============================================================
def normalize_variation_id(value):
    """Convert Variation IDs to a consistent string format."""
    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except (ValueError, TypeError, OverflowError):
        return str(value).strip()


@st.cache_data(show_spinner=False)
def load_csv(path_string, modified_time):
    """Load a CSV and normalize its column names."""
    df = pd.read_csv(path_string, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


@st.cache_resource(show_spinner=False)
def load_model(path_string, modified_time):
    """Load the trained model pipeline once and reuse it."""
    return joblib.load(path_string)


def load_application_data():
    """Load model, lookup dataset, and reference dataset."""
    required_files = [MODEL_PATH, LOOKUP_PATH, DATASET_PATH]
    errors = [f"Missing file: `{p}`" for p in required_files if not p.exists()]
    if errors:
        return None, None, None, errors

    try:
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        lookup = load_csv(str(LOOKUP_PATH), LOOKUP_PATH.stat().st_mtime).copy()
        dataset = load_csv(str(DATASET_PATH), DATASET_PATH.stat().st_mtime)

        missing = [c for c in LOOKUP_REQUIRED_COLUMNS if c not in lookup.columns]
        if missing:
            errors.append("Lookup dataset is missing columns: " + ", ".join(missing))

        missing_feats = [c for c in REQUIRED_FEATURES if c not in lookup.columns]
        if missing_feats:
            errors.append("Model features missing from lookup: " + ", ".join(missing_feats))

        if errors:
            return None, lookup, dataset, errors

        lookup["VariationID_key"] = lookup["VariationID"].apply(normalize_variation_id)
        return model, lookup, dataset, []

    except Exception as exc:
        return None, None, None, [f"Data loading error: {exc}"]


model, variant_lookup, dataset, loading_errors = load_application_data()

# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================
def prepare_model_input(row):
    """Validate and prepare one record using the six model features."""
    missing = [feature for feature in REQUIRED_FEATURES if feature not in row.index]
    if missing:
        raise ValueError(f"Required model features are missing: {missing}")

    gene = str(row["Gene"]).strip()
    variant_type = str(row["Variant type"]).strip()
    consequence = str(row["Molecular consequence"]).strip()
    original_aa = str(row["Original_AA"]).strip()
    new_aa = str(row["New_AA"]).strip()
    protein_position = pd.to_numeric(row["Protein_Position"], errors="coerce")

    if gene not in GENES:
        raise ValueError(f"Gene must be one of: {', '.join(GENES)}.")
    if pd.isna(protein_position) or protein_position <= 0:
        raise ValueError("Protein position must be a valid positive number.")
    if original_aa not in AMINO_ACIDS:
        raise ValueError("Invalid original amino acid.")
    if new_aa not in AMINO_ACIDS:
        raise ValueError("Invalid new amino acid.")
    if original_aa == new_aa:
        raise ValueError("Original and new amino acids must differ.")
    if not variant_type or variant_type.lower() == "nan":
        raise ValueError("Variant type is missing.")
    if not consequence or consequence.lower() == "nan":
        raise ValueError("Molecular consequence is missing.")

    model_input = pd.DataFrame([{
        "Gene": gene,
        "Variant type": variant_type,
        "Molecular consequence": consequence,
        "Protein_Position": float(protein_position),
        "Original_AA": original_aa,
        "New_AA": new_aa,
    }])
    return model_input[REQUIRED_FEATURES]


def get_prediction(input_df):
    """Return model prediction and class probabilities."""
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    return (
        prediction,
        float(probabilities[classes.index(0)]),
        float(probabilities[classes.index(1)]),
    )


def result_card(pred, p_ben, p_path):
    """Build the prediction result card."""
    cls = "pathogenic" if pred == 1 else "benign"
    label = "Pathogenic" if pred == 1 else "Benign"
    highest_probability = max(p_ben, p_path)
    colour = "var(--path)" if pred == 1 else "var(--benign)"

    return f"""
<div class="result {cls}">
<div style="display:flex; gap:28px; align-items:center; flex-wrap:wrap;">
<div style="flex:1; min-width:240px;">
<div class="verdict {cls}">{label}</div>
<div class="sub-verdict">Highest class probability: {highest_probability:.1%}.</div>
<div class="bar"><div class="top"><span>Benign</span><span>{p_ben:.1%}</span></div>
<div class="track"><div class="fill" style="--w:{p_ben*100:.1f}%; background:var(--benign);"></div></div></div>
<div class="bar"><div class="top"><span>Pathogenic</span><span>{p_path:.1%}</span></div>
<div class="track"><div class="fill" style="--w:{p_path*100:.1f}%; background:var(--path);"></div></div></div>
</div>
<div class="gauge" style="--target:{highest_probability*100:.1f}; --c:{colour};">
<div class="inner"><div class="num">{highest_probability:.0%}</div><div class="lbl">highest class probability</div></div>
</div>
</div>
</div>"""


def show_prediction(input_df):
    """Display model prediction and probabilities. Returns 1/0."""
    pred, p_ben, p_path = get_prediction(input_df)
    st.markdown(result_card(pred, p_ben, p_path), unsafe_allow_html=True)
    st.caption(
        "These are model-estimated class probabilities, not calibrated clinical risk "
        "or diagnostic confidence."
    )
    return pred


def substitution_card(original, new):
    if original == new:
        chips = '<span class="chip warn">Amino acids must differ</span>'
    else:
        original_class = AMINO_ACID_CLASSES.get(original, "Unknown")
        new_class = AMINO_ACID_CLASSES.get(new, "Unknown")
        chips = (
            f'<span class="chip">{AMINO_ACIDS.get(original, "Unknown")}: {original_class}</span>'
            f'<span class="chip">{AMINO_ACIDS.get(new, "Unknown")}: {new_class}</span>'
            + (
                '<span class="chip ok">Same chemical class</span>'
                if original_class == new_class
                else '<span class="chip warn">Chemical class changes</span>'
            )
        )

    return (
        f'<div class="sub"><div class="row"><div class="aa">{original}</div><div class="arrow">→</div>'
        f'<div class="aa new">{new}</div></div><div class="chips">{chips}</div>'
        f'<div class="cap">Amino acid substitution</div></div>'
    )


def record_card(items):
    """Render variant information in a responsive card."""
    cells = "".join(
        f'<div><div class="k">{key}</div><div class="v">{value}</div></div>'
        for key, value in items
    )
    return f'<div class="rec"><div class="grid">{cells}</div></div>'


def reference_is_pathogenic(variant):
    """Interpret the ClinVar reference as 1 (pathogenic) / 0 (benign) / None."""
    try:
        return int(float(variant["Label"]))
    except (ValueError, TypeError):
        text = str(variant["Germline classification"]).lower()
        if "pathogenic" in text and "benign" not in text:
            return 1
        if "benign" in text:
            return 0
        return None


def classification_chip(text):
    """Return a styled classification label."""
    text = str(text)
    lowered = text.lower()
    css_class = (
        "bad" if "pathogenic" in lowered and "benign" not in lowered
        else "ok" if "benign" in lowered
        else "warn"
    )
    return f'<span class="chip {css_class}">{text}</span>'


# ============================================================
# 6. HEADER
# ============================================================
st.markdown(
    '<div class="hero"><p class="hero-title">🧬 Genetic Variant Predictor</p>'
    '<p class="hero-subtitle">Look up a ClinVar variant or build one by hand, then see whether the '
    'Random Forest predicts benign or pathogenic, with its estimated class probabilities.</p>'
    '<div class="bases">' + "<span></span>" * 24 + '</div></div>',
    unsafe_allow_html=True,
)

# ============================================================
# 7. DATA STATUS
# ============================================================
if loading_errors:
    st.error("Application data could not be loaded.")
    for error in loading_errors:
        st.write(f"- {error}")

    with st.expander("Diagnostic information"):
        st.write("Project directory:", str(BASE_DIR))
        st.write("Lookup file:", str(LOOKUP_PATH))
        st.write("Lookup exists:", LOOKUP_PATH.exists())
        st.write("Model exists:", MODEL_PATH.exists())
        st.write("Reference dataset exists:", DATASET_PATH.exists())
        if variant_lookup is not None:
            st.write("Columns actually loaded:", variant_lookup.columns.tolist())
    st.stop()

# ============================================================
# 8. DASHBOARD METRICS
# ============================================================
total_variants = len(variant_lookup)
n_pathogenic = int(
    (pd.to_numeric(variant_lookup["Label"], errors="coerce") == 1).sum()
)
pathogenic_share = n_pathogenic / max(total_variants, 1)

c1, c2, c3, c4 = st.columns(4)
for col, label, value in [
    (c1, "Total variants", f"{total_variants:,}"),
    (c2, "Selected genes", "06"),
    (c3, "Pathogenic share", f"{pathogenic_share:.0%}"),
    (c4, "Model", "Random Forest"),
]:
    col.markdown(
        f'<div class="stat"><div class="k">{label}</div><div class="v">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")
st.warning(
    "Research prototype only. Not a diagnostic tool. Performance on previously unseen genes "
    "was below the majority-class baseline in the gene-held-out evaluation. Predictions must "
    "not guide medical decisions."
)

tab_lookup, tab_manual, tab_about = st.tabs(
    ["🔎 ClinVar Lookup", "🧪 Manual Prediction", "ℹ️ About"]
)

# ============================================================
# TAB 1: CLINVAR LOOKUP
# ============================================================
with tab_lookup:
    st.subheader("Search ClinVar variant")
    st.write(
        "Enter a ClinVar Variation ID to retrieve a known variant and evaluate it "
        "with the trained model."
    )

    with st.form("lookup_form"):
        variation_id = st.text_input(
            "ClinVar Variation ID",
            placeholder="Example: 53423",
        )
        lookup_button = st.form_submit_button("Search variant")

    if "lookup_id" not in st.session_state:
        st.session_state.lookup_id = ""

    if lookup_button:
        st.session_state.lookup_id = normalize_variation_id(variation_id)
        if not st.session_state.lookup_id:
            st.warning("Please enter a Variation ID.")

    if st.button("Try a random variant"):
        st.session_state.lookup_id = variant_lookup["VariationID_key"].sample(1).iloc[0]

    search_id = st.session_state.lookup_id

    if search_id:
        matches = variant_lookup[
            variant_lookup["VariationID_key"] == search_id
        ]

        if matches.empty:
            st.error(f"No variant found for Variation ID {search_id}.")
        else:
            variant = matches.iloc[0]
            st.success(f"Variant {search_id} found.")

            st.markdown("### Variant information")
            st.markdown(
                record_card([
                    ("Gene", variant["Gene"]),
                    ("Variation ID", search_id),
                    ("Accession", variant["Accession"]),
                    ("Variant name", variant["Name"]),
                    ("Protein change", variant["Protein change"]),
                    ("Variant type", variant["Variant type"]),
                    ("Protein position", variant["Protein_Position"]),
                    (
                        "ClinVar classification",
                        classification_chip(variant["Germline classification"]),
                    ),
                ]),
                unsafe_allow_html=True,
            )

            original_aa = str(variant["Original_AA"])
            new_aa = str(variant["New_AA"])
            if original_aa in AMINO_ACIDS and new_aa in AMINO_ACIDS:
                st.markdown(
                    substitution_card(original_aa, new_aa),
                    unsafe_allow_html=True,
                )

            st.markdown("### Model evaluation")
            try:
                model_input = prepare_model_input(variant)
                prediction = show_prediction(model_input)

                reference = reference_is_pathogenic(variant)
                if reference is not None:
                    if prediction == reference:
                        st.success(
                            "The model prediction agrees with the ClinVar classification."
                        )
                    else:
                        st.warning(
                            "The model prediction disagrees with the ClinVar classification."
                        )

                with st.expander("View model input features"):
                    st.dataframe(
                        model_input,
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as exc:
                st.error(f"Could not prepare model input or predict: {exc}")

            st.caption(
                "Important: this variant is already in the ClinVar dataset used to develop "
                "the model. Its prediction is not independent validation."
            )

# ============================================================
# TAB 2: MANUAL PREDICTION
# ============================================================
with tab_manual:
    available_genes = [
        gene for gene in GENES
        if gene in variant_lookup["Gene"].astype(str).unique()
    ]
    available_variant_types = sorted(
        variant_lookup["Variant type"].dropna().astype(str).unique().tolist()
    )
    available_consequences = sorted(
        variant_lookup["Molecular consequence"].dropna().astype(str).unique().tolist()
    )

    def aa_format(code):
        return f"{code} - {AMINO_ACIDS[code]}"

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Manual variant prediction")
        st.write("Enter the six features used by the Random Forest model.")

        gene_col, type_col = st.columns(2)
        gene = gene_col.selectbox("Gene", available_genes)
        variant_type = type_col.selectbox(
            "Variant type",
            available_variant_types,
        )
        molecular_consequence = st.selectbox(
            "Molecular consequence",
            available_consequences,
            format_func=lambda value: (
                value if len(value) <= 70 else value[:67] + "..."
            ),
        )

        position_col, original_col, new_col = st.columns([1.2, 1, 1])
        protein_position = position_col.number_input(
            "Protein position",
            min_value=1,
            max_value=100000,
            value=1,
            step=1,
        )
        original_aa = original_col.selectbox(
            "Original amino acid",
            list(AMINO_ACIDS),
            format_func=aa_format,
        )
        new_aa = new_col.selectbox(
            "New amino acid",
            list(AMINO_ACIDS),
            index=1,
            format_func=aa_format,
        )

    with right:
        st.subheader("Substitution")
        st.markdown(
            substitution_card(original_aa, new_aa),
            unsafe_allow_html=True,
        )
        st.caption(f"{gene}: {GENE_INFO.get(gene, '')}")

    if st.button("Predict variant"):
        if original_aa == new_aa:
            st.error("Original and new amino acids cannot be identical.")
        else:
            manual_record = pd.Series({
                "Gene": gene,
                "Variant type": variant_type,
                "Molecular consequence": molecular_consequence,
                "Protein_Position": protein_position,
                "Original_AA": original_aa,
                "New_AA": new_aa,
            })

            try:
                manual_input = prepare_model_input(manual_record)
                st.markdown("### Prediction result")
                show_prediction(manual_input)

                exact_matches = variant_lookup[
                    (variant_lookup["Gene"].astype(str) == gene)
                    & (
                        pd.to_numeric(
                            variant_lookup["Protein_Position"],
                            errors="coerce",
                        ) == protein_position
                    )
                    & (
                        variant_lookup["Original_AA"].astype(str) == original_aa
                    )
                    & (
                        variant_lookup["New_AA"].astype(str) == new_aa
                    )
                ]

                if exact_matches.empty:
                    st.info(
                        "No variant with this gene, position and substitution "
                        "exists in the project dataset."
                    )
                else:
                    st.markdown("### Matching ClinVar records")
                    st.dataframe(
                        exact_matches[
                            [
                                "VariationID",
                                "Accession",
                                "Name",
                                "Germline classification",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                if "missense variant" not in molecular_consequence.lower().split("|"):
                    st.warning(
                        "The selected molecular consequence does not appear to be a "
                        "standalone missense variant. This prediction may not represent "
                        "a typical missense substitution."
                    )

                if variant_type.lower() not in [
                    "single nucleotide variant",
                    "snv",
                ]:
                    st.warning(
                        "The selected variant type may not represent a simple "
                        "single-nucleotide substitution."
                    )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ============================================================
# TAB 3: ABOUT
# ============================================================
with tab_about:
    st.subheader("About this project")
    about_left, about_right = st.columns(2, gap="large")

    with about_left:
        st.markdown(
            "**Multi-gene genetic variant pathogenicity prediction.** A Random Forest "
            "classifies missense variants as pathogenic or benign using selected "
            "ClinVar features."
        )
        st.markdown("**Selected genes**")
        for gene_name, condition in GENE_INFO.items():
            st.write(f"- **{gene_name}**: {condition}")

        st.markdown("**Model features**")
        st.write(
            "Gene, variant type, molecular consequence, protein position, "
            "original amino acid, new amino acid."
        )

        st.markdown("**Model information**")
        st.write(
            "- Algorithm: Random Forest classifier\n"
            "- Dataset: NCBI ClinVar\n"
            "- Classification: binary (pathogenic / benign)\n"
            "- Evaluation: stratified train-test split, protein-grouped "
            "cross-validation and gene-held-out evaluation"
        )

    with about_right:
        st.markdown("**Variants per gene**")
        st.bar_chart(variant_lookup["Gene"].astype(str).value_counts())

        st.markdown("**Important limitations**")
        st.warning(
            "This is an academic research prototype, not a clinical diagnostic tool.\n\n"
            "- Predictions must not be used for medical decisions.\n"
            "- Performance on previously unseen genes was below the majority-class baseline "
            "in the gene-held-out evaluation.\n"
            "- ClinVar lookup predictions are not independent validation because these "
            "variants are in the source dataset.\n"
            "- Model-estimated class probabilities are not clinical risk estimates and "
            "have not necessarily been calibrated."
        )
        st.caption(
            "Data source: NCBI ClinVar, clinical interpretations of genetic variants."
        )

st.markdown(
    '<div class="footer">Genetic Variant Pathogenicity Prediction · '
    'Academic machine learning project · Research use only</div>',
    unsafe_allow_html=True,
)
