"""UI Streamlit — revue steward (human-in-the-loop).

Deuxième porte d'entrée (CONTEXT.md §11) : un écran pour les non-devs (data
steward, médecin) qui consomme le ``MappingService`` partagé avec l'API. Couche
d'affichage volontairement mince ; toute la logique testable est dans
``ui/service.py``.

Lancement : ``gor ui`` (ou ``streamlit run src/governed_omop_rag/ui/app.py``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from governed_omop_rag.config import get_settings
from governed_omop_rag.feedback import FeedbackStore, feedback_records_from_decisions
from governed_omop_rag.service import MappingService, MapStrategy
from governed_omop_rag.ui.service import (
    collect_validated,
    requests_from_records,
    suggestion_to_row,
    to_source_to_concept_map,
)


@st.cache_resource
def _service(bronze_dir: str) -> MappingService:
    return MappingService(get_settings(), Path(bronze_dir) if bronze_dir else None)


def main() -> None:
    st.set_page_config(page_title="governed-omop-rag", layout="wide")
    st.title("governed-omop-rag — revue steward")
    st.caption("Mapping CIM-10 FR / libellés → concepts standard OMOP, sous supervision humaine.")

    with st.sidebar:
        bronze_dir = st.text_input("Répertoire OHDSI (bronze)", value="tests/fixtures")
        strategy = st.selectbox("Stratégie", [s.value for s in MapStrategy], index=0)
        st.info(
            "Données publiques/synthétiques. Outil d'aide à la décision : "
            "la validation humaine est requise."
        )

    service = _service(bronze_dir)
    st.success(f"{service.concepts_indexed} concepts indexés.")

    records: list[dict[str, object]] = []
    tab_file, tab_text = st.tabs(["Fichier CSV/Excel", "Libellés (texte)"])
    with tab_file:
        upload = st.file_uploader(
            "Colonnes : source_code, source_label, source_vocabulary", type=["csv", "xlsx"]
        )
        if upload is not None:
            df = pd.read_excel(upload) if upload.name.endswith(".xlsx") else pd.read_csv(upload)
            records = df.to_dict("records")
    with tab_text:
        text = st.text_area("Un libellé par ligne")
        if text.strip():
            records = [{"source_label": line} for line in text.splitlines() if line.strip()]

    if st.button("Mapper", type="primary") and records:
        requests = requests_from_records(records)
        st.session_state["suggestions"] = service.map_many(requests, MapStrategy(strategy))

    suggestions = st.session_state.get("suggestions")
    if not suggestions:
        st.stop()

    st.subheader(f"{len(suggestions)} suggestion(s) — validez, corrigez ou rejetez")
    decisions: list[tuple[object, int | None]] = []
    for i, suggestion in enumerate(suggestions):
        row = suggestion_to_row(suggestion)
        header = (
            f"{row['source_code'] or row['source_label']} → "
            f"{row['target_concept_id']} ({row['source']}, conf {row['confidence']})"
        )
        with st.expander(header):
            st.write(row["justification"])
            options: dict[str, int] = {}
            for c in suggestion.candidates:
                label = f"{c.concept_id} — {c.concept_name} ({c.vocabulary_id}) [{c.score:.2f}]"
                options[label] = c.concept_id
            labels = ["(rejeter)", *options.keys()]
            default_index = 0
            for j, key in enumerate(options):
                if options[key] == suggestion.target_concept_id:
                    default_index = j + 1
                    break
            choice = st.selectbox("Décision", labels, index=default_index, key=f"dec_{i}")
            target = None if choice == "(rejeter)" else options[choice]
            decisions.append((suggestion, target))

    validated = collect_validated(decisions)
    st.divider()

    col_export, col_feedback = st.columns(2)
    with col_export:
        if validated:
            table = pd.DataFrame(to_source_to_concept_map(validated))
            st.dataframe(table, width="stretch")
            st.download_button(
                "Exporter source_to_concept_map (CSV)",
                table.to_csv(index=False).encode("utf-8"),
                "source_to_concept_map.csv",
                "text/csv",
            )
        else:
            st.warning("Aucune ligne validée pour l'instant.")
    with col_feedback:
        st.caption(
            "Le feedback (accepté/corrigé/rejeté) alimente l'amélioration continue "
            "et peut enrichir le gold set d'évaluation."
        )
        if st.button("Enregistrer le feedback"):
            store = FeedbackStore(get_settings().feedback_path)
            try:
                n = store.record(feedback_records_from_decisions(decisions))
                total = store.count()
            finally:
                store.close()
            st.success(f"{n} décision(s) enregistrée(s) — {total} au total.")


main()
