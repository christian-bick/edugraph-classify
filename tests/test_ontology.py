import pytest

from edugraph_classify.ontology import OntologyCatalog, OntologyError


def test_pinned_catalog_resolves_dimensions_without_name_inference() -> None:
    catalog = OntologyCatalog.load("0.21.0")

    labels = catalog.labels(["ProcedureUnderstanding", "Addition", "DegreeScale"])

    assert labels.areas == ("Addition",)
    assert labels.scopes == ("DegreeScale",)
    assert labels.abilities == ("ProcedureUnderstanding",)
    assert len(catalog.snapshot_sha256) == 64


def test_catalog_rejects_wrong_version_unknown_and_duplicate_labels() -> None:
    with pytest.raises(OntologyError, match="expected edugraph-py"):
        OntologyCatalog.load("0.20.0")

    catalog = OntologyCatalog.load("0.21.0")
    with pytest.raises(OntologyError, match="unknown ontology"):
        catalog.labels(["NotAnOntologyLabel"])
    with pytest.raises(OntologyError, match="repeats ontology"):
        catalog.labels(["Addition", "Addition"])
    with pytest.raises(OntologyError, match="array"):
        catalog.labels("Addition")
    with pytest.raises(OntologyError, match="non-empty"):
        catalog.labels([""])
