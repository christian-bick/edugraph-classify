import pytest
from edugraph import Area, Scope, has_part, specializes_transitive, structures_transitive

from edugraph_classify import ontology
from edugraph_classify.ontology import OntologyCatalog, OntologyError


def test_pinned_catalog_resolves_dimensions_without_name_inference() -> None:
    catalog = OntologyCatalog.load("0.30.0")

    labels = catalog.labels(["ProcedureUnderstanding", "Addition", "DegreeScale"])

    assert labels.areas == ("Addition",)
    assert labels.scopes == ("DegreeScale",)
    assert labels.abilities == ("ProcedureUnderstanding",)
    assert len(catalog.snapshot_sha256) == 64


def test_catalog_rejects_wrong_version_unknown_and_duplicate_labels() -> None:
    with pytest.raises(OntologyError, match="expected edugraph-py"):
        OntologyCatalog.load("0.20.0")

    catalog = OntologyCatalog.load("0.30.0")
    with pytest.raises(OntologyError, match="unknown ontology"):
        catalog.labels(["NotAnOntologyLabel"])
    with pytest.raises(OntologyError, match="repeats ontology"):
        catalog.labels(["Addition", "Addition"])
    with pytest.raises(OntologyError, match="array"):
        catalog.labels("Addition")
    with pytest.raises(OntologyError, match="non-empty"):
        catalog.labels([""])


@pytest.mark.parametrize(
    "label", ["CircularShapes", "LengthMeasurement", "ErrorCorrection", "JustificationScope"]
)
def test_catalog_rejects_organizational_labels_without_rewriting_them(label: str) -> None:
    catalog = OntologyCatalog.load("0.30.0")
    assert label in catalog.dimensions
    with pytest.raises(OntologyError, match="organizational"):
        catalog.labels([label])


def test_specialization_families_and_new_scopes_remain_valid_explicit_labels() -> None:
    catalog = OntologyCatalog.load("0.30.0")
    labels = catalog.labels(["Rectangle", "ProofMethod", "SixthFractions"])
    assert labels.areas == ("Rectangle",)
    assert labels.scopes == ("ProofMethod", "SixthFractions")
    assert labels.abilities == ()
    assert catalog.labels([]).to_mapping() == {"areas": [], "scopes": [], "abilities": []}


def test_release_relations_keep_inheritance_separate_from_membership() -> None:
    assert Area.Rectangle in specializes_transitive(Area.Square)
    assert Area.HalfCircle in has_part(Area.CircularShapes)
    assert Area.Circle not in specializes_transitive(Area.HalfCircle)
    assert Scope.ProofMethod in specializes_transitive(Scope.DirectProof)
    assert Scope.JustificationScope not in specializes_transitive(Scope.DirectProof)
    assert Scope.JustificationScope in structures_transitive(Scope.DirectProof)

    labels = OntologyCatalog.load("0.30.0").labels(["Square", "DirectProof"])
    assert labels.areas == ("Square",)
    assert labels.scopes == ("DirectProof",)


def test_snapshot_is_order_independent_but_detects_definition_and_relation_changes(monkeypatch) -> None:
    original_relations = ontology.relations
    baseline = OntologyCatalog.load("0.30.0")

    def reversed_relations(member):
        return {
            name: list(reversed(value)) if isinstance(value, list) else value
            for name, value in reversed(list(original_relations(member).items()))
        }

    monkeypatch.setattr(ontology, "relations", reversed_relations)
    assert OntologyCatalog.load("0.30.0").snapshot_sha256 == baseline.snapshot_sha256

    def changed_relations(member):
        result = dict(original_relations(member))
        if member == Area.Square:
            result["specializes"] = [Area.Polygon]
        return result

    monkeypatch.setattr(ontology, "relations", changed_relations)
    assert OntologyCatalog.load("0.30.0").snapshot_sha256 != baseline.snapshot_sha256

    monkeypatch.setattr(ontology, "relations", original_relations)
    original_definition = Area.definition

    def changed_definition(member):
        return "Changed meaning" if member == Area.Square else original_definition.fget(member)

    monkeypatch.setattr(
        Area,
        "definition",
        property(changed_definition),
    )
    assert OntologyCatalog.load("0.30.0").snapshot_sha256 != baseline.snapshot_sha256

    monkeypatch.setattr(Area, "definition", original_definition)
    original_eligibility = ontology.is_label_eligible
    monkeypatch.setattr(
        ontology,
        "is_label_eligible",
        lambda member: False if member == Area.Square else original_eligibility(member),
    )
    assert OntologyCatalog.load("0.30.0").snapshot_sha256 != baseline.snapshot_sha256


def test_historical_ontology_pin_is_not_silently_upgraded() -> None:
    with pytest.raises(OntologyError, match="expected edugraph-py 0.21.0, found 0.30.0"):
        OntologyCatalog.load("0.21.0")


@pytest.mark.parametrize("label", ["AbsoluteValue", "AxiomDefinition"])
def test_removed_identifiers_are_not_mapped_to_new_concepts(label: str) -> None:
    with pytest.raises(OntologyError, match="unknown ontology"):
        OntologyCatalog.load("0.30.0").labels([label])
