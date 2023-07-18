from sam_ml.data.regio import get_plz_mapping


def test_get_plz_mapping():
    df = get_plz_mapping()
    assert df.columns == ["ort", "plz", "landkreis", "bundesland"], "columns ['ort', 'plz', 'landkreis', 'bundesland'] should be included"
