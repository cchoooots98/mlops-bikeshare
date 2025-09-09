# Basic pytest to ensure test discovery and a simple project import work.


def test_sanity():
    # English: Always-true assertion so CI has at least one passing test.
    assert True


def test_feature_columns_importable():
    # English: Verify we can import the project schema and it contains at least one feature.
    from src.features.schema import FEATURE_COLUMNS

    assert isinstance(FEATURE_COLUMNS, list)
    assert len(FEATURE_COLUMNS) > 0
