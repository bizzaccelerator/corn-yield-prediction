import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction import DictVectorizer

# Add the scripts directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../scripts"))

# Import the clean module
try:
    from scripts.support_scripts import clean
except ImportError:
    try:
        from support_scripts import clean
    except ImportError:
        import clean


class TestLoadAndUnderstandData:
    """Test suite for load_and_understand_data function"""

    def test_load_corn_csv(self):
        """Test loading the corn.csv file created by CI"""
        if os.path.exists("corn.csv"):
            result = clean.load_and_understand_data("corn.csv")
            assert isinstance(result, pd.DataFrame)
            assert len(result) >= 1
            assert "Education" in result.columns
            assert "Yield" in result.columns

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises error"""
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            clean.load_and_understand_data("nonexistent_file_12345.csv")


class TestPrepareData:
    """Test suite for prepare_data function"""

    @pytest.fixture
    def sample_corn_data(self):
        """Fixture providing sample corn data matching the real structure"""
        return pd.DataFrame(
            {
                "Education": ["Primary", "Secondary", None, "Tertiary", "Primary"],
                "Gender": ["Male", "Female", "Male", "Female", "Male"],
                "Age bracket": ["20-30", "30-40", "40-50", "50-60", "20-30"],
                "Household size": [4, 5, 6, 3, 4],
                "Acreage": [2.5, 3.0, 3.5, 4.0, 2.8],
                "Fertilizer amount": [100, 150, 200, 180, 120],
                "Laborers": [2, 3, 4, 2, 3],
                "Yield": [1000, 1200, 1400, 1300, 1100],
                "Main credit source": ["Bank", "NGO", "Bank", "Family", "Bank"],
                "Farm records": ["Yes", "No", "Yes", "No", "Yes"],
                "Main advisory source": [
                    "Extension",
                    "Radio",
                    "TV",
                    "Extension",
                    "Radio",
                ],
                "Extension provider": ["Gov", "Private", "Gov", "NGO", "Gov"],
                "Advisory format": [
                    "Group",
                    "Individual",
                    "Group",
                    "Individual",
                    "Group",
                ],
                "Advisory language": [
                    "English",
                    "Local",
                    "English",
                    "Local",
                    "English",
                ],
            }
        )

    def test_prepare_data_removes_missing_acreage(self, sample_corn_data):
        """Test that rows with missing acreage are removed"""
        # Add a row with missing acreage
        sample_corn_data.loc[len(sample_corn_data)] = {
            "Education": "Primary",
            "Gender": "Male",
            "Age bracket": "20-30",
            "Household size": 4,
            "Acreage": None,
            "Fertilizer amount": 100,
            "Laborers": 2,
            "Yield": 1000,
            "Main credit source": "Bank",
            "Farm records": "Yes",
            "Main advisory source": "Extension",
            "Extension provider": "Gov",
            "Advisory format": "Group",
            "Advisory language": "English",
        }

        result = clean.prepare_data(sample_corn_data)
        assert result["acreage"].isna().sum() == 0
        assert len(result) == 5  # Should exclude the row with missing acreage

    def test_prepare_data_fills_missing_education(self, sample_corn_data):
        """Test that missing education values are filled"""
        result = clean.prepare_data(sample_corn_data)
        assert result["education"].isna().sum() == 0
        assert "No educated" in result["education"].values

    def test_prepare_data_standardizes_column_names(self, sample_corn_data):
        """Test that column names are lowercased and spaces removed"""
        result = clean.prepare_data(sample_corn_data)

        expected_columns = [
            "education",
            "gender",
            "age_bracket",
            "household_size",
            "acreage",
            "fertilizer_amount",
            "laborers",
            "yield",
            "main_credit_source",
            "farm_records",
            "main_advisory_source",
            "extension_provider",
            "advisory_format",
            "advisory_language",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Column {col} not found in result"

    def test_prepare_data_returns_dataframe(self, sample_corn_data):
        """Test that function returns a DataFrame"""
        result = clean.prepare_data(sample_corn_data)
        assert isinstance(result, pd.DataFrame)


class TestSelectSignificantFeatures:
    """Test suite for select_significant_features function"""

    @pytest.fixture
    def prepared_corn_data(self):
        """Fixture providing prepared corn data"""
        return pd.DataFrame(
            {
                "education": [
                    "Primary",
                    "Secondary",
                    "Tertiary",
                    "Primary",
                    "Secondary",
                ],
                "gender": ["Male", "Female", "Male", "Female", "Male"],
                "age_bracket": ["20-30", "30-40", "40-50", "50-60", "20-30"],
                "household_size": [4, 5, 6, 3, 4],
                "acreage": [2.5, 3.0, 3.5, 4.0, 2.8],
                "fertilizer_amount": [100, 150, 200, 180, 120],
                "laborers": [2, 3, 4, 2, 3],
                "yield": [1000, 1200, 1400, 1300, 1100],
                "main_credit_source": ["Bank", "NGO", "Bank", "Family", "Bank"],
                "farm_records": ["Yes", "No", "Yes", "No", "Yes"],
                "main_advisory_source": [
                    "Extension",
                    "Radio",
                    "TV",
                    "Extension",
                    "Radio",
                ],
                "extension_provider": ["Gov", "Private", "Gov", "NGO", "Gov"],
                "advisory_format": [
                    "Group",
                    "Individual",
                    "Group",
                    "Individual",
                    "Group",
                ],
                "advisory_language": [
                    "English",
                    "Local",
                    "English",
                    "Local",
                    "English",
                ],
            }
        )

    def test_select_significant_features_returns_correct_columns(
        self, prepared_corn_data
    ):
        """Test that only significant features are returned"""
        result = clean.select_significant_features(prepared_corn_data)

        expected_columns = [
            "education",
            "age_bracket",
            "household_size",
            "laborers",
            "main_advisory_source",
            "acreage",
            "fertilizer_amount",
            "yield",
        ]

        assert list(result.columns) == expected_columns

    def test_select_significant_features_excludes_nonsignificant(
        self, prepared_corn_data
    ):
        """Test that non-significant features are excluded"""
        result = clean.select_significant_features(prepared_corn_data)

        excluded_columns = [
            "gender",
            "main_credit_source",
            "farm_records",
            "extension_provider",
            "advisory_format",
            "advisory_language",
        ]

        for col in excluded_columns:
            assert col not in result.columns

    def test_select_significant_features_preserves_row_count(self, prepared_corn_data):
        """Test that number of rows is preserved"""
        result = clean.select_significant_features(prepared_corn_data)
        assert len(result) == len(prepared_corn_data)


class TestEncodeAndSplitData:
    """Test suite for encode_and_split_data function"""

    @pytest.fixture
    def cleaned_corn_data(self):
        """Fixture providing cleaned corn data with enough samples"""
        np.random.seed(42)
        n_samples = 100

        return pd.DataFrame(
            {
                "education": np.random.choice(
                    ["Primary", "Secondary", "Tertiary"], n_samples
                ),
                "age_bracket": np.random.choice(
                    ["20-30", "30-40", "40-50", "50-60"], n_samples
                ),
                "household_size": np.random.randint(3, 8, n_samples),
                "laborers": np.random.randint(1, 6, n_samples),
                "main_advisory_source": np.random.choice(
                    ["Extension", "Radio", "TV"], n_samples
                ),
                "acreage": np.random.uniform(1.5, 4.5, n_samples),
                "fertilizer_amount": np.random.randint(50, 200, n_samples),
                "yield": np.random.uniform(1000, 2000, n_samples),
            }
        )

    def test_encode_and_split_returns_seven_elements(self, cleaned_corn_data):
        """Test that function returns correct number of elements"""
        result = clean.encode_and_split_data(cleaned_corn_data)
        assert len(result) == 7  # X_train, X_val, X_test, y_train, y_val, y_test, dv

    def test_encode_and_split_correct_proportions(self, cleaned_corn_data):
        """Test that splits have approximately correct proportions"""
        X_train, X_val, X_test, y_train, y_val, y_test, dv = (
            clean.encode_and_split_data(
                cleaned_corn_data, test_size=0.2, val_size=0.25, random_state=42
            )
        )

        total = len(X_train) + len(X_val) + len(X_test)

        # Check proportions with tolerance
        test_ratio = len(X_test) / total
        val_ratio = len(X_val) / total
        train_ratio = len(X_train) / total

        assert (
            0.15 <= test_ratio <= 0.25
        ), f"Test ratio {test_ratio} outside expected range"
        assert (
            0.15 <= val_ratio <= 0.25
        ), f"Val ratio {val_ratio} outside expected range"
        assert (
            0.55 <= train_ratio <= 0.65
        ), f"Train ratio {train_ratio} outside expected range"

    def test_encode_and_split_vectorizer_is_fitted(self, cleaned_corn_data):
        """Test that returned vectorizer is fitted"""
        *_, dv = clean.encode_and_split_data(cleaned_corn_data)

        assert isinstance(dv, DictVectorizer)
        assert hasattr(dv, "vocabulary_")
        assert len(dv.vocabulary_) > 0

    def test_encode_and_split_consistent_shapes(self, cleaned_corn_data):
        """Test that X and y have matching lengths"""
        X_train, X_val, X_test, y_train, y_val, y_test, dv = (
            clean.encode_and_split_data(cleaned_corn_data)
        )

        assert X_train.shape[0] == len(y_train)
        assert X_val.shape[0] == len(y_val)
        assert X_test.shape[0] == len(y_test)

    def test_encode_and_split_reproducibility(self, cleaned_corn_data):
        """Test that same random_state produces same splits"""
        result1 = clean.encode_and_split_data(cleaned_corn_data, random_state=42)
        result2 = clean.encode_and_split_data(cleaned_corn_data, random_state=42)

        np.testing.assert_array_equal(result1[0], result2[0])  # X_train


class TestSaveOutputs:
    """Test suite for save_outputs function"""

    @pytest.fixture
    def sample_splits(self):
        """Fixture providing sample splits"""
        np.random.seed(42)
        X_train = np.random.rand(60, 10)
        X_val = np.random.rand(20, 10)
        X_test = np.random.rand(20, 10)
        y_train = pd.Series(np.random.uniform(1000, 2000, 60))
        y_val = pd.Series(np.random.uniform(1000, 2000, 20))
        y_test = pd.Series(np.random.uniform(1000, 2000, 20))

        dv = DictVectorizer()
        dv.fit([{"feature_" + str(i): i for i in range(5)} for _ in range(10)])

        return X_train, X_val, X_test, y_train, y_val, y_test, dv

    def test_save_outputs_creates_directory(self, sample_splits, tmp_path):
        """Test that output directory is created"""
        output_dir = tmp_path / "test_output"

        clean.save_outputs(*sample_splits, output_dir=str(output_dir))

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_save_outputs_creates_csv_files(self, sample_splits, tmp_path):
        """Test that CSV files are created"""
        output_dir = tmp_path / "test_output"

        clean.save_outputs(*sample_splits, output_dir=str(output_dir))

        expected_files = [
            "X_train.csv",
            "X_val.csv",
            "X_test.csv",
            "y_train.csv",
            "y_val.csv",
            "y_test.csv",
        ]

        for filename in expected_files:
            file_path = output_dir / filename
            assert file_path.exists(), f"File {filename} was not created"

    def test_save_outputs_creates_vectorizer(self, sample_splits, tmp_path):
        """Test that vectorizer pickle file is created"""
        output_dir = tmp_path / "test_output"

        # Change to tmp directory to save vectorizer there
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            clean.save_outputs(*sample_splits, output_dir="test_output")
            assert (tmp_path / "vectorizer.pkl").exists()
        finally:
            os.chdir(original_dir)

    def test_save_outputs_csv_readable(self, sample_splits, tmp_path):
        """Test that saved CSV files can be read"""
        output_dir = tmp_path / "test_output"

        clean.save_outputs(*sample_splits, output_dir=str(output_dir))

        X_train_loaded = pd.read_csv(output_dir / "X_train.csv")
        assert X_train_loaded.shape[0] == 60
        assert X_train_loaded.shape[1] == 10


class TestProcessCornData:
    """Integration tests for the main process_corn_data function"""

    def test_process_corn_data_with_mock_csv(self, tmp_path):
        """Test the complete pipeline with mock data"""
        # Create a mock corn.csv file
        csv_file = tmp_path / "test_corn.csv"

        np.random.seed(42)
        n_samples = 100
        test_data = pd.DataFrame(
            {
                "Education": np.random.choice(
                    ["Primary", "Secondary", "Tertiary", None], n_samples
                ),
                "Gender": np.random.choice(["Male", "Female"], n_samples),
                "Age bracket": np.random.choice(
                    ["20-30", "30-40", "40-50", "50-60"], n_samples
                ),
                "Household size": np.random.randint(3, 8, n_samples),
                "Acreage": np.random.choice([2.5, 3.0, 3.5, 4.0, None], n_samples),
                "Fertilizer amount": np.random.randint(50, 200, n_samples),
                "Laborers": np.random.randint(1, 6, n_samples),
                "Yield": np.random.uniform(1000, 2000, n_samples),
                "Main credit source": np.random.choice(
                    ["Bank", "NGO", "Family"], n_samples
                ),
                "Farm records": np.random.choice(["Yes", "No"], n_samples),
                "Main advisory source": np.random.choice(
                    ["Extension", "Radio", "TV"], n_samples
                ),
                "Extension provider": np.random.choice(
                    ["Gov", "Private", "NGO"], n_samples
                ),
                "Advisory format": np.random.choice(["Group", "Individual"], n_samples),
                "Advisory language": np.random.choice(["English", "Local"], n_samples),
            }
        )
        test_data.to_csv(csv_file, index=False)

        output_dir = tmp_path / "outputs"

        # Change to tmp directory for vectorizer.pkl
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = clean.process_corn_data(str(csv_file), output_dir=str(output_dir))

            # Test return value
            assert isinstance(result, dict)
            expected_keys = [
                "X_train",
                "X_val",
                "X_test",
                "y_train",
                "y_val",
                "y_test",
                "vectorizer",
            ]
            for key in expected_keys:
                assert key in result, f"Key {key} not in result"

            # Test file creation
            assert output_dir.exists()
            assert (output_dir / "X_train.csv").exists()
            assert (tmp_path / "vectorizer.pkl").exists()

        finally:
            os.chdir(original_dir)

    def test_process_corn_data_returns_valid_types(self, tmp_path):
        """Test that returned data has correct types"""
        csv_file = tmp_path / "test_corn.csv"

        np.random.seed(42)
        test_data = pd.DataFrame(
            {
                "Education": ["Primary"] * 50,
                "Gender": ["Male"] * 50,
                "Age bracket": ["20-30"] * 50,
                "Household size": [4] * 50,
                "Acreage": [2.5] * 50,
                "Fertilizer amount": [100] * 50,
                "Laborers": [2] * 50,
                "Yield": [1500] * 50,
                "Main credit source": ["Bank"] * 50,
                "Farm records": ["Yes"] * 50,
                "Main advisory source": ["Extension"] * 50,
                "Extension provider": ["Gov"] * 50,
                "Advisory format": ["Group"] * 50,
                "Advisory language": ["English"] * 50,
            }
        )
        test_data.to_csv(csv_file, index=False)

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = clean.process_corn_data(str(csv_file), output_dir="output")

            assert isinstance(result["X_train"], np.ndarray)
            assert isinstance(result["y_train"], pd.Series)
            assert isinstance(result["vectorizer"], DictVectorizer)

        finally:
            os.chdir(original_dir)


class TestSaveToCsv:
    """Test suite for save_to_csv helper function"""

    def test_save_dataframe_to_csv(self, tmp_path):
        """Test saving a DataFrame"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        filepath = tmp_path / "test_df.csv"

        clean.save_to_csv(df, str(filepath))

        assert filepath.exists()
        loaded = pd.read_csv(filepath)
        pd.testing.assert_frame_equal(df, loaded)

    def test_save_array_to_csv(self, tmp_path):
        """Test saving a numpy array"""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        filepath = tmp_path / "test_arr.csv"

        clean.save_to_csv(arr, str(filepath))

        assert filepath.exists()
        loaded = pd.read_csv(filepath)
        assert loaded.shape == (3, 2)


# Cleanup fixture
@pytest.fixture(scope="session", autouse=True)
def cleanup_test_files():
    """Cleanup any test files created during testing"""
    yield

    # Remove vectorizer.pkl if it exists in current directory
    if os.path.exists("vectorizer.pkl"):
        try:
            os.remove("vectorizer.pkl")
        except:
            pass

    # Remove data_splits directory if it exists
    if os.path.exists("data_splits"):
        try:
            import shutil

            shutil.rmtree("data_splits")
        except:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
