import pytest
from toolbox.utils import get_aoi_from_nuts


class TestGetAOIFromNuts:
    @pytest.mark.skip(reason="Requires GEE initialization")
    def test_get_aoi_from_nuts_country_only(self):
        country_aoi, county_aoi = get_aoi_from_nuts(country_code="AT", county_name=None)

        assert country_aoi is not None

    @pytest.mark.skip(reason="Requires GEE initialization")
    def test_get_aoi_from_nuts_with_county(self):
        country_aoi, county_aoi = get_aoi_from_nuts(
            country_code="AT", county_name="Tirol"
        )

        assert country_aoi is not None
        assert county_aoi is not None

    def test_get_aoi_from_nuts_invalid_country(self):
        with pytest.raises(Exception):
            get_aoi_from_nuts(country_code="XX")

    def test_get_aoi_from_nuts_invalid_county(self):
        with pytest.raises(Exception):
            get_aoi_from_nuts(country_code="AT", county_name="InvalidCountyName")
