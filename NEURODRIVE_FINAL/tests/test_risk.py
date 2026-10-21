from autonomy.risk_engine import RiskEngine

def test_critical_distance():
    assert RiskEngine().assess([{"class_name":"car","distance_m":3,"direction":"FRONT"}],{"offset":0})["risk_level"] == "CRITICAL"

def test_low_risk():
    assert RiskEngine().assess([{"class_name":"car","distance_m":50,"direction":"FRONT"}],{"offset":0})["risk_level"] == "LOW"

def test_unknown_without_distance():
    assert RiskEngine().assess([{"class_name":"person"}],{"offset":0})["risk_level"] == "UNKNOWN"

def test_direction_reported():
    result = RiskEngine().assess([{"class_name":"car","distance_m":8,"direction":"LEFT"}],{"offset":0})
    assert result["direction"] == "LEFT"
