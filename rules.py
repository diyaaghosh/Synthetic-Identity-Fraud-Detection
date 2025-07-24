def apply_rules(user_row):
    score = 0
    reasons = []

    # Rule 1: Free email + very short session
    if user_row.get("email_is_free") == 1 and user_row.get("session_length_in_minutes", 999) < 2:
        score += 0.2
        reasons.append("Free email + very short session")

    # Rule 2: High 6h transaction velocity
    if user_row.get("velocity_6h", 0) > 10:
        score += 0.2
        reasons.append("High transaction velocity in 6h")

    # Rule 3: Very high proposed credit limit
    if user_row.get("proposed_credit_limit", 0) > 50000:
        score += 0.2
        reasons.append("Unusually high credit limit request")

    # Rule 4: Known fraudulent device
    if user_row.get("device_fraud_count", 0) > 0:
        score += 0.3
        reasons.append("Device linked to previous fraud")

    # Rule 5: Unknown housing + employment
    if user_row.get("housing_status_Unknown", 0) == 1 and user_row.get("employment_status_Unknown", 0) == 1:
        score += 0.1
        reasons.append("Unknown housing and employment status")

    # Rule 6: Very young or very old age
    age = user_row.get("customer_age", 30)
    if age < 18 or age > 80:
        score += 0.1
        reasons.append("Suspicious age: under 18 or over 80")

    # Rule 7: Short address history
    if user_row.get("current_address_months_count", 999) < 6:
        score += 0.2
        reasons.append("Short current address history")

    # Rule 8: Credit risk score is very low
    if user_row.get("credit_risk_score", 0) < 300:
        score += 0.2
        reasons.append("Low credit risk score")

    # Rule 9: IP used across many ZIP codes (zip_count_4w)
    if user_row.get("zip_count_4w", 0) > 3:
        score += 0.1
        reasons.append("IP used in multiple ZIPs recently")

    # Rule 10: Multiple devices tied to one account (device_distinct_emails_8w)
    if user_row.get("device_distinct_emails_8w", 0) > 3:
        score += 0.1
        reasons.append("Multiple accounts on same device")

    return round(score, 3), reasons
