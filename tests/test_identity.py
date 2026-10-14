from app.data import REPO
from app.identity import evaluate

PH = REPO.policyholders


def test_three_fields_verify():
    r = evaluate({"full_name": "Margaret Chen", "dob": "1985-03-15", "ssn_last4": "4472"}, PH)
    assert r.verified and r.party_id == "P9"


def test_two_fields_need_more_and_reveal_nothing():
    r = evaluate({"full_name": "Margaret Chen", "dob": "1999-01-01"}, PH)
    assert not r.verified and r.reason == "need_more"   # mismatch is not disclosed below 3 fields


def test_policy_number_does_not_count():
    r = evaluate({"full_name": "Margaret Chen", "dob": "1985-03-15", "policy_number": "POL-9921"}, PH)
    assert not r.verified and r.still_needed == 1


def test_any_contradiction_blocks():
    r = evaluate({"full_name": "Margaret Chen", "dob": "1985-03-15", "ssn_last4": "4472", "phone": "555-111-2222"}, PH)
    assert not r.verified and r.reason == "mismatch"


def test_phone_and_email_normalization_and_aliases():
    r = evaluate({"full_name": "yaven li", "phone": "(650) 521-2830", "email": "YAWEN.LI@example.com"}, PH)
    assert r.verified and r.party_id == "P13"


def test_national_id_counts_as_last4():
    r = evaluate({"full_name": "Ma Tian", "dob": "1964-09-10", "ssn_last4": "6688"}, PH)
    assert r.verified and r.party_id == "P12"


def test_fields_from_two_people_do_not_combine():
    # name + dob from Margaret, last4 from Ava
    r = evaluate({"full_name": "Margaret Chen", "dob": "1985-03-15", "ssn_last4": "9180"}, PH)
    assert not r.verified
