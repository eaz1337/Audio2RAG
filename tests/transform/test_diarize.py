from transform.diarize import SpeakerTurn, assign_speaker


class TestAssignSpeaker:
    def test_picks_turn_with_greatest_overlap(self):
        turns = [
            SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00"),
            SpeakerTurn(start=1.0, end=5.0, speaker="SPEAKER_01"),
        ]

        assert assign_speaker(0.5, 3.0, turns) == "SPEAKER_01"

    def test_returns_none_when_no_turn_overlaps(self):
        turns = [SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00")]

        assert assign_speaker(5.0, 6.0, turns) is None

    def test_returns_none_for_empty_turns(self):
        assert assign_speaker(0.0, 1.0, []) is None

    def test_exact_match_within_a_single_turn(self):
        turns = [SpeakerTurn(start=0.0, end=10.0, speaker="SPEAKER_00")]

        assert assign_speaker(2.0, 3.0, turns) == "SPEAKER_00"
