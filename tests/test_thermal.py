from datetime import datetime

from arxiv_gnn.thermal import throttle_events


def test_throttle_events_are_read_from_the_heat_log(tmp_path):
    log = tmp_path / "temps.log"
    log.write_text(
        "2026-11-04 12:00:00 70C load=1.0 ralenti=0\n"
        "12:00:10 RALENTI 91C\n"
        "2026-11-04 12:00:10 91C load=2.0 ralenti=1\n"
        "12:00:40 NORMAL 77C\n"
        "2026-11-04 12:00:40 77C load=1.0 ralenti=0\n"
    )
    ts = lambda hms: datetime.strptime(f"2026-11-04 {hms}", "%Y-%m-%d %H:%M:%S").timestamp()
    assert len(throttle_events(ts("12:00:05"), ts("12:00:20"), log)) == 2
    assert throttle_events(ts("12:01:00"), ts("12:02:00"), log) == []
    assert throttle_events(ts("12:00:00"), ts("12:00:30"), tmp_path / "absent.log") == []
