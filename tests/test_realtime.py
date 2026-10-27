import pytest
import time
from src.realtime.stream_engine import RealTimeStreamEngine, StreamingBurstDetector, StreamingTemplateRegistry

def test_streaming_burst_detector():
    detector = StreamingBurstDetector(alpha=0.3, z_threshold=2.0)
    # Simulate normal low velocity
    for i in range(10):
        z, is_burst = detector.update(10.0, time.time() + i)
        assert not is_burst

    # Simulate sudden massive spike
    z_spike, is_spike = detector.update(150.0, time.time() + 11)
    assert is_spike
    assert z_spike > 2.0


def test_streaming_template_registry():
    registry = StreamingTemplateRegistry(similarity_threshold=0.70)
    tpl1 = "Pushpa 2 is an absolute sensational blockbuster! Must watch with family!"
    tpl2 = "Pushpa 2 is an absolute sensational blockbuster! Must watch with whole family!"
    
    id1, hits1 = registry.register_or_match(tpl1, "user_1", "paid-commercial", time.time())
    assert id1 is not None
    assert hits1 == 1

    id2, hits2 = registry.register_or_match(tpl2, "user_2", "paid-commercial", time.time())
    assert id2 == id1
    assert hits2 == 2


def test_realtime_stream_engine_processing():
    film_cfg = {
        "id": "pushpa2_2024",
        "title": "Pushpa 2",
        "promotional_keywords": ["blockbuster", "mass film of the year", "rampage"],
        "fan_markers": ["icon star", "allu arjun", "bunny", "thaggede le"],
        "cast": ["Allu Arjun", "Rashmika Mandanna", "Fahadh Faasil"],
        "director": "Sukumar"
    }
    engine = RealTimeStreamEngine(film_cfg, imdb_rating=7.4)

    # 1. Test Commercial post
    res_comm = engine.process_post(
        "Pushpa 2 is sensational blockbuster! Mass film of the year with records broken everywhere!",
        author="pr_desk_1"
    )
    assert res_comm["motive"] == "paid-commercial"
    assert res_comm["scores"]["marketing"] > 0.3

    # 2. Test Fan post
    res_fan = engine.process_post(
        "Icon Star Allu Arjun anna mass rampage! Bunny fans assemble #PushpaRaj #ThaggedeLe",
        author="bunny_cult"
    )
    assert res_fan["motive"] == "fan-coordinated"
    assert res_fan["scores"]["fan"] > 0.3

    # 3. Test Organic post
    res_org = engine.process_post(
        "Fahadh Faasil was great as Shekhawat, but second half dragged. Editing could have been better.",
        author="movie_fan"
    )
    assert res_org["motive"] == "organic"
    assert res_org["scores"]["specificity"] > 0.2

    # Verify Summary
    state = engine.get_summary_state()
    assert state["posts_processed"] == 3
    assert state["motive_counts"]["paid-commercial"] == 1
    assert state["motive_counts"]["fan-coordinated"] == 1
    assert state["motive_counts"]["organic"] == 1
