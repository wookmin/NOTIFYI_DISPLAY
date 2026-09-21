from src.display.expression_display import (
    DisplayPolicy,
    Expression,
    ExpressionController,
)
from src.perception.posture_features import PostureAngles
from src.posture.classifier import classify


def posture(torso, neck, confidence=1.0):
    return classify(PostureAngles(0.0, torso, neck, confidence))


def test_bad_posture_changes_to_frown_after_sustain():
    controller = ExpressionController(DisplayPolicy(3.0, 1.0))
    bad = posture(25.0, 3.0)

    assert controller.update(bad, 0.0) is Expression.SMILE
    assert controller.update(bad, 2.9) is Expression.SMILE
    assert controller.update(bad, 3.0) is Expression.FROWN


def test_good_posture_recovers_to_smile():
    controller = ExpressionController(DisplayPolicy(0.0, 1.0))
    bad = posture(25.0, 3.0)
    good = posture(5.0, 3.0)

    assert controller.update(bad, 0.0) is Expression.FROWN
    assert controller.update(good, 0.9) is Expression.FROWN
    assert controller.update(good, 2.0) is Expression.SMILE


def test_unknown_holds_current_expression():
    controller = ExpressionController(DisplayPolicy(0.0, 1.0))
    bad = posture(25.0, 3.0)
    unknown = posture(25.0, 3.0, confidence=0.0)

    assert controller.update(bad, 0.0) is Expression.FROWN
    assert controller.update(unknown, 10.0) is Expression.FROWN
