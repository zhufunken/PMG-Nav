"""Offline checks for language-grounded room-entry checkpoints."""

from pmg_nav.runtime import PMGEpisodeState


def _item(node_id, label, position):
    return {
        "node_id": node_id,
        "label": label,
        "position": position,
        "distance": 0.0,
    }


def test_room_entry_uses_door_beyond_far_carpet(monkeypatch):
    monkeypatch.setenv("STAGE2_M1_VARIANT", "simple_oracle")
    state = PMGEpisodeState(
        "Turn right and follow the blue carpet to the end of the hallway. "
        "Enter the room with the reception desk.",
        "M1",
        start_yaw=3.14,
    )
    state._start_robot_position = (-8.2, 52.2, 0.01)
    objects = [
        _item("/Root/SM_FloorCarpetPath_5", "sm floorcarpetpath 5", (-2.04, 17.84, 0.0)),
        _item("/Root/SM_FloorCarpetPath_6", "sm floorcarpetpath 6", (-2.04, 23.84, 0.0)),
        _item("/Root/SM_DoorEntrance_door_L", "sm doorentrance door l", (-3.0, 11.993, 0.0)),
        _item("/Root/BP_Elevator_frame2", "bp elevator frame2", (-15.0, -1.5, 0.0)),
        _item("/Root/SM_ReceptionTableStraight5", "sm receptiontablestraight5", (-1.873, -1.0, 0.0)),
    ]
    result = state._infer_room_entry_checkpoint(objects)
    assert result is not None
    assert result["support"]["node_id"].endswith("FloorCarpetPath_5")
    assert result["entry"]["node_id"].endswith("DoorEntrance_door_L")
    assert result["reason"] == "room_entry_checkpoint:far_carpet_to_door"


def test_ordinary_object_instruction_does_not_switch_to_room_entry(monkeypatch):
    monkeypatch.setenv("STAGE2_M1_VARIANT", "simple_oracle")
    state = PMGEpisodeState("Stop near the reception desk.", "M1")
    state._start_robot_position = (-8.2, 52.2, 0.01)
    objects = [
        _item("desk", "reception desk", (-1.8, -1.0, 0.0)),
        _item("door", "door entrance", (-3.0, 12.0, 0.0)),
        _item("carpet", "floor carpet path", (-2.0, 17.8, 0.0)),
    ]
    assert state._infer_room_entry_checkpoint(objects) is None
