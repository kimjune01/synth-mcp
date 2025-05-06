import unittest
from pathlib import Path
import json
import tempfile
import shutil
import sys
import asyncio
from server import MidiCompositionServer, ProjectState, TrackState
import pytest
import os


class TestMidiCompositionServer(unittest.TestCase):
    def setUp(self):
        print("Setting up test...", file=sys.stderr)
        # Create a temporary directory for testing
        self.test_dir = Path(tempfile.mkdtemp())
        print(f"Created temp dir: {self.test_dir}", file=sys.stderr)

        self.server = MidiCompositionServer()
        print("Created server instance", file=sys.stderr)

        self.server.workspace_dir = self.test_dir / "workspace"
        self.server.workspace_dir.mkdir(exist_ok=True)
        print(f"Created workspace dir: {self.server.workspace_dir}", file=sys.stderr)

        print("Test setup complete", file=sys.stderr)

    def tearDown(self):
        print("Tearing down test...", file=sys.stderr)
        try:
            if hasattr(self, "test_dir"):
                print(f"Removing temp dir: {self.test_dir}", file=sys.stderr)
                shutil.rmtree(self.test_dir)
            print("Test teardown complete", file=sys.stderr)
        except Exception as e:
            print(f"Error in teardown: {e}", file=sys.stderr)

    async def async_test_project_creation(self):
        """Test creating a new project"""
        try:
            print("\nStarting project creation test...", file=sys.stderr)
            project_name = "test_project"
            tempo = 120
            time_signature = (4, 4)

            # Create project
            print("Creating project...", file=sys.stderr)
            result = self.server.create_project(project_name, tempo, time_signature)
            print(f"Project creation result: {result}", file=sys.stderr)
            self.assertTrue(result, "Project creation failed")

            # Verify project was created
            print("Verifying project...", file=sys.stderr)
            self.assertIsNotNone(self.server.current_project)
            self.assertEqual(self.server.current_project.name, project_name)
            self.assertEqual(self.server.current_project.tempo, tempo)
            self.assertEqual(self.server.current_project.time_signature, time_signature)

            # Verify project file was created
            print("Checking project file...", file=sys.stderr)
            project_file = self.server.workspace_dir / f"{project_name}.json"
            self.assertTrue(project_file.exists())

            # Verify file contents
            print("Reading project file...", file=sys.stderr)
            with open(project_file) as f:
                data = json.load(f)
                self.assertEqual(data["name"], project_name)
                self.assertEqual(data["tempo"], tempo)
                self.assertEqual(tuple(data["time_signature"]), time_signature)
            print("Project creation test complete", file=sys.stderr)
        except Exception as e:
            print(f"Error in test: {e}", file=sys.stderr)
            raise

    def test_project_creation(self):
        """Run the async test in an event loop"""
        asyncio.run(self.async_test_project_creation())

    async def async_test_debug_project_file(self):
        """Test the debug function for project files"""
        try:
            print("\nStarting debug function test...", file=sys.stderr)

            # Test with non-existent project
            print("Testing non-existent project...", file=sys.stderr)
            debug_info = self.server.debug_project_file("nonexistent")
            self.assertFalse(debug_info["file_exists"])
            self.assertFalse(debug_info["current_project"])
            self.assertFalse(debug_info["in_projects_dict"])
            self.assertIsNone(debug_info["file_contents"])
            print("Non-existent project test passed", file=sys.stderr)

            # Create a project and test debug info
            print("Creating test project...", file=sys.stderr)
            project_name = "debug_test"
            self.server.create_project(project_name, 120, (4, 4))

            print("Getting debug info for created project...", file=sys.stderr)
            debug_info = self.server.debug_project_file(project_name)

            # Verify debug info
            self.assertTrue(debug_info["file_exists"])
            self.assertTrue(debug_info["current_project"])
            self.assertTrue(debug_info["in_projects_dict"])
            self.assertGreater(debug_info["file_size"], 0)
            self.assertIsNotNone(debug_info["file_contents"])

            # Verify file contents
            contents = debug_info["file_contents"]
            self.assertEqual(contents["name"], project_name)
            self.assertEqual(contents["tempo"], 120)
            self.assertEqual(tuple(contents["time_signature"]), (4, 4))

            print("Debug function test complete", file=sys.stderr)

        except Exception as e:
            print(f"Error in debug test: {e}", file=sys.stderr)
            raise

    def test_debug_project_file(self):
        """Run the async debug test in an event loop"""
        asyncio.run(self.async_test_debug_project_file())

    def test_track_management(self):
        """Test track creation and management"""
        # Create a project first
        self.server.create_project("test_project", 120, (4, 4))

        # Create a track
        track_name = "piano"
        instrument = 0
        channel = 0
        self.server.create_track(track_name, instrument, channel)

        # Verify track was created
        self.assertIn(track_name, self.server.current_project.tracks)
        track = self.server.current_project.tracks[track_name]
        self.assertEqual(track.name, track_name)
        self.assertEqual(track.instrument, instrument)
        self.assertEqual(track.channel, channel)

        # Test track muting
        self.server.mute_track(track_name)
        self.assertTrue(self.server.current_project.tracks[track_name].is_muted)

        # Test track solo
        self.server.solo_track(track_name)
        self.assertTrue(self.server.current_project.tracks[track_name].is_solo)

        # Test track volume
        volume = 0.8
        self.server.set_track_volume(track_name, volume)
        self.assertEqual(self.server.current_project.tracks[track_name].volume, volume)

        # Test track pan
        pan = 0.5
        self.server.set_track_pan(track_name, pan)
        self.assertEqual(self.server.current_project.tracks[track_name].pan, pan)

    def test_project_persistence(self):
        """Test saving and loading projects"""
        # Create and save a project
        project_name = "persistence_test"
        self.server.create_project(project_name, 120, (4, 4))
        self.server.create_track("piano", 0, 0)

        # Save the project
        self.server.save_project()

        # Create a new server instance to test loading
        new_server = MidiCompositionServer()
        new_server.workspace_dir = self.server.workspace_dir

        # Load the project
        self.assertTrue(new_server.load_project(project_name))

        # Verify loaded state
        self.assertEqual(new_server.current_project.name, project_name)
        self.assertEqual(new_server.current_project.tempo, 120)
        self.assertEqual(new_server.current_project.time_signature, (4, 4))
        self.assertIn("piano", new_server.current_project.tracks)

        # Clean up the new server
        new_server.stop_midi_server()

    def test_synth_settings(self):
        """Test FluidSynth settings management"""
        # Test gain setting
        gain = 1.0  # Default gain value
        self.assertEqual(self.server.synth_settings["gain"], gain)

        # Set new gain value
        new_gain = 0.8
        self.server.set_gain(new_gain)
        self.assertEqual(self.server.synth_settings["gain"], new_gain)

        # Test reverb settings
        reverb_params = {"room_size": 0.5, "damping": 0.2, "width": 0.7, "level": 0.8}
        self.server.set_reverb(**reverb_params)
        self.assertEqual(self.server.synth_settings["reverb"], reverb_params)

        # Test chorus settings
        chorus_params = {"nr": 4, "level": 1.5, "speed": 0.4, "depth": 10.0, "type": 1}
        self.server.set_chorus(**chorus_params)
        self.assertEqual(self.server.synth_settings["chorus"], chorus_params)


@pytest.fixture
def server():
    """Create a fresh server instance for each test."""
    server = MidiCompositionServer()
    # Clean up workspace directory before each test
    if server.workspace_dir.exists():
        for file in server.workspace_dir.glob("*.json"):
            file.unlink()
    return server


def test_create_project(server):
    """Test project creation functionality."""
    # Create a new project
    success = server.create_project("test_project", 120, (4, 4))
    assert success is True

    # Verify project was created
    assert server.current_project is not None
    assert server.current_project.name == "test_project"
    assert server.current_project.tempo == 120
    assert server.current_project.time_signature == (4, 4)

    # Verify project file was created
    project_file = server.workspace_dir / "test_project.json"
    assert project_file.exists()

    # Verify file contents
    with open(project_file) as f:
        data = json.load(f)
        assert data["name"] == "test_project"
        assert data["tempo"] == 120
        assert data["time_signature"] == [4, 4]
        assert data["tracks"] == {}


def test_create_track(server):
    """Test track creation functionality."""
    # Create project first
    server.create_project("test_project", 120, (4, 4))

    # Create a track
    success = server.create_track("piano", 0, 0)
    assert success is True

    # Verify track was created
    assert "piano" in server.current_project.tracks
    track = server.current_project.tracks["piano"]
    assert track.name == "piano"
    assert track.instrument == 0
    assert track.channel == 0

    # Verify track was saved
    project_file = server.workspace_dir / "test_project.json"
    with open(project_file) as f:
        data = json.load(f)
        assert "piano" in data["tracks"]
        track_data = data["tracks"]["piano"]
        assert track_data["name"] == "piano"
        assert track_data["instrument"] == 0
        assert track_data["channel"] == 0


def test_add_note_event(server):
    """Test adding note events to a track."""
    # Create project and track
    server.create_project("test_project", 120, (4, 4))
    server.create_track("piano", 0, 0)

    # Add a note event
    track = server.current_project.tracks["piano"]
    track.events.append({"type": "note", "note": 60, "velocity": 100, "time": 0})

    # Verify note event was added
    assert len(track.events) == 1
    event = track.events[0]
    assert event["type"] == "note"
    assert event["note"] == 60
    assert event["velocity"] == 100
    assert event["time"] == 0

    # Verify event was saved
    server._save_state()
    project_file = server.workspace_dir / "test_project.json"
    with open(project_file) as f:
        data = json.load(f)
        track_data = data["tracks"]["piano"]
        assert len(track_data["events"]) == 1
        saved_event = track_data["events"][0]
        assert saved_event["type"] == "note"
        assert saved_event["note"] == 60
        assert saved_event["velocity"] == 100
        assert saved_event["time"] == 0


def test_export_midi(server):
    """Test MIDI export functionality."""
    # Create project and track
    server.create_project("test_project", 120, (4, 4))
    server.create_track("piano", 0, 0)

    # Add some note events
    track = server.current_project.tracks["piano"]
    track.events.extend(
        [
            {"type": "note", "note": 60, "velocity": 100, "time": 0},  # Middle C
            {"type": "note", "note": 64, "velocity": 100, "time": 480},  # E4
            {"type": "note", "note": 67, "velocity": 100, "time": 960},  # G4
        ]
    )

    # Export to MIDI
    midi_path = "test_output.mid"
    success = server.export_midi(midi_path)
    assert success is True

    # Verify MIDI file was created
    assert os.path.exists(midi_path)

    # Clean up
    os.remove(midi_path)


def test_project_state_serialization():
    """Test ProjectState serialization and deserialization."""
    # Create a project state
    tracks = {
        "piano": TrackState(
            name="piano",
            instrument=0,
            channel=0,
            events=[{"type": "note", "note": 60, "velocity": 100, "time": 0}],
        )
    }
    project = ProjectState(
        name="test_project", tempo=120, time_signature=(4, 4), tracks=tracks
    )

    # Convert to dict
    data = project.to_dict()

    # Convert back to ProjectState
    new_project = ProjectState.from_dict(data)

    # Verify data was preserved
    assert new_project.name == project.name
    assert new_project.tempo == project.tempo
    assert new_project.time_signature == project.time_signature
    assert len(new_project.tracks) == len(project.tracks)
    assert "piano" in new_project.tracks
    assert len(new_project.tracks["piano"].events) == 1
    assert new_project.tracks["piano"].events[0]["note"] == 60


def test_inspect_projects(server):
    """Test project inspection functionality."""
    # Create multiple test projects
    server.create_project("project1", 120, (4, 4))
    server.create_track("piano", 0, 0)

    # Add some note events to project1
    track = server.current_project.tracks["piano"]
    track.events.append({"type": "note", "note": 60, "velocity": 100, "time": 0})
    server._save_state()

    # Create second project
    server.create_project("project2", 140, (3, 4))
    server.create_track("guitar", 24, 1)
    server._save_state()

    # Test inspection
    result = server.inspect_projects()
    assert result["success"] is True
    data = result["data"]

    # Verify project count
    assert data["project_count"] == 2

    # Verify project1 details
    project1 = data["projects"]["project1"]
    assert project1["tempo"] == 120
    assert project1["time_signature"] == [4, 4]
    assert "piano" in project1["tracks"]
    assert project1["tracks"]["piano"]["instrument"] == 0
    assert project1["tracks"]["piano"]["event_count"] == 1

    # Verify project2 details
    project2 = data["projects"]["project2"]
    assert project2["tempo"] == 140
    assert project2["time_signature"] == [3, 4]
    assert "guitar" in project2["tracks"]
    assert project2["tracks"]["guitar"]["instrument"] == 24
    assert project2["tracks"]["guitar"]["event_count"] == 0


def test_play_audio_file(server):
    """Test audio file playback functionality."""
    # Create a test project and export MIDI
    server.create_project("test_project", 120, (4, 4))
    server.create_track("piano", 0, 0)

    # Add a note event
    track = server.current_project.tracks["piano"]
    track.events.append({"type": "note", "note": 60, "velocity": 100, "time": 0})

    # Export to MIDI
    midi_path = "test_output.mid"
    server.export_midi(midi_path)

    # Test playback
    result = server.play_audio_file("test_project", midi_path)
    assert result["success"] is True

    # Clean up
    os.remove(midi_path)


def test_remove_project(server):
    """Test project removal functionality."""
    # Create a test project
    server.create_project("test_project", 120, (4, 4))
    server.create_track("piano", 0, 0)
    server._save_state()

    # Verify project exists
    project_file = server.workspace_dir / "test_project.json"
    assert project_file.exists()

    # Remove the project
    result = server.remove_project("test_project")
    assert result["success"] is True

    # Verify project is removed
    assert not project_file.exists()
    assert "test_project" not in server.projects
    assert server.current_project is None

    # Test removing non-existent project
    result = server.remove_project("nonexistent")
    assert result["success"] is False
    assert "not found" in result["error"]

    # Test removing project while it's current
    server.create_project("current_project", 120, (4, 4))
    assert server.current_project is not None
    result = server.remove_project("current_project")
    assert result["success"] is True
    assert server.current_project is None


if __name__ == "__main__":
    unittest.main()
