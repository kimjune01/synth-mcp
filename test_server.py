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
from midi_server import MidiCompositionServer, FLUIDSYNTH_AVAILABLE
from soundfont_manager import SoundfontManager, FLUIDSYNTH_AVAILABLE
import httpx
import mido


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
        assert self.server.create_project("test_project", 120, (4, 4))["success"]

        # Create a track
        track_name = "piano"
        instrument = "piano"
        channel = 0
        result = self.server.create_track(track_name, instrument, channel)
        assert result["success"]

        # Verify track was created
        self.assertIn(track_name, self.server.current_project.tracks)
        track = self.server.current_project.tracks[track_name]
        self.assertEqual(track.name, track_name)
        self.assertEqual(track.instrument, instrument)
        self.assertEqual(track.channel, channel)

        # Test track muting
        assert self.server.mute_track(track_name)["success"]
        self.assertTrue(self.server.current_project.tracks[track_name].is_muted)

        # Test track solo
        assert self.server.solo_track(track_name)["success"]
        self.assertTrue(self.server.current_project.tracks[track_name].is_solo)

        # Test track volume
        volume = 0.8
        assert self.server.set_track_volume(track_name, volume)["success"]
        self.assertEqual(self.server.current_project.tracks[track_name].volume, volume)

        # Test track pan
        pan = 0.5
        assert self.server.set_track_pan(track_name, pan)["success"]
        self.assertEqual(self.server.current_project.tracks[track_name].pan, pan)

    def test_project_persistence(self):
        """Test saving and loading projects"""
        # Create and save a project
        project_name = "persistence_test"
        assert self.server.create_project(project_name, 120, (4, 4))["success"]
        assert self.server.create_track("piano", "piano", 0)["success"]

        # Save the project
        assert self.server.save_project()["success"]

        # Create a new server instance to test loading
        new_server = MidiCompositionServer()
        new_server.workspace_dir = self.server.workspace_dir

        # Load the project
        assert new_server.load_project(project_name)["success"]

        # Verify loaded state
        self.assertEqual(new_server.current_project.name, project_name)
        self.assertEqual(new_server.current_project.tempo, 120)
        self.assertEqual(new_server.current_project.time_signature, (4, 4))
        self.assertIn("piano", new_server.current_project.tracks)

    def test_synth_settings(self):
        """Test FluidSynth settings management"""
        # Skip test if FluidSynth is not available
        if not FLUIDSYNTH_AVAILABLE:
            pytest.skip("FluidSynth is not available")

        # Test gain setting
        gain = 1.0  # Default gain value
        self.server.soundfont_manager.fs.setting("synth.gain", gain)
        self.assertEqual(
            self.server.soundfont_manager.fs.get_setting("synth.gain"), gain
        )

        # Test reverb settings
        reverb_params = {"room_size": 0.5, "damping": 0.2, "width": 0.7, "level": 0.8}
        self.server.soundfont_manager.set_reverb(**reverb_params)
        self.assertEqual(
            self.server.soundfont_manager.fs.get_reverb_roomsize(),
            reverb_params["room_size"],
        )
        self.assertEqual(
            self.server.soundfont_manager.fs.get_reverb_damp(), reverb_params["damping"]
        )
        self.assertEqual(
            self.server.soundfont_manager.fs.get_reverb_width(), reverb_params["width"]
        )
        self.assertEqual(
            self.server.soundfont_manager.fs.get_reverb_level(), reverb_params["level"]
        )

        # Test chorus settings
        chorus_params = {"nr": 4, "level": 0.8, "speed": 0.4, "depth": 10.0, "type": 0}
        self.server.soundfont_manager.set_chorus(**chorus_params)
        self.assertEqual(
            self.server.soundfont_manager.fs.get_chorus_nr(), chorus_params["nr"]
        )
        self.assertEqual(
            self.server.soundfont_manager.fs.get_chorus_level(), chorus_params["level"]
        )
        self.assertEqual(
            self.server.soundfont_manager.fs.get_chorus_speed(), chorus_params["speed"]
        )
        self.assertEqual(
            self.server.soundfont_manager.fs.get_chorus_depth(), chorus_params["depth"]
        )
        self.assertEqual(
            self.server.soundfont_manager.fs.get_chorus_type(), chorus_params["type"]
        )


@pytest.fixture
def soundfont_manager():
    """Create a fresh soundfont manager for each test."""
    manager = SoundfontManager()
    yield manager
    manager.cleanup()


@pytest.fixture
def server():
    """Create a fresh server instance for each test."""
    server = MidiCompositionServer()
    # Clean up workspace directory before each test
    if server.workspace_dir.exists():
        for file in server.workspace_dir.glob("*.json"):
            file.unlink()
    yield server
    # Cleanup after tests
    if server.current_project:
        server.remove_project(server.current_project.name)
    server.cleanup()


def test_create_project(server):
    """Test project creation functionality."""
    assert server.create_project("test_project", 120, (4, 4))
    assert server.current_project is not None
    assert server.current_project.name == "test_project"
    assert server.current_project.tempo == 120
    assert server.current_project.time_signature == (4, 4)


def test_create_track(server):
    """Test track creation functionality."""
    assert server.create_project("test_project", 120, (4, 4))["success"]
    result = server.create_track("track1", "piano", 0)
    assert result["success"]
    assert "track1" in server.current_project.tracks
    track = server.current_project.tracks["track1"]
    assert track.instrument == "piano"
    assert track.channel == 0


def test_mute_track(server):
    server.create_project("test_project", 120, (4, 4))
    assert server.create_track("track1", "piano", 0)["success"]
    result = server.mute_track("track1")
    assert result["success"]
    assert server.current_project.tracks["track1"].is_muted


def test_solo_track(server):
    server.create_project("test_project", 120, (4, 4))
    assert server.create_track("track1", "piano", 0)["success"]
    result = server.solo_track("track1")
    assert result["success"]
    assert server.current_project.tracks["track1"].is_solo


def test_set_track_volume(server):
    server.create_project("test_project", 120, (4, 4))
    assert server.create_track("track1", "piano", 0)["success"]
    result = server.set_track_volume("track1", 0.5)
    assert result["success"]
    assert server.current_project.tracks["track1"].volume == 0.5


def test_set_track_pan(server):
    server.create_project("test_project", 120, (4, 4))
    assert server.create_track("track1", "piano", 0)["success"]
    result = server.set_track_pan("track1", -0.5)
    assert result["success"]
    assert server.current_project.tracks["track1"].pan == -0.5


def test_save_and_load_project(server):
    # Create and save a project
    assert server.create_project("test_project", 120, (4, 4))["success"]
    assert server.create_track("track1", "piano", 0)["success"]
    assert server.set_track_volume("track1", 0.5)["success"]
    assert server.save_project()["success"]

    # Create a new server instance and load the project
    new_server = MidiCompositionServer()
    assert new_server.load_project("test_project")["success"]
    assert new_server.current_project is not None
    assert new_server.current_project.name == "test_project"
    assert "track1" in new_server.current_project.tracks
    assert new_server.current_project.tracks["track1"].volume == 0.5


def test_export_midi(server, tmp_path):
    server.create_project("test_project", 120, (4, 4))
    server.create_track("track1", "piano", 0)
    midi_path = tmp_path / "test.mid"
    assert server.export_midi(str(midi_path))
    assert midi_path.exists()


def test_export_audio(server, tmp_path):
    server.create_project("test_project", 120, (4, 4))
    server.create_track("track1", "piano", 0)
    audio_path = tmp_path / "test.wav"
    assert server.export_audio(str(audio_path), "wav")
    assert audio_path.exists()


def test_play_audio_file(server, tmp_path):
    # Create a test audio file
    audio_path = tmp_path / "test.wav"

    # Create a simple WAV file with 1 second of silence
    import wave
    import struct

    with wave.open(str(audio_path), "w") as wav_file:
        # Set parameters
        nchannels = 1
        sampwidth = 2
        framerate = 44100
        nframes = framerate
        comptype = "NONE"
        compname = "not compressed"

        wav_file.setparams(
            (nchannels, sampwidth, framerate, nframes, comptype, compname)
        )

        # Write 1 second of silence
        for _ in range(framerate):
            value = struct.pack("h", 0)
            wav_file.writeframes(value)

    # Test playing the audio file
    result = server.play_audio_file(str(audio_path))
    assert result["success"]

    # Test playing non-existent file
    result = server.play_audio_file("nonexistent.wav")
    assert not result["success"]
    assert "error" in result
    assert "File not found" in result["error"]


def test_play_audio_file_mcp(server, tmp_path):
    """Test the MCP play_audio_file function."""
    # Create a test audio file
    audio_path = tmp_path / "test.wav"

    # Create a simple WAV file with 1 second of silence
    import wave
    import struct

    with wave.open(str(audio_path), "w") as wav_file:
        # Set parameters
        nchannels = 1
        sampwidth = 2
        framerate = 44100
        nframes = framerate
        comptype = "NONE"
        compname = "not compressed"

        wav_file.setparams(
            (nchannels, sampwidth, framerate, nframes, comptype, compname)
        )

        # Write 1 second of silence
        for _ in range(framerate):
            value = struct.pack("h", 0)
            wav_file.writeframes(value)

    # Test playing the audio file
    result = server.play_audio_file(str(audio_path))
    assert result["success"]

    # Test playing non-existent file
    result = server.play_audio_file("nonexistent.wav")
    assert not result["success"]
    assert "error" in result
    assert "File not found" in result["error"]


def test_remove_project(server):
    # Create a test project
    server.create_project("test_project", 120, (4, 4))
    server.create_track("track1", "piano", 0)
    server.save_project()

    # Test removing the project
    result = server.remove_project("test_project")
    assert result["success"]
    assert not server.workspace_dir.joinpath("test_project.json").exists()
    assert server.current_project is None

    # Test removing non-existent project
    result = server.remove_project("non_existent_project")
    assert not result["success"]
    assert "error" in result


def test_inspect_projects(server):
    # Create a test project
    assert server.create_project("test_project", 120, (4, 4))["success"]
    assert server.create_track("track1", "piano", 0)["success"]
    assert server.save_project()["success"]

    # Test inspecting projects
    result = server.inspect_projects()
    assert result["success"]
    assert "data" in result
    assert "projects" in result["data"]
    assert "test_project" in result["data"]["projects"]
    assert (
        result["data"]["projects"]["test_project"]["tracks"]["track1"]["instrument"]
        == "piano"
    )


def test_add_notes(server):
    """Test adding multiple notes to a track."""
    # Create a project and track
    assert server.create_project("test_project", 120, (4, 4))["success"]
    assert server.create_track("track1", "piano", 0)["success"]

    # Test adding valid notes
    notes = [
        {"note": 60, "velocity": 100, "time": 0},  # Middle C
        {"note": 64, "velocity": 100, "time": 480},  # E
        {"note": 67, "velocity": 100, "time": 960},  # G
    ]
    result = server.add_notes("track1", notes)
    assert result["success"]

    # Verify notes were added
    track = server.current_project.tracks["track1"]
    assert len(track.events) == 3
    assert track.events[0]["note"] == 60
    assert track.events[1]["note"] == 64
    assert track.events[2]["note"] == 67

    # Test invalid note values
    invalid_notes = [{"note": 128, "velocity": 100}]  # Note > 127
    result = server.add_notes("track1", invalid_notes)
    assert not result["success"]
    assert "must be between 0 and 127" in result["error"]

    # Test invalid velocity values
    invalid_notes = [{"note": 60, "velocity": 128}]  # Velocity > 127
    result = server.add_notes("track1", invalid_notes)
    assert not result["success"]
    assert "must be between 0 and 127" in result["error"]

    # Test missing required fields
    invalid_notes = [{"note": 60}]  # Missing velocity
    result = server.add_notes("track1", invalid_notes)
    assert not result["success"]
    assert "must have 'note' and 'velocity' fields" in result["error"]

    # Test non-existent track
    result = server.add_notes("nonexistent_track", notes)
    assert not result["success"]
    assert "not found" in result["error"]


def test_add_notes_mcp(server):
    """Test the MCP add_notes function."""
    # Create a project and track
    assert server.create_project("test_project", 120, (4, 4))["success"]
    assert server.create_track("track1", "piano", 0)["success"]
    assert server.save_project()["success"]

    # Test adding valid notes
    notes = [
        {"note": 60, "velocity": 100, "time": 0},
        {"note": 64, "velocity": 100, "time": 480},
        {"note": 67, "velocity": 100, "time": 960},
    ]
    result = server.add_notes("track1", notes)
    assert result["success"]

    # Test non-existent project
    result = server.add_notes("nonexistent_track", notes)
    assert not result["success"]
    assert "not found" in result["error"]


def test_soundfont_management(soundfont_manager, tmp_path):
    """Test soundfont management functionality."""
    # Test with non-existent file
    result = soundfont_manager.add_soundfont("nonexistent.sf2")
    assert not result["success"]
    assert "not found" in result["error"]

    # Test with non-sf2 file
    invalid_path = tmp_path / "test.txt"
    with open(invalid_path, "w") as f:
        f.write("not a soundfont")
    result = soundfont_manager.add_soundfont(str(invalid_path))
    assert not result["success"]
    assert "must be a .sf2 soundfont file" in result["error"]

    # Test with invalid sf2 file
    invalid_sf2 = tmp_path / "test.sf2"
    with open(invalid_sf2, "w") as f:
        f.write("invalid sf2 data")
    result = soundfont_manager.add_soundfont(str(invalid_sf2))
    if FLUIDSYNTH_AVAILABLE:
        assert not result["success"]
        assert "Failed to load soundfont" in result["error"]
    else:
        assert not result["success"]
        assert "FluidSynth is not available" in result["error"]

    # Test listing soundfonts (should be empty)
    result = soundfont_manager.list_soundfonts()
    if FLUIDSYNTH_AVAILABLE:
        assert result["success"]
        assert "soundfonts" in result["data"]
        assert len(result["data"]["soundfonts"]) == 0
    else:
        assert not result["success"]
        assert "FluidSynth is not available" in result["error"]

    # Test removing non-existent soundfont
    result = soundfont_manager.remove_soundfont(999)
    assert not result["success"]
    assert "not found" in result["error"]


def test_soundfont_management_mcp(server, tmp_path):
    """Test the MCP soundfont management functions."""
    # Test with non-existent file
    result = server.add_soundfont("nonexistent.sf2")
    assert not result["success"]
    assert "not found" in result["error"]

    # Test with invalid sf2 file
    invalid_sf2 = tmp_path / "test.sf2"
    with open(invalid_sf2, "w") as f:
        f.write("invalid sf2 data")
    result = server.add_soundfont(str(invalid_sf2))
    if FLUIDSYNTH_AVAILABLE:
        assert not result["success"]
        assert "Failed to load soundfont" in result["error"]
    else:
        assert not result["success"]
        assert "FluidSynth is not available" in result["error"]

    # Test listing soundfonts (should be empty)
    result = server.list_soundfonts()
    if FLUIDSYNTH_AVAILABLE:
        assert result["success"]
        assert "soundfonts" in result["data"]
        assert len(result["data"]["soundfonts"]) == 0
    else:
        assert not result["success"]
        assert "FluidSynth is not available" in result["error"]


def test_remove_soundfont_mcp(server):
    """Test removing a soundfont using MCP."""
    result = server.remove_soundfont(1)
    assert not result["success"]
    assert "not found" in result["error"].lower()


def test_find_soundfont(server):
    """Test searching for soundfonts without downloading."""
    # Test finding an existing soundfont
    result = server.find_soundfont("Piano")
    assert result["success"]
    assert len(result["data"]["matches"]) > 0
    for match in result["data"]["matches"]:
        assert "name" in match
        assert "url" in match
        assert "category" in match

    # Test searching for non-existent soundfont
    result = server.find_soundfont("nonexistent")
    assert not result["success"]
    assert "no soundfont found matching" in result["error"].lower()


def test_download_soundfont(server, monkeypatch):
    """Test downloading a soundfont."""

    # Mock the find_soundfont method to return a known soundfont
    def mock_find_soundfont(name):
        return {
            "success": True,
            "data": {
                "matches": [
                    {
                        "name": "Piano",
                        "category": "Piano",
                        "url": "http://example.com/piano.sf2",
                    }
                ]
            },
        }

    monkeypatch.setattr(server.soundfont_manager, "find_soundfont", mock_find_soundfont)

    # Mock the httpx.get function
    class MockResponse:
        status_code = 200
        content = b"mock soundfont data"

    def mock_get(url):
        return MockResponse()

    monkeypatch.setattr(httpx, "get", mock_get)

    # Test downloading a known soundfont
    result = server.download_soundfont("Piano")
    assert result["success"]
    assert "Piano" in result["data"]["name"]
    assert result["data"]["category"] == "Piano"
    assert os.path.exists(result["data"]["filepath"])

    # Clean up downloaded file
    if result["success"]:
        os.remove(result["data"]["filepath"])


def test_create_track_with_soundfont(server, monkeypatch):
    """Test creating a track with automatic soundfont handling."""
    # Create a project first
    assert server.create_project("TestProject", 120, (4, 4))["success"]

    # Mock the find_soundfont and download_soundfont methods
    def mock_find_soundfont(name):
        return {
            "success": True,
            "data": {
                "matches": [
                    {
                        "name": name,
                        "category": name,
                        "url": f"http://example.com/{name}.sf2",
                    }
                ]
            },
        }

    def mock_download_soundfont(name):
        return {
            "success": True,
            "data": {
                "name": name,
                "category": name,
                "filepath": f"soundfonts/{name}.sf2",
            },
        }

    monkeypatch.setattr(server.soundfont_manager, "find_soundfont", mock_find_soundfont)
    monkeypatch.setattr(
        server.soundfont_manager, "download_soundfont", mock_download_soundfont
    )

    # Test creating a track with a piano (program 0)
    result = server.create_track("Piano Track", "piano", 0)
    assert result["success"]
    assert "Piano Track" in server.current_project.tracks
    assert server.current_project.tracks["Piano Track"].instrument == "piano"


def test_create_track_invalid_instrument(server):
    """Test creating a track with an invalid instrument name."""
    # Create a project first
    assert server.create_project("TestProject", 120, (4, 4))["success"]

    # Try to create a track with an invalid instrument name
    result = server.create_track("Invalid Track", "invalid", 0)
    assert not result["success"]
    assert (
        "Could not find or download soundfont for instrument invalid" in result["error"]
    )


def test_create_track_missing_soundfont(server, monkeypatch):
    """Test creating a track when soundfont can't be found or downloaded."""
    # Create a project first
    assert server.create_project("TestProject", 120, (4, 4))["success"]

    # Mock the find_soundfont and download_soundfont methods to simulate failure
    def mock_find_soundfont(name):
        return {"success": False, "error": "Soundfont not found"}

    def mock_download_soundfont(name):
        return {"success": False, "error": "Download failed"}

    monkeypatch.setattr(server.soundfont_manager, "find_soundfont", mock_find_soundfont)
    monkeypatch.setattr(
        server.soundfont_manager, "download_soundfont", mock_download_soundfont
    )

    # Try to create a track with a piano (program 0)
    result = server.create_track("Piano Track", "piano", 0)
    assert not result["success"]
    assert "Could not find or download soundfont" in result["error"]
    assert "Piano Track" not in server.current_project.tracks


def test_create_track_no_project(server):
    """Test creating a track when no project is loaded."""
    result = server.create_track("Test Track", "piano", 0)
    assert not result["success"]
    assert "No project is currently loaded" in result["error"]


def test_create_track_existing_soundfont(server, monkeypatch):
    """Test creating a track when soundfont is already available."""
    # Create a project first
    assert server.create_project("TestProject", 120, (4, 4))["success"]

    # Mock the find_soundfont method to simulate existing soundfont
    def mock_find_soundfont(name):
        return {
            "success": True,
            "data": {
                "matches": [
                    {
                        "name": name,
                        "category": name,
                        "url": f"http://example.com/{name}.sf2",
                    }
                ]
            },
        }

    monkeypatch.setattr(server.soundfont_manager, "find_soundfont", mock_find_soundfont)

    # Create a track with a piano (program 0)
    result = server.create_track("Piano Track", "piano", 0)
    assert result["success"]
    assert "Piano Track" in server.current_project.tracks
    assert server.current_project.tracks["Piano Track"].instrument == "piano"


def test_export_midi_with_strings(server, tmp_path):
    """Test exporting MIDI with string instruments."""
    server.create_project("test_project", 120, (4, 4))
    server.create_track("strings", "strings", 0)

    # Add some notes to the string track
    notes = [
        {"type": "note", "note": 60, "velocity": 64, "time": 0, "duration": 480},
        {"type": "note", "note": 64, "velocity": 64, "time": 480, "duration": 480},
        {"type": "note", "note": 67, "velocity": 64, "time": 960, "duration": 480},
    ]
    server.add_notes("strings", notes)

    midi_path = tmp_path / "test_strings.mid"
    result = server.export_midi(str(midi_path))
    assert result["success"]
    assert midi_path.exists()

    # Read back the MIDI file and verify program number
    midi_file = mido.MidiFile(str(midi_path))
    assert len(midi_file.tracks) == 1

    # Find program change message
    program_msgs = [msg for msg in midi_file.tracks[0] if msg.type == "program_change"]
    assert len(program_msgs) == 1
    assert program_msgs[0].program == 48  # String Ensemble 1

    # Verify notes
    note_on_msgs = [
        msg for msg in midi_file.tracks[0] if msg.type == "note_on" and msg.velocity > 0
    ]
    assert len(note_on_msgs) == 3
    assert note_on_msgs[0].note == 60  # C4
    assert note_on_msgs[1].note == 64  # E4
    assert note_on_msgs[2].note == 67  # G4


def test_get_midi_settings(server):
    """Test getting MIDI settings."""
    result = server.get_midi_settings()
    assert result["success"]
    assert "data" in result
    assert result["data"]["ticks_per_beat"] == 480
    assert result["data"]["max_note_value"] == 127
    assert result["data"]["max_velocity"] == 127
    assert result["data"]["max_channel"] == 15
    assert result["data"]["drum_channel"] == 9


def test_export_midi_simultaneous_notes(server, tmp_path):
    """Test exporting MIDI with simultaneous notes."""
    server.create_project("test_project", 120, (4, 4))
    server.create_track("piano", "piano", 0)

    # Add three simultaneous notes (C major chord)
    notes = [
        {"note": 60, "velocity": 80, "time": 0, "duration": 480},  # C4
        {"note": 64, "velocity": 80, "time": 0, "duration": 480},  # E4
        {"note": 67, "velocity": 80, "time": 0, "duration": 480},  # G4
    ]
    server.add_notes("piano", notes)

    midi_path = tmp_path / "test_simultaneous.mid"
    result = server.export_midi(str(midi_path))
    assert result["success"]
    assert midi_path.exists()

    # Read back the MIDI file and verify notes
    midi_file = mido.MidiFile(str(midi_path))
    assert len(midi_file.tracks) == 1

    # Get all note_on messages
    note_on_msgs = [
        msg for msg in midi_file.tracks[0] if msg.type == "note_on" and msg.velocity > 0
    ]
    assert len(note_on_msgs) == 3

    # Verify that all notes start at time 0
    for msg in note_on_msgs:
        assert msg.time == 0

    # Verify the notes are C4, E4, G4
    notes = sorted(msg.note for msg in note_on_msgs)
    assert notes == [60, 64, 67]  # C4, E4, G4


if __name__ == "__main__":
    pytest.main([__file__])
