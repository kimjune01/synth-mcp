import unittest
from pathlib import Path
import json
import tempfile
import shutil
import sys
import asyncio
from server import MidiCompositionServer, ProjectState, TrackState


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


if __name__ == "__main__":
    unittest.main()
