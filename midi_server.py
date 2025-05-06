import logging
import sys
import json
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple, TypedDict
import threading
import mido

try:
    from pyfluidsynth.fluidsynth import Synth as FluidSynth

    logging.info("Successfully imported FluidSynth")
    FLUIDSYNTH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Error importing FluidSynth: {e}")
    logging.warning(
        "FluidSynth features will be disabled. Audio export and real-time playback will not be available."
    )
    FLUIDSYNTH_AVAILABLE = False
except Exception as e:
    logging.error(f"Unexpected error importing FluidSynth: {e}")
    logging.warning(
        "FluidSynth features will be disabled. Audio export and real-time playback will not be available."
    )
    FLUIDSYNTH_AVAILABLE = False

try:
    from pythonosc import osc_server
    from pythonosc import dispatcher

    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False


# Type definitions
class ProjectInfo(TypedDict):
    name: str
    tempo: int
    time_signature: Tuple[int, int]
    tracks: Dict[str, Any]


class TrackInfo(TypedDict):
    name: str
    instrument: int
    channel: int
    is_muted: bool
    is_solo: bool
    volume: float
    pan: float


class DebugInfo(TypedDict):
    file_exists: bool
    file_path: str
    file_size: int
    file_contents: Optional[Dict[str, Any]]
    current_project: bool
    in_projects_dict: bool
    error: Optional[str]


@dataclass
class TrackState:
    name: str
    instrument: int
    channel: int
    is_muted: bool = False
    is_solo: bool = False
    volume: float = 1.0
    pan: float = 0.0
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProjectState:
    name: str
    tempo: int
    time_signature: Tuple[int, int]
    tracks: Dict[str, TrackState]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectState":
        tracks = {k: TrackState(**v) for k, v in data.get("tracks", {}).items()}
        return cls(
            name=data["name"],
            tempo=data["tempo"],
            time_signature=tuple(data["time_signature"]),
            tracks=tracks,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tempo": self.tempo,
            "time_signature": self.time_signature,
            "tracks": {k: asdict(v) for k, v in self.tracks.items()},
        }


class MidiCompositionServer:
    def __init__(self) -> None:
        # FluidSynth state
        self.fs: Optional[FluidSynth] = None if FLUIDSYNTH_AVAILABLE else None
        self.sfid: Optional[int] = None
        self.synth_settings: Dict[str, Any] = {
            "gain": 1.0,
            "reverb": {"room_size": 0.2, "damping": 0.0, "width": 0.5, "level": 0.9},
            "chorus": {"nr": 3, "level": 2.0, "speed": 0.3, "depth": 8.0, "type": 0},
        }

        # Project state
        self.current_project: Optional[ProjectState] = None
        self.projects: Dict[str, ProjectState] = {}

        # Real-time collaboration state
        self.connected_clients: set[str] = set()
        self.state_lock: threading.Lock = threading.Lock()

        # Persistence settings
        self.workspace_dir: Path = Path("workspace")
        self.workspace_dir.mkdir(exist_ok=True)

    def _save_state(self) -> None:
        """Save current state to disk"""
        if not self.current_project:
            return

        try:
            state_file = self.workspace_dir / f"{self.current_project.name}.json"
            with open(state_file, "w") as f:
                json.dump(self.current_project.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving state: {e}")

    def _load_state(self, project_name: str) -> bool:
        """Load state from disk"""
        try:
            state_file = self.workspace_dir / f"{project_name}.json"
            if not state_file.exists():
                return False

            with open(state_file) as f:
                data = json.load(f)
                project = ProjectState.from_dict(data)
                self.current_project = project
                self.projects[project_name] = project
                return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False

    def initialize_fluidsynth(self) -> None:
        """Initialize FluidSynth with default settings"""
        try:
            from pyfluidsynth.fluidsynth import Synth

            self.fs = Synth()
            self.fs.start()

            # Load default soundfont
            soundfont_path = os.path.join(
                os.path.dirname(__file__), "Mario_Party_DS_HQ.sf2"
            )
            if os.path.exists(soundfont_path):
                sfid = self.fs.sfload(soundfont_path)
                if sfid != -1:
                    self.fs.program_select(0, sfid, 0, 0)
                    logging.info("Successfully loaded default soundfont")
                else:
                    logging.warning("Failed to load default soundfont")
            else:
                logging.warning(f"Default soundfont not found at {soundfont_path}")

            # Apply initial settings
            self.fs.setting("synth.gain", self.synth_settings["gain"])
            reverb = self.synth_settings["reverb"]
            self.set_reverb(
                reverb["room_size"], reverb["damping"], reverb["width"], reverb["level"]
            )
            chorus = self.synth_settings["chorus"]
            self.set_chorus(
                chorus["nr"],
                chorus["level"],
                chorus["speed"],
                chorus["depth"],
                chorus["type"],
            )
            logging.info("Successfully initialized FluidSynth")
        except ImportError:
            logging.warning(
                "FluidSynth not available - audio features will be disabled"
            )
            self.fs = None
        except Exception as e:
            logging.error(f"Error initializing FluidSynth: {str(e)}")
            self.fs = None

    def load_soundfont(self, soundfont_path: str) -> bool:
        """Load SoundFont files (.sf2) for instrument sounds"""
        with self.state_lock:
            if self.fs and Path(soundfont_path).exists():
                self.sfid = self.fs.sfload(soundfont_path)
                self.fs.sfont_select(0, self.sfid)
                return True
            return False

    def set_gain(self, gain_value: float) -> None:
        """Control the overall volume of the synth"""
        with self.state_lock:
            self.synth_settings["gain"] = gain_value
            if self.fs:
                self.fs.setting("synth.gain", gain_value)

    def set_reverb(
        self, room_size: float, damping: float, width: float, level: float
    ) -> None:
        """Configure reverb effects"""
        with self.state_lock:
            self.synth_settings["reverb"] = {
                "room_size": room_size,
                "damping": damping,
                "width": width,
                "level": level,
            }
            if self.fs:
                self.fs.set_reverb_roomsize(room_size)
                self.fs.set_reverb_damp(damping)
                self.fs.set_reverb_width(width)
                self.fs.set_reverb_level(level)

    def set_chorus(
        self, nr: int, level: float, speed: float, depth: float, type: int
    ) -> None:
        """Configure chorus effects"""
        with self.state_lock:
            self.synth_settings["chorus"] = {
                "nr": nr,
                "level": level,
                "speed": speed,
                "depth": depth,
                "type": type,
            }
            if self.fs:
                self.fs.set_chorus_nr(nr)
                self.fs.set_chorus_level(level)
                self.fs.set_chorus_speed(speed)
                self.fs.set_chorus_depth(depth)
                self.fs.set_chorus_type(type)

    def play_note(self, note: int, velocity: int, channel: int = 0) -> None:
        """Play a single note"""
        with self.state_lock:
            if self.fs:
                self.fs.noteon(channel, note, velocity)

    def stop_note(self, note: int, channel: int = 0) -> None:
        """Stop a single note"""
        with self.state_lock:
            if self.fs:
                self.fs.noteoff(channel, note)

    def program_change(self, program: int, channel: int = 0) -> None:
        """Change the instrument program"""
        with self.state_lock:
            if self.fs:
                self.fs.program_change(channel, program)

    def control_change(self, control: int, value: int, channel: int = 0) -> None:
        """Send a control change message"""
        with self.state_lock:
            if self.fs:
                self.fs.cc(channel, control, value)

    def pitch_bend(self, value: int, channel: int = 0) -> None:
        """Send a pitch bend message"""
        with self.state_lock:
            if self.fs:
                self.fs.pitch_bend(channel, value)

    def all_notes_off(self, channel: int = 0) -> None:
        """Stop all notes on a channel"""
        with self.state_lock:
            if self.fs:
                self.fs.all_notes_off(channel)

    def play_audio_file(self, project_name: str, audio_path: str) -> Dict[str, Any]:
        """Play an audio file using afplay."""
        try:
            import subprocess

            subprocess.run(["afplay", audio_path])
            return {"success": True}
        except Exception as e:
            logging.error(f"Error playing audio file: {e}")
            return {"success": False, "error": str(e)}

    def create_track(self, name: str, instrument: int, channel: int) -> bool:
        """Create a new track with specified instrument"""
        with self.state_lock:
            if not self.current_project:
                return False
            track = TrackState(name=name, instrument=instrument, channel=channel)
            self.current_project.tracks[name] = track
            self._save_state()
            return True

    def mute_track(self, track_id: str) -> bool:
        """Silence a specific track"""
        with self.state_lock:
            if not self.current_project or track_id not in self.current_project.tracks:
                return False
            self.current_project.tracks[track_id].is_muted = True
            self._save_state()
            return True

    def solo_track(self, track_id: str) -> bool:
        """Solo a specific track"""
        with self.state_lock:
            if not self.current_project or track_id not in self.current_project.tracks:
                return False
            self.current_project.tracks[track_id].is_solo = True
            self._save_state()
            return True

    def set_track_volume(self, track_id: str, volume: float) -> bool:
        """Adjust volume for individual tracks"""
        with self.state_lock:
            if not self.current_project or track_id not in self.current_project.tracks:
                return False
            self.current_project.tracks[track_id].volume = volume
            self._save_state()
            return True

    def set_track_pan(self, track_id: str, pan: float) -> bool:
        """Adjust stereo positioning"""
        with self.state_lock:
            if not self.current_project or track_id not in self.current_project.tracks:
                return False
            self.current_project.tracks[track_id].pan = pan
            self._save_state()
            return True

    def create_project(
        self, name: str, tempo: int, time_signature: Tuple[int, int]
    ) -> bool:
        """Initialize a new composition project"""
        try:
            project = ProjectState(
                name=name, tempo=tempo, time_signature=time_signature, tracks={}
            )
            self.current_project = project
            self.projects[name] = project
            self._save_state()
            return True
        except Exception as e:
            print(f"Error creating project: {e}")
            return False

    def save_project(self, path: Optional[str] = None) -> bool:
        """Save the current project state"""
        if not self.current_project:
            return False

        with self.state_lock:
            self._save_state()
            if path:
                # Also save as MIDI if path provided
                self.export_midi(path)
            return True

    def load_project(self, name: str) -> bool:
        """Load a saved project"""
        return self._load_state(name)

    def remove_project(self, project_name: str) -> Dict[str, Any]:
        """Remove a project from the workspace."""
        try:
            # Get project file path
            project_file = self.workspace_dir / f"{project_name}.json"

            # Check if project exists
            if not project_file.exists():
                return {
                    "success": False,
                    "error": f"Project '{project_name}' not found",
                }

            # Remove from current project if it's the one being deleted
            if self.current_project and self.current_project.name == project_name:
                self.current_project = None

            # Remove from projects dict
            if project_name in self.projects:
                del self.projects[project_name]

            # Delete the project file
            project_file.unlink()

            return {"success": True}

        except Exception as e:
            logging.error(f"Error removing project {project_name}: {e}")
            return {"success": False, "error": str(e)}

    def export_midi(self, path: str) -> bool:
        """Export the composition as a standard MIDI file"""
        if not self.current_project:
            return False

        with self.state_lock:
            # Create a new MIDI file
            mid = mido.MidiFile()

            # Add tracks for each track in the project
            for track_name, track in self.current_project.tracks.items():
                midi_track = mido.MidiTrack()
                mid.tracks.append(midi_track)

                # Add track events
                for event in track.events:
                    if event["type"] == "note":
                        midi_track.append(
                            mido.Message(
                                "note_on",
                                note=event["note"],
                                velocity=event["velocity"],
                                time=event.get("time", 0),
                            )
                        )

            # Save the MIDI file
            mid.save(path)
            return True

    def export_audio(self, path: str, format: str) -> bool:
        """Render the composition to audio using FluidSynth"""
        if not self.current_project:
            return False

        if not FLUIDSYNTH_AVAILABLE:
            return False

        try:
            # Initialize FluidSynth if not already done
            if not self.fs:
                self.initialize_fluidsynth()

            # First export to MIDI
            midi_path = path.replace(f".{format}", ".mid")
            if not self.export_midi(midi_path):
                return False

            # Use FluidSynth to convert MIDI to WAV
            if format.lower() == "wav":
                cmd = [
                    "fluidsynth",
                    "-ni",
                    "-F",
                    path,
                    "-r",
                    "44100",
                    "-g",
                    "1.0",
                    "Mario_Party_DS_HQ.sf2",
                    midi_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    return False
                os.remove(midi_path)
                return True
            return False
        except:
            return False

    def start_midi_server(self, port: int) -> None:
        """Start a server that listens for MIDI events"""
        if not OSC_AVAILABLE:
            logging.warning("OSC server not available")
            return

        try:
            dispatcher_obj = dispatcher.Dispatcher()
            dispatcher_obj.map("/midi/note", self._handle_midi_note)
            dispatcher_obj.map("/midi/control", self._handle_midi_control)

            self.server = osc_server.ThreadingOSCUDPServer(
                ("127.0.0.1", port), dispatcher_obj
            )
            logging.info(f"Serving on {self.server.server_address}")
            self.server.serve_forever()
        except OSError as e:
            logging.error(f"Failed to start OSC server: {e}")
        except Exception as e:
            logging.error(f"Unexpected error starting OSC server: {e}")

    def stop_midi_server(self) -> None:
        """Stop the MIDI server"""
        try:
            if hasattr(self, "server"):
                self.server.shutdown()
                self.server.server_close()
                logging.info("OSC server stopped")

            # Clean up FluidSynth
            self.cleanup()
        except Exception as e:
            logging.error(f"Error stopping server: {e}")

    def __del__(self) -> None:
        """Cleanup when the server is destroyed"""
        try:
            self.stop_midi_server()
        except Exception as e:
            print(f"Error in cleanup: {e}")

    def _handle_midi_note(self, address: str, *args: Any) -> None:
        """Handle incoming MIDI note messages"""
        with self.state_lock:
            if not self.current_project:
                return
            # Process MIDI note message and update state
            track_name, note, velocity, time = args
            if track_name in self.current_project.tracks:
                self.current_project.tracks[track_name].events.append(
                    {"type": "note", "note": note, "velocity": velocity, "time": time}
                )

    def _handle_midi_control(self, address: str, *args: Any) -> None:
        """Handle incoming MIDI control messages"""
        with self.state_lock:
            if not self.current_project:
                return
            # Process MIDI control message and update state
            track_name, control, value = args
            if track_name in self.current_project.tracks:
                self.current_project.tracks[track_name].events.append(
                    {"type": "control", "control": control, "value": value}
                )

    def debug_project_file(self, project_name: str) -> Dict[str, Any]:
        """Debug function to inspect a project file and its contents."""
        try:
            debug_info = {
                "file_exists": False,
                "file_path": "",
                "file_size": 0,
                "file_contents": None,
                "current_project": False,
                "in_projects_dict": False,
                "error": None,
            }

            # Get project file path
            project_file = self.workspace_dir / f"{project_name}.json"
            debug_info["file_path"] = str(project_file)

            # Check if file exists and get size
            if project_file.exists():
                debug_info["file_exists"] = True
                debug_info["file_size"] = project_file.stat().st_size

                # Try to read and parse file contents
                try:
                    with open(project_file) as f:
                        debug_info["file_contents"] = json.load(f)
                except json.JSONDecodeError as e:
                    debug_info["error"] = f"JSON decode error: {str(e)}"
                except Exception as e:
                    debug_info["error"] = f"Error reading file: {str(e)}"

            # Check project state
            debug_info["current_project"] = (
                self.current_project is not None
                and self.current_project.name == project_name
            )
            debug_info["in_projects_dict"] = project_name in self.projects

            return debug_info

        except Exception as e:
            return {
                "file_exists": False,
                "file_path": "",
                "file_size": 0,
                "file_contents": None,
                "current_project": False,
                "in_projects_dict": False,
                "error": f"Debug function error: {str(e)}",
            }

    def inspect_projects(self) -> Dict[str, Any]:
        """Inspect all projects in the workspace directory."""
        projects = {}

        try:
            # Get all JSON files in the workspace directory
            project_files = list(self.workspace_dir.glob("*.json"))

            for project_file in project_files:
                try:
                    with open(project_file) as f:
                        data = json.load(f)
                        project_name = project_file.stem
                        projects[project_name] = {
                            "name": data.get("name", project_name),
                            "tempo": data.get("tempo", 120),
                            "time_signature": data.get("time_signature", [4, 4]),
                            "tracks": {
                                track_name: {
                                    "instrument": track_data.get("instrument", 0),
                                    "channel": track_data.get("channel", 0),
                                    "is_muted": track_data.get("is_muted", False),
                                    "is_solo": track_data.get("is_solo", False),
                                    "volume": track_data.get("volume", 1.0),
                                    "pan": track_data.get("pan", 0.0),
                                    "event_count": len(track_data.get("events", [])),
                                }
                                for track_name, track_data in data.get(
                                    "tracks", {}
                                ).items()
                            },
                        }
                except json.JSONDecodeError as e:
                    logging.error(f"Error reading project file {project_file}: {e}")
                    continue
                except Exception as e:
                    logging.error(f"Unexpected error processing {project_file}: {e}")
                    continue

            return {
                "success": True,
                "data": {"project_count": len(projects), "projects": projects},
            }
        except Exception as e:
            logging.error(f"Error inspecting projects: {e}")
            return {"success": False, "error": str(e)}

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.fs:
            try:
                self.fs.delete()
                logging.info("Successfully cleaned up FluidSynth")
            except Exception as e:
                logging.error(f"Error cleaning up FluidSynth: {str(e)}")
        self.fs = None
