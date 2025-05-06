import os
import json
import logging
from typing import Dict, Any, Optional
import httpx

try:
    from pyfluidsynth.fluidsynth import Synth as FluidSynth

    FLUIDSYNTH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Error importing FluidSynth: {e}")
    logging.warning("FluidSynth features will be disabled.")
    FLUIDSYNTH_AVAILABLE = False
except Exception as e:
    logging.error(f"Unexpected error importing FluidSynth: {e}")
    logging.warning("FluidSynth features will be disabled.")
    FLUIDSYNTH_AVAILABLE = False


class SoundfontManager:
    def __init__(self):
        """Initialize the soundfont manager."""
        self.fs: Optional[FluidSynth] = None
        self.soundfonts: Dict[int, str] = {}  # Map soundfont IDs to their paths
        self.current_sfid: Optional[int] = None

        # Initialize FluidSynth if available
        if FLUIDSYNTH_AVAILABLE:
            try:
                self.fs = FluidSynth()
            except Exception as e:
                logging.error(f"Failed to initialize FluidSynth: {e}")
                self.fs = None

    def find_soundfont(self, search_term: str) -> Dict[str, Any]:
        """Search for a soundfont by name in the soundfontSources.json file.

        Args:
            search_term: The name or partial name of the soundfont to search for

        Returns:
            Dict containing success status and matching soundfont info if found
        """
        try:
            # Load soundfont sources
            with open("soundfontSources.json", "r") as f:
                sources = json.load(f)

            # Search through all soundfonts
            matches = []
            for name, url in sources.items():
                # Check if search term matches (case-insensitive)
                if search_term.lower() in name.lower():
                    matches.append(
                        {
                            "name": name.capitalize(),
                            "url": url,
                            "category": name.capitalize(),
                        }
                    )

            if matches:
                return {
                    "success": True,
                    "data": {"matches": matches, "count": len(matches)},
                }

            return {
                "success": False,
                "error": f"No soundfont found matching '{search_term}'",
            }

        except FileNotFoundError:
            return {"success": False, "error": "soundfontSources.json not found"}
        except json.JSONDecodeError:
            return {"success": False, "error": "Invalid JSON in soundfontSources.json"}
        except Exception as e:
            return {
                "success": False,
                "error": f"Error searching for soundfont: {str(e)}",
            }

    def download_soundfont(self, search_term: str) -> Dict[str, Any]:
        """Search for and download a soundfont by name.

        Args:
            search_term: The name or partial name of the soundfont to search for

        Returns:
            Dict containing success status and either the downloaded file path or error message
        """
        try:
            # First find the soundfont
            result = self.find_soundfont(search_term)
            if not result["success"]:
                return result

            # Get the first match
            match = result["data"]["matches"][0]
            url = match["url"]

            # Create soundfonts directory if it doesn't exist
            os.makedirs("soundfonts", exist_ok=True)

            # Download the file
            response = httpx.get(url)
            if response.status_code == 200:
                # Create filename from URL
                filename = os.path.basename(url)
                filepath = os.path.join("soundfonts", filename)

                # Save the file
                with open(filepath, "wb") as f:
                    f.write(response.content)

                return {
                    "success": True,
                    "data": {
                        "name": match["name"],
                        "category": match["category"],
                        "filepath": filepath,
                    },
                }

            return {
                "success": False,
                "error": f"Failed to download soundfont: HTTP {response.status_code}",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to download soundfont: {str(e)}",
            }

    def add_soundfont(self, soundfont_path: str) -> Dict[str, Any]:
        """Add a new soundfont to FluidSynth.

        Args:
            soundfont_path: Path to the .sf2 soundfont file

        Returns:
            Dict containing success status and soundfont ID if successful
        """
        if not FLUIDSYNTH_AVAILABLE:
            return {"success": False, "error": "FluidSynth is not available"}

        if not os.path.exists(soundfont_path):
            return {
                "success": False,
                "error": f"Soundfont file not found: {soundfont_path}",
            }

        if not soundfont_path.lower().endswith(".sf2"):
            return {"success": False, "error": "File must be a .sf2 soundfont file"}

        try:
            # Load the soundfont
            sfid = self.fs.sfload(soundfont_path)
            if sfid == -1:
                return {"success": False, "error": "Failed to load soundfont"}

            # Store the soundfont ID and path
            self.soundfonts[sfid] = soundfont_path
            self.current_sfid = sfid

            # Set as default soundfont for channel 0
            self.fs.program_select(0, sfid, 0, 0)

            return {
                "success": True,
                "data": {"soundfont_id": sfid, "path": soundfont_path},
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def remove_soundfont(self, soundfont_id: int) -> Dict[str, Any]:
        """Remove a soundfont from FluidSynth.

        Args:
            soundfont_id: ID of the soundfont to remove

        Returns:
            Dict containing success status and error message if failed
        """
        if not FLUIDSYNTH_AVAILABLE:
            return {"success": False, "error": "FluidSynth is not available"}

        if soundfont_id not in self.soundfonts:
            return {"success": False, "error": f"Soundfont {soundfont_id} not found"}

        try:
            self.fs.sfunload(soundfont_id)
            del self.soundfonts[soundfont_id]
            if self.current_sfid == soundfont_id:
                self.current_sfid = None
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_soundfonts(self) -> Dict[str, Any]:
        """List all loaded soundfonts.

        Returns:
            Dict containing success status and list of loaded soundfonts
        """
        if not FLUIDSYNTH_AVAILABLE:
            return {"success": False, "error": "FluidSynth is not available"}

        try:
            soundfonts = []
            for sfid, path in self.soundfonts.items():
                soundfonts.append(
                    {"id": sfid, "name": os.path.basename(path), "path": path}
                )

            return {"success": True, "data": {"soundfonts": soundfonts}}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def cleanup(self):
        """Clean up FluidSynth resources."""
        if self.fs:
            try:
                self.fs.delete()
                logging.info("Successfully cleaned up FluidSynth")
            except Exception as e:
                logging.error(f"Error cleaning up FluidSynth: {str(e)}")
        self.fs = None
        self.soundfonts.clear()
        self.current_sfid = None

    def set_reverb(
        self, room_size: float, damping: float, width: float, level: float
    ) -> None:
        """Configure reverb effects.

        Args:
            room_size: Room size (0.0-1.2)
            damping: Damping (0.0-1.0)
            width: Width (0.0-100.0)
            level: Level (0.0-1.0)
        """
        if self.fs:
            self.fs.set_reverb_roomsize(room_size)
            self.fs.set_reverb_damp(damping)
            self.fs.set_reverb_width(width)
            self.fs.set_reverb_level(level)

    def set_chorus(
        self,
        nr: int = 3,
        level: float = 2.0,
        speed: float = 0.3,
        depth: float = 8.0,
        type: int = 0,
    ):
        """Set chorus parameters.

        Args:
            nr (int): Number of voices in the chorus (default: 3)
            level (float): Chorus level (default: 2.0)
            speed (float): Modulation speed in Hz (default: 0.3)
            depth (float): Modulation depth (default: 8.0)
            type (int): Chorus waveform type (default: 0)
        """
        if not self.fs:
            return

        # Use both settings API and direct method calls
        self.fs.setting("synth.chorus.active", 1)
        self.fs.set_chorus_nr(nr)
        self.fs.set_chorus_level(level)
        self.fs.set_chorus_speed(speed)
        self.fs.set_chorus_depth(depth)
        self.fs.set_chorus_type(type)
